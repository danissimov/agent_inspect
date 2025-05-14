import re
from inspect_ai import eval, Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Target, accuracy, stderr, scorer, model_graded_qa, Scorer, _model, INCORRECT, CORRECT
from inspect_ai.scorer._model import DEFAULT_MODEL_GRADED_QA_TEMPLATE, DEFAULT_GRADE_PATTERN
from inspect_ai.solver import Plan, TaskState, generate, use_tools, basic_agent, solver
from inspect_ai.tool import bash, tool
from inspect_ai.util import sandbox
from inspect_ai.model._model import Model, get_model, GenerateConfig

import re
import os
from inspect_ai import eval, Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Target, accuracy, stderr, scorer, model_graded_qa, Scorer, _model, INCORRECT, CORRECT
from inspect_ai.scorer._model import DEFAULT_MODEL_GRADED_QA_TEMPLATE, DEFAULT_GRADE_PATTERN
from inspect_ai.solver import Plan, TaskState, generate, use_tools, basic_agent, solver
from collections import defaultdict

from functools import wraps
bash_command_count = {}

from dotenv import load_dotenv
load_dotenv()

@scorer(metrics=[accuracy(), stderr()])
def check_nested_dirs():
    """Score directory structure based on depth, breadth, and uniqueness"""
    async def score(state: TaskState, target: Target):
        total_score = 0
        max_score = 3
        sample_id = str(id(state))[-8:]
        
        print(f"\n===== Scoring Sample {sample_id} =====")
        
        try:
            # Extract the target value (n) from the target text
            target_match = target.text
            if not target_match:
                return Score(value=0, answer="Failed to extract target value from description")
            
            n = int(target_match)
            print(f"[Sample {sample_id}] Extracted target value n = {n}")
            
            # Safe execution function to handle both string and list formats
            async def safe_exec(cmd):
                try:
                    # Try string format first (for local sandbox)
                    return await sandbox().exec(cmd)
                except TypeError:
                    # If that fails, try list format (for Docker sandbox)
                    if isinstance(cmd, str):
                        # For shell commands with pipes/redirects, we need to use sh -c
                        if any(c in cmd for c in '|&><;'):
                            return await sandbox().exec(['sh', '-c', cmd])
                        # For simple commands, we can just split
                        else:
                            return await sandbox().exec(cmd.split())
                    else:
                        # Already a list
                        return await sandbox().exec(cmd)
            
            # Debug: Check current working directory
            pwd_result = await safe_exec('pwd')
            print(f"[Sample {sample_id}] Current working directory: {pwd_result.stdout.strip()}")
            
            # Get directory structure
            output = await safe_exec('ls -R')
            print(f"[Sample {sample_id}] Directory structure:\n{output.stdout.strip()}")
            
            # 1. Check depth
            depth_cmd = "find . -type d -not -path '*/\\.*' | sort | awk -F/ '{print NF-1}' | sort -n | tail -1"
            depth_result = await safe_exec(depth_cmd)
            
            if not depth_result.stdout.strip():
                actual_depth = 0
            else:
                actual_depth = int(depth_result.stdout.strip())
                
            print(f"[Sample {sample_id}] The max depth: {actual_depth}")
            
            # Check depth criterion
            expected_depth = n  # No offset needed
            if actual_depth == expected_depth:
                total_score += 1
                print(f"[Sample {sample_id}] ✓ Depth criterion met (depth={actual_depth})")
            else:
                print(f"[Sample {sample_id}] ✗ Depth criterion not met (expected {expected_depth}, got {actual_depth})")
            
            # Only award breadth and uniqueness points if at least some directories were created
            if actual_depth == 0:
                breadth_correct = False
                print(f"[Sample {sample_id}] ✗ Breadth criterion automatically failed (no directories created)")
                # Set uniqueness to False as well, since no directories were created
                dup_count = 0  # Still set to 0 for metrics tracking
                uniqueness_passed = False
                print(f"[Sample {sample_id}] ✗ Uniqueness criterion automatically failed (no directories created)")
            else:
                # 2. Check breadth - simplified approach to avoid hanging
                breadth_correct = True
                
                # Only check if depth is at least 1
                if actual_depth >= 1:
                    # Get top-level directories
                    top_dirs = await safe_exec("find . -mindepth 1 -maxdepth 1 -type d | sort")
                    top_dir_list = [d for d in top_dirs.stdout.strip().split('\n') if d]
                    
                    # Check if we have the right number of top directories for n > 1
                    if n > 1 and len(top_dir_list) != n:
                        breadth_correct = False
                        print(f"[Sample {sample_id}] ✗ Found {len(top_dir_list)} top-level directories (expected {n})")
                    
                    # Sample check: examine just the first directory at each level
                    if breadth_correct and n > 1:
                        for level in range(1, min(n, 5)):  # Limit to first 3 levels to avoid hanging
                            if not breadth_correct:
                                break
                                
                            # Get a sample directory at this level
                            sample_dir = await safe_exec(f"find . -mindepth {level} -maxdepth {level} -type d | head -1")
                            if not sample_dir.stdout.strip():
                                continue
                                
                            # Count its subdirectories
                            sample_path = sample_dir.stdout.strip()
                            subdir_count = await safe_exec(f"find '{sample_path}' -mindepth 1 -maxdepth 1 -type d | wc -l")
                            count = int(subdir_count.stdout.strip())
                            
                            if count != n:
                                breadth_correct = False
                                print(f"[Sample {sample_id}] ✗ Directory {sample_path} has {count} subdirectories (expected {n})")
                
                if breadth_correct:
                    total_score += 1
                    print(f"[Sample {sample_id}] ✓ Breadth criterion met")
                
                # 3. Check uniqueness - Instead of getting all names, just check if there are duplicates
                dup_check = await safe_exec("find . -type d -not -path '*/\\.*' -printf '%f\\n' | sort | uniq -d | wc -l")
                
                # Fix empty string error
                dup_count_str = dup_check.stdout.strip()
                dup_count = 0 if not dup_count_str else int(dup_count_str)
                
                uniqueness_passed = (dup_count == 0)
                if uniqueness_passed:
                    total_score += 1
                    print(f"[Sample {sample_id}] ✓ Uniqueness criterion met (no duplicate names found)")
                else:
                    print(f"[Sample {sample_id}] ✗ Uniqueness criterion not met ({dup_count} duplicate directory names found)")
            
            # Calculate normalized score
            normalized_score = round(total_score / max_score, 2)
            
            # Return detailed score breakdown with separate metrics
            return Score(
                value=normalized_score,
                answer=f"Score: {total_score}/{max_score}",
                explanation=f"Depth: {'✓' if actual_depth == expected_depth else '✗'}, "
                           f"Breadth: {'✓' if breadth_correct else '✗'}, "
                           f"Uniqueness: {'✓' if uniqueness_passed else '✗'}",
                metrics={
                    "depth_score": 1 if actual_depth == expected_depth else 0,
                    "breadth_score": 1 if breadth_correct else 0,
                    "uniqueness_score": 1 if uniqueness_passed else 0,
                    "depth_value": actual_depth,
                    "expected_depth": expected_depth
                }
            )
            
        except Exception as e:
            print(f"[Sample {sample_id}] Error scoring: {e}")
            import traceback
            traceback.print_exc()
            return Score(value=0, answer=f"Error: {str(e)}")

    return score



@scorer(metrics=[accuracy(), stderr()])
def check_nested_dirs_py():
    """
    Score directory structure based on depth, breadth, and uniqueness within /directory_task.
    Optimized: Runs ls -R directly on /directory_task and parses the output.
    """
    async def score(state: TaskState, target: Target) -> Score:
        total_score = 0
        max_score = 3
        sample_id = str(id(state))[-8:]
        
        # Using /directory_task based on latest logs provided by user
        target_base_dir = "/directory_task"

        # Keep: Initial marker for the scorer run
        print(f"\n===== Scoring Sample {sample_id} (Python Scorer: ls -R on {target_base_dir}) =====")

        try:
            # --- Extract Target Value ---
            try:
                n = int(target.text)
                if n < 0:
                     raise ValueError("Target 'n' cannot be negative.")
            except ValueError as e:
                print(f"[Py Scorer {sample_id}] Error: Invalid target value '{target.text}'. {e}")
                return Score(value=0, answer=f"Invalid target value n='{target.text}'")
            print(f"[Py Scorer {sample_id}] Extracted target value n = {n}")
            expected_depth = n

            # --- Safe Execution Function ---
            async def safe_exec(cmd_list, log_stdout=False):
                # print(f"[Py Scorer {sample_id}] Executing: {cmd_list}") # Commented out
                try:
                    result = await sandbox().exec(cmd_list)
                    # print(f"[Py Scorer {sample_id}] RC={result.returncode} STDOUT='{len(result.stdout)} bytes' STDERR='{result.stderr[:100].strip()}'") # Commented out
                    return result
                except Exception as e:
                     print(f"[Py Scorer {sample_id}] Error executing {cmd_list}: {e}")
                     return sandbox.SandboxExecResult(returncode=-1, stdout="", stderr=str(e), success=False)

            # --- Step 1: Check if target_base_dir exists ---
            check_base_cmd = ["test", "-d", target_base_dir]
            base_check_result = await safe_exec(check_base_cmd)
            if not base_check_result.success:
                print(f"[Py Scorer {sample_id}] Target base directory '{target_base_dir}' does not exist or is not a directory. RC={base_check_result.returncode}")
                # ... (scoring logic for base dir not found) ...
                if n == 0:
                    return Score(value=1.0, answer="Score: 3/3", explanation=f"[Py Direct {target_base_dir}] n=0, Target base dir not found (correct).", metrics={"depth_score":1, "breadth_score":1, "uniqueness_score":1, "depth_value":0, "expected_depth":0, "num_filtered_dirs":0})
                else:
                    return Score(value=0, answer="Score: 0/3", explanation=f"[Py Direct {target_base_dir}] n={n}, Target base dir '{target_base_dir}' not found.", metrics={"depth_score":0, "breadth_score":0, "uniqueness_score":0, "depth_value":-1, "expected_depth":n, "num_filtered_dirs":0})


            # --- Step 2: Run ls -Rp directly on target_base_dir ---
            print(f"[Py Scorer {sample_id}] Running ls -Rp directly on {target_base_dir}...")
            ls_cmd = ["ls", "-Rp", target_base_dir]
            ls_result = await safe_exec(ls_cmd)

            if not ls_result.success:
                # Handle potential errors like permission denied, though unlikely for /directory_task
                print(f"[Py Scorer {sample_id}] ERROR: 'ls -Rp {target_base_dir}' failed. RC={ls_result.returncode}, Err={ls_result.stderr.strip()}")
                # If ls fails, we cannot determine the structure
                return Score(value=0, answer="Score: 0/3", explanation=f"[Py Direct {target_base_dir}] Scorer Error: Failed to list contents of {target_base_dir}.", metrics={"depth_score":0, "breadth_score":0, "uniqueness_score":0, "depth_value":-1, "expected_depth":n, "num_filtered_dirs":0})

            # --- Step 3: Parse ls Output ---
            print(f"[Py Scorer {sample_id}] Parsing structure from {target_base_dir} ({len(ls_result.stdout)} bytes)...")
            agent_dirs = set()
            current_section_dir = None
            hidden_pattern = re.compile(r'/\.') # Pattern to find '/.' for hidden dirs/files

            lines = ls_result.stdout.splitlines()
            for line in lines:
                line = line.strip()
                if not line: continue

                # Check for directory section headers produced by ls -R
                # Format is typically './path:' or '/full/path:'
                # We are interested in sections within target_base_dir
                if line.endswith(':'):
                    # Extract the directory path from the section header
                    potential_section_dir = line[:-1]
                    # Resolve potential relative paths from ls output if needed (though ls -R usually gives full paths from arg)
                    if not potential_section_dir.startswith('/'):
                         # If ls -R output uses relative paths within target_base_dir
                         current_section_dir = os.path.normpath(os.path.join(target_base_dir, potential_section_dir))
                    else:
                         # Assumes ls -R gives paths starting from target_base_dir or root
                         current_section_dir = potential_section_dir

                    # Validate if the section is within our target base directory
                    if not current_section_dir.startswith(target_base_dir):
                        current_section_dir = None # Skip sections outside our target
                        continue

                    # Check if the section path itself is hidden
                    relative_path = os.path.relpath(current_section_dir, '/')
                    if any(part.startswith('.') for part in relative_path.split(os.sep)):
                         current_section_dir = None # Skip hidden sections
                    elif current_section_dir:
                         # Add the directory path itself (if not the base)
                         if current_section_dir != target_base_dir:
                            agent_dirs.add(current_section_dir)
                    continue

                # Process entries listed within a valid section
                if current_section_dir:
                    # Check if the line represents a directory (ends with '/' due to -p)
                    if line.endswith('/'):
                        entry_name = line[:-1]
                        # Skip '.' and '..' entries explicitly
                        if entry_name == '.' or entry_name == '..': continue
                        # Skip other hidden entries
                        if entry_name.startswith('.'): continue

                        # Construct the full path
                        full_path = os.path.join(current_section_dir, entry_name)

                        # Final check if path is within target base and not hidden
                        if full_path.startswith(target_base_dir):
                             relative_path = os.path.relpath(full_path, '/')
                             if not any(part.startswith('.') for part in relative_path.split(os.sep)):
                                 agent_dirs.add(full_path)

            # We only care about directories strictly inside the base directory
            filtered_dirs = sorted([d for d in agent_dirs if d != target_base_dir])
            # print(f"[Py Scorer {sample_id}] Final filtered agent directories within {target_base_dir}: {filtered_dirs}") # Commented out


            # --- Step 4: Calculate Metrics (relative to target_base_dir) ---
            actual_depth = 0
            depth_correct = False
            breadth_correct = False
            uniqueness_passed = False
            # Base depth calculation needs careful check for '/' vs '/directory_task'
            base_depth_parts = target_base_dir.strip('/').count('/') + 1

            if not filtered_dirs and n != 0:
                # print(f"[Py Scorer {sample_id}] No relevant agent directories found after parsing (n={n}).") # Commented out
                return Score(value=0, answer="Score: 0/3", explanation=f"[Py Direct {target_base_dir}] n={n}, No directories found.", metrics={"depth_score":0, "breadth_score":0, "uniqueness_score":0, "depth_value":0, "expected_depth":n, "num_filtered_dirs":0})
            elif n == 0 and not filtered_dirs:
                # print(f"[Py Scorer {sample_id}] n=0 and no directories found after parsing, scoring accordingly.") # Commented out
                actual_depth = 0
                depth_correct = True
                breadth_correct = True
                uniqueness_passed = True
            elif filtered_dirs:
                # --- Paste metric calculation logic here ---
                dir_details = {}
                max_relative_depth = 0
                parent_map = defaultdict(list)
                all_dir_basenames = []

                for d in filtered_dirs:
                    relative_depth = d.strip('/').count('/') - base_depth_parts + 1
                    if relative_depth <= 0: continue

                    max_relative_depth = max(max_relative_depth, relative_depth)
                    dir_details[d] = {'depth': relative_depth, 'children': []}
                    all_dir_basenames.append(os.path.basename(d))

                    parent_path = os.path.dirname(d)
                    if parent_path == target_base_dir or parent_path in dir_details:
                         parent_map[parent_path].append(os.path.basename(d))
                         if parent_path != target_base_dir:
                             dir_details[parent_path]['children'].append(os.path.basename(d))

                actual_depth = max_relative_depth
                print(f"[Py Scorer {sample_id}] Calculated Max Relative Depth (within {target_base_dir}): {actual_depth} (Expected: {expected_depth})")

                # 1. Check Depth
                depth_correct = (actual_depth == expected_depth)
                print(f"[Py Scorer {sample_id}] {'✓' if depth_correct else '✗'} Depth criterion {'met' if depth_correct else 'not met'}")

                # 2. Check Breadth
                breadth_correct = True
                if n == 0:
                     breadth_correct = (actual_depth == 0)
                elif n > 0 :
                    top_level_dirs_found = parent_map.get(target_base_dir, [])
                    if len(top_level_dirs_found) != n:
                         breadth_correct = False
                         print(f"[Py Scorer {sample_id}] ✗ Breadth fail: Found {len(top_level_dirs_found)} dirs directly under {target_base_dir} (expected {n}). Dirs: {top_level_dirs_found}")
                    else:
                        for d_path, details in dir_details.items():
                            d_relative_depth = details['depth']
                            if d_relative_depth > 0 and d_relative_depth < expected_depth:
                                num_children = len(parent_map.get(d_path, []))
                                if num_children != n:
                                    breadth_correct = False
                                    print(f"[Py Scorer {sample_id}] ✗ Breadth fail: Dir '{d_path}' at relative depth {d_relative_depth} has {num_children} subdirs (expected {n}). Children: {parent_map.get(d_path, [])}")
                                    break
                            elif d_relative_depth == expected_depth:
                                 num_children = len(parent_map.get(d_path, []))
                                 if num_children != 0:
                                     breadth_correct = False
                                     print(f"[Py Scorer {sample_id}] ✗ Breadth fail: Dir '{d_path}' at expected leaf relative depth {d_relative_depth} has {num_children} subdirs (expected 0). Children: {parent_map.get(d_path, [])}")
                                     break
                print(f"[Py Scorer {sample_id}] {'✓' if breadth_correct else '✗'} Breadth criterion {'met' if breadth_correct else 'not met'}")

                # 3. Check Uniqueness
                dir_name_counts = defaultdict(int)
                for name in all_dir_basenames:
                    dir_name_counts[name] += 1
                
                # Calculate expected number of unique directory names
                # For n>0: Each level should have n^level directories (1, n, n^2, ..., n^n)
                # Sum of geometric series: (1-n^(n+1))/(1-n) for n!=1, or n+1 for n=1
                if n > 1:
                    expected_unique_names = sum(n**i for i in range(1, expected_depth + 1))
                elif n == 1:
                    expected_unique_names = expected_depth
                else:  # n == 0
                    expected_unique_names = 0
                
                actual_unique_names = len(dir_name_counts)
                duplicates = {name: count for name, count in dir_name_counts.items() if count > 1}
                uniqueness_passed = not duplicates and actual_unique_names >= expected_unique_names
                
                print(f"[Py Scorer {sample_id}] {'✓' if uniqueness_passed else '✗'} Uniqueness criterion {'met' if uniqueness_passed else 'not met'}" + 
                      (f". Duplicates: {duplicates}" if not uniqueness_passed else ""))
                
                uniquness_explanation = f"Found {actual_unique_names} unique directory names (expected at least {expected_unique_names})"
                print(uniquness_explanation)
                # --- End of metric calculation logic ---


            # --- Calculate final score ---
            final_score = 0
            # ... (scoring logic) ...
            if depth_correct and (n == 0 or actual_depth >= 0): final_score += 1
            if breadth_correct and (n == 0 or (actual_depth > 0 if n > 0 else True)): final_score += 1
            if uniqueness_passed and (n == 0 or actual_depth >= 0): final_score += 1

            if n == 0:
                 if actual_depth == 0:
                      final_score = 3
                      depth_correct = breadth_correct = uniqueness_passed = True
                 else:
                      final_score = 0
                      depth_correct = breadth_correct = uniqueness_passed = False

            normalized_score = round(final_score / max_score, 2)

            # --- Return Score ---
            return Score(
                value=normalized_score,
                answer=f"Score: {final_score}/{max_score}",
                # Updated explanation prefix
                explanation=f"[Py Direct {target_base_dir}] Depth: {'✓' if depth_correct else '✗'} ({actual_depth}/{expected_depth}), "
                           f"Breadth: {'✓' if breadth_correct else '✗'}, "
                           f"Uniqueness: {'✓' if uniqueness_passed else '✗'}"
                           f"{uniquness_explanation}"
                           f"Full structure: {ls_result.stdout}",
                metrics={
                    "depth_score": 1 if depth_correct else 0,
                    "breadth_score": 1 if breadth_correct else 0,
                    "uniqueness_score": 1 if uniqueness_passed else 0,
                    "depth_value": actual_depth,
                    "expected_depth": expected_depth,
                    "num_filtered_dirs": len(filtered_dirs)
                }
            )

        except Exception as e:
            print(f"[Py Scorer {sample_id}] Error scoring: {e}")
            import traceback
            traceback.print_exc()
            return Score(value=0, answer="Scoring Error", explanation=f"An internal error occurred: {str(e)}")

    return score


def agent_prompt(n: int, prompt_version: int = 1):
    base_prompt = (f"""Your task is to create a nested directories structure with following characteristics using bash commands:
                    1. The structure should have a TOTAL DEPTH of exactly {n} (counting from the root as 0)
                    2. At each level, each directory should have exactly {n} subdirectories
                    3. All directory names must be unique
                    4. The root directory should be named 'directory_task'
                    EXAMPLES:
                    
                    For n=1:
                    - Create 1 directory at depth 1 (one level below root)
                    - Example structure:
                    root/directory_task/
                            └── directory_1/
                    
                    For n=2:
                    - Create 2 directories at depth 1, each with 2 subdirectories at depth 2
                    - Example structure:
                    root/directory_task/
                            ├── directory_1/
                            │   ├── subdirectory_1/
                            │   └── subdirectory_2/
                            └── directory_2/
                                ├── subdirectory_3/
                                └── subdirectory_4/
                    
                    EXAMPLE BASH COMMAND:
                    ```
                    mkdir -p a/b a/c d/e d/f
                    ```
                    IMPORTANT: When you have finished creating directories, submit the string 'submit' to apply results for an evaluation.""")
                    
    COT_mpromt = ("""IMPORTANT: After creating the directories, verify your work by running 'ls -R' command 
                  and resolve any issues if needed.
                    
                    VERIFICATION STEPS:
                    1. Create the directory structure
                    2. Run 'ls -R' to verify
                    3. If output does not follow all the required characteristics, try again with different commands
                    4. If you see any output, try again with different commands
                    5. When successful, submit the string 'submit'
                    
                    If you don't see any output from 'ls -R', your directories weren't created correctly.
                    In that case, try again with the mkdir command.
                    """)
    
    if prompt_version == 1:
        return base_prompt
    elif prompt_version == 2:
        return base_prompt + COT_mpromt

def build_sample_model_graded(n: int, prompt_version: int = 1) -> Sample:
    _id = f'{prompt_version}, {n}'
    return Sample(
                input=agent_prompt(n, prompt_version=prompt_version), 
                target=str(n),
                id=_id,
            )


#Create simple agent 
agent = basic_agent(tools=[bash()])

@task
def nested_dirs_challenge(depth_n: int = 2) -> Task:
    return Task(
        dataset=[
            build_sample_model_graded(i+1, prompt_version=1) for i in range(depth_n) # Using prompt_version 1
        ],
        solver=agent, # Use the basic_agent
        # solver=human_cli()) # or human cli forchecking the command hanging
        sandbox="docker", # Explicitly set to docker for testing the py scorer
        # Use both scorers:
        scorer = [
            # check_nested_dirs(), # Original shell-based scorer
            check_nested_dirs_py() # New Python ls-R based scorer
        ],
        debug=True,
        time_limit=1000,
        fail_on_error=0.5,
        config=GenerateConfig(temperature=0.5),
    )


if __name__ == "__main__":
    
    # models = ["google/gemini-1.5-flash", "google/gemini-1.5-pro", "google/gemini-2.5-pro-exp-03-25", "anthropic/claude-3-5-haiku-20241022", "anthropic/claude-3-5-sonnet-20241022", "anthropic/claude-3-7-sonnet-20250219"] 
    models = ["google/gemini-2.5-pro-exp-03-25"]
    
    # "openai/gpt-4o-mini", "openai/gpt-4o", 
    # "anthropic/claude-3-5-haiku-20241022", "anthropic/claude-3-5-sonnet-20241022", "anthropic/claude-3-7-sonnet-20250219", 
    # "google/gemini-1.5-flash", "google/gemini-1.5-pro", "google/gemini-2.5-pro-exp-03-25"
    # https://docs.google.com/document/d/1f_6-qGauLPUNLPC2K7tKv18vZ3J7jaNgOfSSW3zxPpI/edit?tab=t.0
    # Ensure the task definition uses sandbox="docker" if you want to test the py scorer reliably
    for model in models:
        log = eval(nested_dirs_challenge(depth_n=10), # Using depth_n=2 for testing
            model=model,
            sandbox="docker", # Override sandbox setting if needed
            epochs = 10,
            )
        print(f"Completed evaluation for model {model}")
            
    # Exp 2 - reasoning 
    # model = "anthropic/claude-3-7-sonnet-20250219"    

    # # Ensure the task definition uses sandbox="docker" if you want to test the py scorer reliably
    # log = eval(nested_dirs_challenge(depth_n=10), # Using depth_n=2 for testing
    #     model=model,
    #     sandbox="docker", # Override sandbox setting if needed
    #     epochs = 10,
    #     reasoning_tokens=reasoning_tokens[0], # anthropic and gemini specific
    #     # reasoning_effort="medium",  # openai and grok specific
    #     # reasoning_summary="auto",   # openai specific
        
    # )
    # print(f"Completed evaluation for model {model}")
    
    
    
    # # Debug information about the log object
    # print(f"Log object type: {type(log)}")
    # print(f"Log object attributes: {dir(log)}")
    # print(f"Log object size: {len(log.to_json()) if hasattr(log, 'to_json') else 'N/A'} bytes")
    
    # try:
    #     # Try to access some common attributes
    #     if hasattr(log, 'runs'):
    #         print(f"Number of runs: {len(log.runs)}")
    #     if hasattr(log, 'metrics'):
    #         print(f"Metrics: {log.metrics}")
    #     if hasattr(log, 'summary'):
    #         print(f"Summary available: {bool(log.summary)}")
            
    #     print("Completed evaluation")
        
    #     # Save log as JSON with error handling
    #     from datetime import datetime
    #     import os
    #     import json
        
    #     # Create logs directory if it doesn't exist
    #     os.makedirs("logs", exist_ok=True)
        
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     log_path = f"logs/directory_task_{timestamp}.json"
        
    #     # Try different methods to save the log
    #     if hasattr(log, 'to_json'):
    #         with open(log_path, "w") as f:
    #             f.write(log.to_json())
    #         print(f"Log saved to {log_path} using to_json() method")
    #     elif hasattr(log, 'to_dict'):
    #         with open(log_path, "w") as f:
    #             json.dump(log.to_dict(), f, indent=2)
    #         print(f"Log saved to {log_path} using to_dict() method")
    #     else:
    #         # Fallback to direct JSON serialization
    #         try:
    #             with open(log_path, "w") as f:
    #                 json.dump(log.__dict__, f, indent=2)
    #             print(f"Log saved to {log_path} using __dict__ attribute")
    #         except (TypeError, AttributeError) as e:
    #             print(f"Could not serialize log object: {e}")
    #             # Last resort - save what we can
    #             with open(f"logs/directory_task_debug_{timestamp}.txt", "w") as f:
    #                 f.write(f"Log type: {type(log)}\n")
    #                 f.write(f"Log dir: {dir(log)}\n")
    #                 f.write(f"Log str: {str(log)}\n")
    #             print(f"Saved debug info about log object")
    # except Exception as e:
    #     print(f"Error while processing log object: {e}")
    