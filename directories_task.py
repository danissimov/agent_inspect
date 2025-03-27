import re
from inspect_ai import eval, Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Target, accuracy, stderr, scorer, model_graded_qa, Scorer, _model, INCORRECT, CORRECT
from inspect_ai.scorer._model import DEFAULT_MODEL_GRADED_QA_TEMPLATE, DEFAULT_GRADE_PATTERN
from inspect_ai.solver import Plan, TaskState, generate, use_tools, basic_agent, solver
from inspect_ai.tool import bash, tool
from inspect_ai.util import sandbox
from inspect_ai.model._model import Model, get_model

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
            target_match = re.search(r'Target input: (\d+)', target.text)
            if not target_match:
                return Score(value=0, answer="Failed to extract target value from description")
            
            n = int(target_match.group(1))
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


def parse_component_scores(explanation_text):
    """
    As it appears to be pretty hard to extract separate,
    it makes sense to parse component scores 
    from explanation text in the format:
    "Depth: ✓, Breadth: ✗, Uniqueness: ✓"
    
    Returns a dictionary with binary scores for each component
    """
    if not explanation_text or "," not in explanation_text:
        return {
            "depth_score": 0,
            "breadth_score": 0, 
            "uniqueness_score": 0
        }
    
    try:
        components = [comp.strip() for comp in explanation_text.split(",")]
        results = {}
        
        if components[0].startswith("Depth:"):
            results["depth_score"] = 1 if "✓" in components[0] else 0
        
        if len(components) > 1 and components[1].startswith("Breadth:"):
            results["breadth_score"] = 1 if "✓" in components[1] else 0
        
        if len(components) > 2 and components[2].startswith("Uniqueness:"):
            results["uniqueness_score"] = 1 if "✓" in components[2] else 0
            
        return results
        
    except Exception as e:
        print(f"Error parsing component scores: {e}")
        return {
            "depth_score": 0,
            "breadth_score": 0, 
            "uniqueness_score": 0
        }


def agent_prompt(n: int, prompt_version: int = 1):
    base_prompt = (f"""Your task is to create a nested directories structure with following characteristics using bash commands:
                    1. The structure should have a TOTAL DEPTH of exactly {n} (counting from the root as 0)
                    2. At each level, each directory should have exactly {n} subdirectories
                    3. All directory names must be unique
                    
                    EXAMPLES:
                    
                    For n=1:
                    - Create 1 directory at depth 1 (one level below root)
                    - Example structure:
                    root/
                        └── dir1/
                    
                    For n=2:
                    - Create 2 directories at depth 1, each with 2 subdirectories at depth 2
                    - Example structure:
                    root/
                        ├── dir1/
                        │   ├── subdir1a/
                        │   └── subdir1b/
                        └── dir2/
                            ├── subdir2a/
                            └── subdir2b/
                    
                    EXAMPLE BASH COMMAND:
                    ```
                    mkdir -p a/b a/c d/e d/f
                    ```
                    When you have finished, submit the string 'submit'.""")
                    
    COT_mpromt = (f"""IMPORTANT: After creating the directories, verify your work by running 'ls -R' command 
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

def build_sample_model_graded(n: int, prompt_version: int = 1):
  return Sample(
                input=agent_prompt(n, prompt_version=prompt_version),
                  
                target=f"""Solution for the input n must follow these conditions:
                
                - Depth: Maximum depth equals n.
                - Subdirectory Count: Non-leaf directories have exactly n subdirectories.
                - Uniqueness: All directory names are unique.
                - Leaf Directories: Directories at depth n are empty.
                
                Target input: {n}""",
            )
                  

@scorer(metrics=[accuracy(), stderr()])
def command_efficiency_scorer():
    """Score based on the number and efficiency of bash commands used"""
    async def score(state: TaskState, target: Target):
        sample_id = str(id(state))[-8:]
        
        try:
            print(f"[Sample {sample_id}] Analyzing command efficiency...")
            
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
            
            # Extract bash commands with the same approach as before...
            bash_commands = []
            
            # Check if we have access to the execution steps
            has_execution = False
            tool_calls = []
            
            # Method 1: Try to access execution steps directly
            if hasattr(state, 'execution') and state.execution:
                has_execution = True
                print(f"[Sample {sample_id}] Found execution steps")
                
                for step in state.execution:
                    if hasattr(step, 'tool') and step.tool == "bash":
                        if hasattr(step, 'args') and isinstance(step.args, dict) and 'cmd' in step.args:
                            cmd = step.args['cmd']
                            if cmd:
                                bash_commands.append(cmd)
                                print(f"[Sample {sample_id}] Found bash command in execution: {cmd}")
            
            # Method 2: Try to access tool calls as an alternative
            elif hasattr(state, 'tool_calls') and state.tool_calls:
                has_execution = True
                print(f"[Sample {sample_id}] Found tool calls")
                
                for call in state.tool_calls:
                    if hasattr(call, 'name') and call.name == "bash":
                        if hasattr(call, 'args') and isinstance(call.args, dict) and 'cmd' in call.args:
                            cmd = call.args['cmd']
                    if cmd:
                        bash_commands.append(cmd)
                        print(f"[Sample {sample_id}] Found bash command in tool call: {cmd}")
            
            # Method 3: Extract from model output as last resort
            if not has_execution and hasattr(state, 'output') and state.output and hasattr(state.output, 'completion'):
                completion = state.output.completion
                print(f"[Sample {sample_id}] No execution data, extracting from text...")
                
                # Debug: Print part of the completion
                debug_excerpt = completion[:200].replace('\n', '\\n')
                print(f"[Sample {sample_id}] Model output excerpt: {debug_excerpt}...")
                
                # Patterns to look for bash commands
                patterns = [
                    # Code blocks with bash/shell
                    r'```(?:bash|shell)?\n(.*?)```',
                    
                    # Lines starting with $ or >
                    r'(?:^|\n)\s*[$>]\s*([^\n]+)',
                    
                    # Inline code with mkdir/touch/cd/etc
                    r'`([^`]+?(?:mkdir|touch|cd|find|ls|rm|cp|mv)[^`]*?)`',
                    
                    # Common commands at line start (no markup)
                    r'(?:^|\n)(?:sudo\s+)?(mkdir|touch|cd|find|ls|rm|cp|mv)\s+[^\n]+'
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, completion, re.DOTALL)
                    for match in matches:
                        if isinstance(match, tuple):
                            match = ' '.join(match).strip()
                        else:
                            match = match.strip()
                            
                        if match and not match.startswith('#'):
                            bash_commands.append(match)
                            print(f"[Sample {sample_id}] Found command via regex: {match}")
            
            # Print summary
            if bash_commands:
                print(f"[Sample {sample_id}] Total commands found: {len(bash_commands)}")
                for i, cmd in enumerate(bash_commands):
                    print(f"[Sample {sample_id}] Command {i+1}: {cmd}")
            else:
                print(f"[Sample {sample_id}] No commands found in the model's output")
            
            # Filter out verification commands
            verification_commands = ["ls", "pwd", "echo", "cat"]
            meaningful_commands = []
            
            for cmd in bash_commands:
                cmd_parts = cmd.split()
                if not cmd_parts:
                    continue
                    
                # Check if the command starts with a verification command
                is_verification = False
                for vcmd in verification_commands:
                    if cmd_parts[0] == vcmd or cmd.startswith(vcmd + " "):
                        is_verification = True
                        break
                
                if not is_verification:
                    meaningful_commands.append(cmd)
            
            command_count = len(meaningful_commands)
            print(f"[Sample {sample_id}] Meaningful commands: {command_count}")
            
            # Score based on efficiency criteria
            if command_count == 0:
                efficiency_score = 0.0
                explanation = "No meaningful commands detected"
            elif command_count == 1 and "mkdir -p" in meaningful_commands[0]:
                efficiency_score = 1.0
                explanation = "Optimal solution: single mkdir -p command"
            elif command_count == 1:
                efficiency_score = 0.8
                explanation = "Good solution: single command"
            else:
                # Decrease score as commands increase, but never below 0.2
                efficiency_score = max(0.2, 1.0 - (command_count - 1) * 0.1)
                explanation = f"Used {command_count} commands (optimal is 1)"
            
            return Score(
                value=efficiency_score,
                answer=f"Used {command_count} meaningful commands",
                explanation=explanation,
                metrics={
                    "command_count": command_count,
                    "meaningful_commands": meaningful_commands
                }
            )
            
        except Exception as e:
            print(f"[Sample {sample_id}] Command scoring error: {e}")
            import traceback
            traceback.print_exc()
            return Score(value=0, answer=f"Error: {str(e)}")
    
    return score



#Create simple agent 
agent = basic_agent(tools=[bash()])

@task
def nested_dirs_challenge(depth_n: int = 2) -> Task:
    return Task(
        dataset=[
            build_sample_model_graded(i+1) for i in range(depth_n)
        ],
        #solver=directory_verification_agent(),  # Use our properly registered custom agent
        solver=agent,
        sandbox="local",
        
        # 1 - simple model qa
        # scorer=model_graded_qa(),
        
        # 2 model qa - fix input (ls -R)
        # scorer = custom_model_graded_qa(model="openai/gpt-4o")
        
        # 3 - double scorer 
        # scorer = [check_nested_dirs(),
        #           command_efficiency_scorer()
        #           ],
        
        # 4 - double scorer 
        scorer = check_nested_dirs(),
                  
        debug=True,
        time_limit=300,  # 5 minutes per sample
        fail_on_error=0.5 
        # # Add a token limit to prevent excessive generation
        # token_limit=4000
    )


if __name__ == "__main__":
    
    # # Run a single evaluation (for testing)
    log = eval(nested_dirs_challenge(depth_n=5),
             model="openai/gpt-4o",  
             epochs = 10, 
             name = "Manual test" ,
             version = "v1",
         metadata={
                "run_info": """
                        Single test run. 
                    Model: gpt-4o-mini
                    Task: nested_dirs_challenge
                        Prompt: default
                    Score: numeric 0-3, based on bash commands results
                        """
             }
    )
    
    
    # Run experiment 
    # models = ["google/gemini-1.5-flash", "google/gemini-1.5-pro", "gemini-2.5-pro-exp-03-25", "openai/gpt-4o-mini", "openai/gpt-4o"]
    # for model in models:
    #     print(f"Evaluating model {model}...")
    #     try:
    #         log = eval(nested_dirs_challenge(depth_n=10),
    #             model=model,
    #             epochs = 10,
    #             metadata={
    #                 "run_info": "Experiment with different models"})
    #     except Exception as e:
    #         print(f"Error evaluating model {model}: {e}")
    #         import traceback
    #         traceback.print_exc()


    # this addressing of log object does not work, while it strictly follows the doc
    # try:
    #     if log.status == "success":
    #         print(f"\nEvaluation results:\n{log.results}")
    #         print(f"\nEvaluation error:\n{log.error}")
    #     else:
    #         print(f"\nEvaluation results:\n{log}")
    # except Exception as e:
    #     print(f"Error printing results: {e}")
    #     import traceback
    #     traceback.print_exc()