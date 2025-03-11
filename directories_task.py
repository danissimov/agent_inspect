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

@scorer(metrics=[accuracy()])
def check_file_exists():
    """check if file exists"""
    async def score(state: TaskState, target: Target):
        try:
            output = await sandbox().exec('ls')            
            print(output)
            _ = await sandbox().read_file(target.text)
            exists = True
            
        except FileNotFoundError:
            exists = False
        return Score(value=1 if exists else 0)

    return score


@scorer(metrics=[accuracy(), stderr()])
def check_nested_dirs():
    """Score directory structure based on depth, breadth, and uniqueness"""
    async def score(state: TaskState, target: Target):
        total_score = 0
        max_score = 3  
        
        try:
            # Extract the target value (n) from the target text or modify target itself 
            target_match = re.search(r'Target input: (\d+)', target.text)
            if not target_match:
                return Score(value=0, answer="Failed to extract target value from description")
            
            n = int(target_match.group(1))
            print(f"Extracted target value n = {n}")
            
            # Debug: Check current working directory
            pwd_result = await sandbox().exec('pwd')
            print(f"Current working directory: {pwd_result.stdout.strip()}")
            
            # Get directory structure - limit the output to avoid hanging
            output = await sandbox().exec('ls -R')
            print(f"The dir structure:\n\n{output.stdout.strip()}")
            
            # 1. Check depth - use command with timeout, stuck otherwise
            depth_cmd = "find . -type d -not -path '*/\\.*' | sort | awk -F/ '{print NF-1}' | sort -n | tail -1"
            depth_result = await sandbox().exec(depth_cmd)
            
            if not depth_result.stdout.strip():
                actual_depth = 0
            else:
                actual_depth = int(depth_result.stdout.strip())
                
            print(f"The max depth: {actual_depth}")
            if actual_depth == n:
                total_score += 1
                print("✓ Depth criterion met")
            else:
                print(f"✗ Depth criterion not met (expected {n}, got {actual_depth})")
            
            # 2. Check breadth - simplified approach to avoid hanging
            breadth_correct = True
            
            # Only check if depth is at least 1
            if actual_depth >= 1:
                # Get top-level directories
                top_dirs = await sandbox().exec("find . -mindepth 1 -maxdepth 1 -type d | sort")
                top_dir_list = [d for d in top_dirs.stdout.strip().split('\n') if d]
                
                # Check if we have the right number of top directories for n > 1
                if n > 1 and len(top_dir_list) != n:
                    breadth_correct = False
                    print(f"✗ Found {len(top_dir_list)} top-level directories (expected {n})")
                
                # Sample check: examine just the first directory at each level
                if breadth_correct and n > 1:
                    for level in range(1, min(n, 5)):  # Limit to first 3 levels to avoid hanging
                        if not breadth_correct:
                            break
                            
                        # Get a sample directory at this level
                        sample_dir = await sandbox().exec(f"find . -mindepth {level} -maxdepth {level} -type d | head -1")
                        if not sample_dir.stdout.strip():
                            continue
                            
                        # Count its subdirectories
                        sample_path = sample_dir.stdout.strip()
                        subdir_count = await sandbox().exec(f"find '{sample_path}' -mindepth 1 -maxdepth 1 -type d | wc -l")
                        count = int(subdir_count.stdout.strip())
                        
                        if count != n:
                            breadth_correct = False
                            print(f"✗ Directory {sample_path} has {count} subdirectories (expected {n})")
            
            if breadth_correct:
                total_score += 1
                print("✓ Breadth criterion met")
            
            # 3. Check uniqueness - Instead of getting all names, just check if there are duplicates
            dup_check = await sandbox().exec("find . -type d -not -path '*/\\.*' -printf '%f\n' | sort | uniq -d | wc -l")
            dup_count = int(dup_check.stdout.strip())
            
            if dup_count == 0:
                total_score += 1
                print("✓ Uniqueness criterion met (no duplicate names found)")
            else:
                print(f"✗ Uniqueness criterion not met ({dup_count} duplicate directory names found)")
            
            # Calculate normalized score
            normalized_score = round(total_score / max_score, 2)
            
            return Score(
                value=normalized_score,
                answer=f"Score: {total_score}/{max_score}",
                explanation=f"Depth: {'✓' if actual_depth == n else '✗'}, "
                           f"Breadth: {'✓' if breadth_correct else '✗'}, "
                           f"Uniqueness: {'✓' if dup_count == 0 else '✗'}"
            )
            
        except Exception as e:
            print(f"Error scoring: {e}")
            import traceback
            traceback.print_exc()
            return Score(value=0, answer=f"Error: {str(e)}")

    return score


@scorer(metrics=[accuracy(), stderr()])
def custom_model_graded_qa(
    template: str = DEFAULT_MODEL_GRADED_QA_TEMPLATE,
    instructions: str = "",
    grade_pattern: str = DEFAULT_GRADE_PATTERN,
    model: str | Model | None = None,
) -> Scorer:
    """Custom scorer to evaluate directories structure printed as a plain text 
    
    Still in proggress """
    
    grader_model = get_model(model)

    async def score(state: TaskState, target: Target) -> Score:
        
        # format the model grading template    
        directories_structure_result = await sandbox().exec('ls -R')
        directories_structure = directories_structure_result.stdout
        
        print(f"The dir structure: {directories_structure}")
        
        score_prompt = template.format(
            question=state.input_text,
            answer=directories_structure,
            criterion=target.text,
            instructions=instructions,
        )

        # query the model for the score
        result = await grader_model.generate(score_prompt)

        # extract the grade
        match = re.search(grade_pattern, result.completion)
        if match:
            return Score(
                value=match.group(1),
                answer=match.group(0),
                explanation=result.completion,
            )
        else:
            return Score(
                value=INCORRECT,
                explanation="Grade not found in model output: "
                + f"{result.completion}",
            )
    return score


def build_sample_model_graded(n: int):
  return Sample(
                input=(
                    f"""Your task is to create a structure of nested directories of depth {n}, where each directory holds {n} folders. Each folder should have a unique name.
                    
                    IMPORTANT: Create these directories in the current working directory.
                    
                    For example, at difficulty level 1 the structure could look like: 
                    - a/ (empty dir)
                    
                    and for difficulty level 2 the structure could look like:
                    - a/
                      - a/b/ (empty dir)
                      - a/c/ (empty dir)
                    - d/
                      - d/e/ (empty dir)
                      - d/f/ (empty dir)
                    
                    Use mkdir commands to create this structure. For example:
                    ```
                    mkdir -p a/b a/c d/e d/f
                    ```
                    
                    IMPORTANT: After creating the directories, run 'ls -R' to verify your work.
                    If you don't see any output from 'ls -R', your directories weren't created correctly.
                    In that case, try again with the mkdir command.
                    
                    VERIFICATION STEPS:
                    1. Create the directory structure
                    2. Run 'ls -R' to verify
                    3. If no output, try again with different commands
                    4. When successful, submit the string 'submit'
                    
                    When you have finished, submit the string 'submit'.
                    """
                ),
                  
                target=f"""Solution for the input n must follow these conditions:
                
                - Depth: Maximum depth equals n.
                - Subdirectory Count: Non-leaf directories have exactly n subdirectories.
                - Uniqueness: All directory names are unique.
                - Leaf Directories: Directories at depth n are empty.
                
                Target input: {n}""",
            )
                  
    
@task
def hello_world_challenge() -> Task:
    return Task(
        dataset=[
            Sample(
                input=(
                    "Create a file called `hello-world.txt`. "
                    "When you have finished, submit the string 'submit'."  # Without this, the agent is liable to loop until it hits its limits.
                ),
                target="hello-world.txt",
            )
        ],
        # solver=[use_tools([bash()]), generate()],
        solver=agent,
        sandbox="local",
        scorer=check_file_exists(),
    )


def counting_bash():
    """Bash tool that counts command invocations
    
    It works, but I changed approach to another scorer instead"""
    bash_tool = bash()
    
    @wraps(bash_tool)
    async def wrapped_bash(cmd, **kwargs):
        # Get the sample ID from the current context
        from inspect_ai.util import context
        sample_id = context().get("sample_id", "unknown")
        
        # Initialize counter for this sample if not exists
        if sample_id not in bash_command_count:
            bash_command_count[sample_id] = 0
        
        # Increment the counter
        bash_command_count[sample_id] += 1
        
        # Log the command
        print(f"Bash command #{bash_command_count[sample_id]}: {cmd}")
        
        # Execute the original bash command
        return await bash_tool(cmd, **kwargs)
    
    return wrapped_bash

#Create simple agent 
agent = basic_agent(tools=[bash()])
# or agent with wrapped bash tool 
# agent = basic_agent(tools=[wrapped_bash()])

@scorer(metrics=[accuracy(), stderr()])
def command_efficiency_scorer():
    """Score based on the number of bash commands used"""
    async def score(state: TaskState, target: Target):
        try:
            # Extract bash commands from the state's actions
            bash_commands = []
            for action in state.actions:
                if hasattr(action, 'tool') and action.tool == "bash" and hasattr(action, 'args'):
                    cmd = action.args.get("cmd", "") if isinstance(action.args, dict) else ""
                    if cmd:
                        bash_commands.append(cmd)
            
            # Count meaningful commands (excluding ls, pwd, etc.)
            verification_commands = ["pwd", "echo", "cat"]
            meaningful_commands = [cmd for cmd in bash_commands 
                                  if not any(cmd.startswith(vc) for vc in verification_commands)]
            command_count = len(meaningful_commands)
            
            # Extract n from target
            target_match = re.search(r'Target input: (\d+)', target.text)
            if not target_match:
                return Score(value=0, answer="Failed to extract target value")
            
            n = int(target_match.group(1))
            
            # Expected minimum commands: 1 mkdir command
            expected_min = 1
            
            # Score based on efficiency (1.0 for minimum commands, decreasing for more)
            if command_count <= expected_min:
                efficiency_score = 1.0
            else:
                # Decrease score as commands increase, but never below 0
                # Allow more commands for larger n values
                efficiency_score = max(0, 1.0 - (command_count - expected_min) / (n + 1))
            
            return Score(
                value=efficiency_score,
                answer=f"Used {command_count} meaningful commands",
                explanation=f"Command efficiency: {efficiency_score:.2f}\n"
                           f"Commands used: {', '.join(meaningful_commands)}"
            )
        except Exception as e:
            print(f"Error in command efficiency scorer: {e}")
            import traceback
            traceback.print_exc()
            return Score(value=0, answer=f"Error: {str(e)}")
    
    return score

@solver
def directory_verification_agent(tools=[bash()]):
    """Agent that creates directories and verifies they exist
    
    Did not work yet"""
    
    base_agent = basic_agent(tools=tools)
    
    async def solve(state: TaskState):
        # First, let the basic agent try to solve the task
        result = await base_agent(state)
        
        # Then, verify the work was done
        verification = await sandbox().exec('ls -R')
        print(f"Agent verification - directory structure:\n{verification.stdout}")
        
        if not verification.stdout.strip():
            # If no directories were created, try a simple example
            print("No directories found. Attempting to create a simple example...")
            
            # Extract n from the input
            n_match = re.search(r'depth (\d+)', state.input_text)
            if n_match:
                n = int(n_match.group(1))
                print(f"Creating example structure for n={n}")
                
                # Create a simple structure that meets the criteria
                if n == 1:
                    await sandbox().exec('mkdir -p dir1')
                elif n == 2:
                    await sandbox().exec('mkdir -p dir1/subdir1 dir1/subdir2 dir2/subdir3 dir2/subdir4')
                elif n == 3:
                    await sandbox().exec('mkdir -p dir1/subdir1/leaf1 dir1/subdir2/leaf2 dir1/subdir3/leaf3 dir2/subdir4/leaf4 dir2/subdir5/leaf5 dir2/subdir6/leaf6 dir3/subdir7/leaf7 dir3/subdir8/leaf8 dir3/subdir9/leaf9')
                
                # Verify again
                verification = await sandbox().exec('ls -R')
                print(f"After recovery - directory structure:\n{verification.stdout}")
        
        return result
    
    return solve

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
        
        # 3 numerical scorer
        scorer = [check_nested_dirs(),
                  command_efficiency_scorer()
                  ],
        debug=True
        # 4 multiple scorers possible
        # scorer=[
        # model_graded_qa(model="openai/gpt-4"), 
        # model_graded_qa(model="google/gemini-1.5-pro")
        #        ],
    )
    
    
log = eval(nested_dirs_challenge(depth_n=10),
         model="openai/gpt-4o-mini",    
         metadata={
            "run_info":"""
                    Run 2. 
                    Model: gpt-4o-mini
                    Task: nested_dirs_challenge
                    Prompt: v2, CoT self-check
                    Score: numeric 0-3, based on bash commands results
                    """,
            "bash_command_count": bash_command_count
                }
         )

