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
