import re
from inspect_ai import eval, Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Target, accuracy, stderr, scorer, model_graded_qa, Scorer, _model, INCORRECT, CORRECT
from inspect_ai.scorer._model import DEFAULT_MODEL_GRADED_QA_TEMPLATE, DEFAULT_GRADE_PATTERN
from inspect_ai.solver import Plan, TaskState, generate, use_tools, basic_agent
from inspect_ai.tool import bash
from inspect_ai.util import sandbox
# from inspect_ai.models import get_model
from inspect_ai.model._model import Model, get_model

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
    """numerical scoring function 0 to 3"""
    async def score(state: TaskState, target: Target):
        score = 0 
        try:
            output = await sandbox().exec('ls -R')
            print(f"The dir structure:\n\n{output.stdout.strip()}")
            
            count = await sandbox().exec('find . | wc -l')
            #check max depth 
            if int(count.stdout.strip()) == int(target.text)*(int(target.text)+1):
              score += 1
            # count all elements 
            nestedness = await sandbox().exec("find . -type d | awk -F/ '{print NF-1}' | sort -n | tail -1")
            print(f"The max depth:\n\n{nestedness.stdout.strip()}")
            if int(nestedness.stdout.strip()) == int(target.text):
              score += 1
            # Check the amount of unique names in the directory
            unique_names_count = await sandbox().exec("find . -type d -printf '%f\n' | sort | uniq | wc -l")
            print(f"The number of unique directory names:\n\n{unique_names_count.stdout.strip()}")
            if int(unique_names_count.stdout.strip()) == int(target.text)*(int(target.text)+1):
              score += 1
        except Exception as e:
            print(f"Error scoring {e}")
            Score(value=0)    
        return Score(value=score, answer="", explanation=output.stdout.strip() )

    return score


@scorer(metrics=[accuracy(), stderr()])
def custom_model_graded_qa(
    template: str = DEFAULT_MODEL_GRADED_QA_TEMPLATE,
    instructions: str = "",
    grade_pattern: str = DEFAULT_GRADE_PATTERN,
    model: str | Model | None = None,
) -> Scorer:
    """ """
    # resolve model
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

# from inspect_ai import TaskState, Target, Score, scorer, metrics
# from inspect_ai.scorer import Model, resource, get_model
# import re

# @scorer(metrics=[accuracy(), stderr()])  # Inherit model-graded metrics
# def ls_model_graded_composite(
#     template: str = DEFAULT_MODEL_GRADED_QA_TEMPLATE,
#     instructions: str = "",
#     grade_pattern: str = DEFAULT_GRADE_PATTERN,
#     model: str | Model | None = None,
# ) -> Scorer:
#     """First runs ls -R, then grades output with LLM."""
    
#     # 1. Create the model-graded QA scorer (reuse existing config)
#     qa_scorer = model_graded_qa(
#         template=template,
#         instructions=instructions,
#         grade_pattern=grade_pattern,
#         model=model
#     )

#     # 2. Define composite scoring logic
#     async def score(state: TaskState, target: Target) -> Score:
#         try:
#             # Capture directory structure
#             ls_result = await sandbox().exec("ls -R")
#             dir_structure = ls_result.stdout
            
#             # Create modified state with directory structure as "answer"
#             modified_state = TaskState(
#                 input_text=state.input_text,  # Preserve original input
#                 output=Output(completion=dir_structure),
#                 **state.other_attributes  # Propagate other state fields
#             )
            
#             # Delegate to model-graded scorer
#             return await qa_scorer(modified_state, target)
            
#         except Exception as e:
#             return Score(
#                 value=0,
#                 explanation=f"Failed to read directory structure: {str(e)}"
#             )

#     return score

agent = basic_agent(tools=[bash()])


def build_sample_model_graded(n: int):
  return Sample(
                input=(
                    f"""Your task is to create a structure of nested directories of depth {n}, where each directory holds {n} folders. Each folder should have a unique name. 
                    For example, at difficulty level 1 the structure could look like: 
                    /home/agent/output/ 
                      - a/ (empty dir)
                    
                    and for difficulty level 2 the structure could look like:
                    /home/agent/output/
                      - /a/
                      -- /a/b/ (empty dir)
                      -- /a/c/ (empty dir)
                      - /d/
                      -- /d/e/ (empty dir)
                      -- /d/f/ (empty dir)
                          """
                    "When you have finished, submit the string 'submit'."  # Without this, the agent is liable to loop until it hits its limits.
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


@task
def nested_dirs_challenge(depth_n: int = 5) -> Task:
    return Task(
        dataset=[
            build_sample_model_graded(i+1) for i in range (depth_n)
        ],
        
        solver=agent,
        sandbox="docker",
        
        # 1 - simple model qa
        # scorer=model_graded_qa(),
        
        # 2 model qa - fix input (ls -R)
        # scorer = custom_model_graded_qa(model="openai/gpt-4o")
        
        # 3 numerical scorer
        scorer = check_nested_dirs()
        
        # 4 multiple scorers possible
        # scorer=[
        # model_graded_qa(model="openai/gpt-4"), 
        # model_graded_qa(model="google/gemini-1.5-pro")
        #        ],
    )
    
eval(nested_dirs_challenge(depth_n=1),
     model="openai/gpt-4o-mini",
     )