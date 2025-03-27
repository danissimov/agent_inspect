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
