My code for the assesment for SPAR AI safety mentorship program 

Evaluate behaviour of agent building nested directories structure using bash using Inspect library by AISI

The Task at difficulty level n requires the agent to create a structure of nested directories of depth n, where each directory holds n folders. Each folder should have a unique name. For example, at difficulty level 1 the structure could look like:

/home/agent/output/
- a/ (empty dir)

and for difficulty level 2 the structure could look like

/home/agent/output/
 - /a/
 -- /a/b/ (empty dir)
 -- /a/c/ (empty dir)
 - /d/
 -- /d/e/ (empty dir)
 -- /d/f/ (empty dir)

Once you've created this task, explore how well an LLM agent does on the task, for various values of n.

The full task description: 
https://docs.google.com/document/d/1QJx7Ys0z7Vpy1sZeuCHtXeko2z9lj2_XAnExZMRpPIM/edit?tab=t.0


All the code is here:
- directories_task.py

To access eval logs:
- inspect view

Environment used:
- conda activate redvenv



Plan 
- 


Exp logs:

1. Run 1 

n = 10 

Prompt_v1 = """Your task is to create a structure of nested directories of depth {n}, where each directory holds {n} folders. Each folder should have a unique name.
                    
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
                    """
                    "When you have finished, submit the string 'submit'."""

model = gpt-4o-mini

scorer=

metadata="""
         Run 1. 
         Model: gpt-4o-mini
         Task: nested_dirs_challenge
         Prompt: v2, CoT self-check
         Score: numeric 0-3, based on bash commands results
         """

- not stored in log.eval



# Run 2 

Prompt_v2 += """IMPORTANT: After creating the directories, run 'ls -R' to verify your work.
                    If you don't see any output from 'ls -R', your directories weren't created correctly.
                    In that case, try again with the mkdir command.
                    
                    VERIFICATION STEPS:
                    1. Create the directory structure
                    2. Run 'ls -R' to verify
                    3. If no output, try again with different commands
                    4. When successful, submit the string 'submit'""",


Can we count how many commands agent required?

Added 2nd scorer - command_efficiency 

Add metadata to eval log?