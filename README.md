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