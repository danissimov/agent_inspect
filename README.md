# LLM agent evaluation framework

Evaluation of agent with bash tool creating nested folders. 
Built within SPAR AI safety mentorship program 

# The task 

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

# The full task description: 
https://docs.google.com/document/d/1QJx7Ys0z7Vpy1sZeuCHtXeko2z9lj2_XAnExZMRpPIM/edit?tab=t.0


All the code is here:
- directories_task.py

To access eval logs run:
- inspect view

Environment used locally:
- conda activate redvenv



Plan 
- 


Exp logs:

1. Run manual experiments with fast model 



Todos:
1. Can we count how many commands agent required?

Added 2nd scorer - command_efficiency  
- but it still fails to work 

2. Add metadata to eval log?
- .eval erases it :(


# Programmatic access to eval object problem: 

Log object type: <class 'inspect_ai._eval.eval.EvalLogs'>
Log object attributes: ['__add__', '__class__', '__class_getitem__', '__contains__', '__delattr__', '__delitem__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__getstate__', '__gt__', '__hash__', '__iadd__', '__imul__', '__init__', '__init_subclass__', '__iter__', '__le__', '__len__', '__lt__', '__module__', '__mul__', '__ne__', '__new__', '__orig_bases__', '__reduce__', '__reduce_ex__', '__repr__', '__reversed__', '__rmul__', '__setattr__', '__setitem__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_ipython_display_', 'append', 'clear', 'copy', 'count', 'extend', 'index', 'insert', 'pop', 'remove', 'reverse', 'sort']
Log object size: N/A bytes
Completed evaluation
Log saved to logs/directory_task_20250423_143429.json using __dict__ attribute