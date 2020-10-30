# CORRECT RESULTS
___________________________________________________________________________________________________________________
|    Check   |   Category  |                                 Description                                |  Grade  |
|------------|-------------|----------------------------------------------------------------------------|---------|
| errors.npy | Correctness |                                DO MANUALLY                                 |   -1    |
|____________|_____________|____________________________________________________________________________|_________|
# FORMATTING
____________________________________________________________________________________________________________________
|  Check  |  Category  |                                   Description                                   |  Grade  |
|---------|------------|---------------------------------------------------------------------------------|---------|
|  flake8 | Linting    |                     File(s) not fully linted with `flake8`                      |  1.5/2  |
|         |            |                               3 lines with lint.                                |         |
|         |            |                                                                                 |         |
|  black  | Formatting |                    File(s) correctly formatted with `black`                     |   2/2   |
|         |            |                                                                                 |         |
|_________|____________|_________________________________________________________________________________|_________|

# BONUS
____________________________________________________________________________________________________________________
|  Check  |   Category   |                                  Description                                  | Bonus % |
|---------|--------------|-------------------------------------------------------------------------------|---------|
|  isort  | Imports      |                     File correctly formatted with `isort`                     |    1    |
|         |              |                                                                               |         |
|         |              |                                                                               |         |
|  mypy   | Static Types |                       6 function(s) missing annotations                       |    0    |
|         |              |                     File has 0 other static type issue(s)                     |         |
|_________|______________|_______________________________________________________________________________|_________|

# DETAILS
-------
NOTE: `mypy` and `isort` messages below are just the outputs of running `mypy` and `isort` on your
submitted code. You don't lose marks if there are lots of outputs here, but they are things than you
can fix for free and/or bonus marks! If you see nothing below these lines, it means you've written
great code, with no significant formatting or typing errors.

```

========================================================================================================================
flake8
========================================================================================================================
   assign1.py:6:1: E266 too many leading '#' for block comment
   assign1.py:8:1: F401 'matplotlib.pyplot as plt' imported but unused
   assign1.py:11:1: F401 'seaborn as sbn' imported but unused


========================================================================================================================
mypy
========================================================================================================================
   assign1.py:17: error: Function is missing a type annotation
   assign1.py:62: error: Function is missing a type annotation
   assign1.py:96: error: Function is missing a type annotation
   assign1.py:114: error: Function is missing a type annotation
   assign1.py:124: error: Function is missing a type annotation
   assign1.py:140: error: Function is missing a type annotation
   Found 6 errors in 1 file (checked 1 source file)


```
