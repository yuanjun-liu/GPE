# GPE

This the official code for "GPE: Global Position Embedding for Trajectory Similarity Computation"

The core file of the GPE method is at  `./traj/GPE/GPE.py`

## prepare

### requirements
```bash
pip install -r requirements.txt
```

### dataset

put data into `../KyData/TrajData/T-drive` like path,

or modify the base path at `_tool/mFile.py`


## run

```bash

python traj/GPE/_main.py

```

will automatically 
- pre-process dataset
- pre-train
- train
- print result table



