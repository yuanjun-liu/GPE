# GPE

This the official code for "GPE: Global Position Embedding for Trajectory Similarity Computation"

The core file of the GPE method is at  `./traj/GPE/GPE.py`

## prepare

### requirements

```bash
pip install -r requirements.txt
```

optional: (for GAT architecture)

```bash
conda install pyg -c pyg
```

### dataset

put data into `../KyData/TrajData/T-drive` like path,

or modify the base path at `_tool/mFile.py`


## run

### main results (on LSTM)

```bash

python traj/GPE/exp_lstm.py

```

will automatically 
- pre-process dataset
- pre-train
- train
- print result table

including
- local scenario results 
- global scenario results 
- results about parameter 
- results about embedding size 


### results on GAT and Transformer

```bash

python traj/exp_gnn_tf/exp_gnn.py

python traj/exp_gnn_tf/exp_tf.py

```

including
- local scenario results 
- global scenario results 

### analyses

base selection

```bash

python traj/exp_base/exp_bases.py

```

parameter 𝜖, embedding size h, running time

```bash

python traj/GPE/exp_lstm.py

```

robust to noise (run after main result on LSTM)

```bash

python traj/GPE/exp_noise.py

```

case study on [ST2Vec](https://github.com/zealscott/ST2Vec/)

- run the ST2Vec code
- modify the ST2Vec based on my patch at `traj/exp_st2vec`
- run the ST2Vec code

