# Understand-Transfer-QO

<!-- Code cleaning is in procedure. Cleaned version will be available before May 7. -->

This is the codebase of the paper "Towards Understanding and Transferring Query Optimizers". 

## Requirements
Sampling random queries and executing benchmark queries require the database systems deployed with data loaded. We use DBMS including [Postgres](), [Microsoft SQL Server]() and [Couchbase](). We use SSB, TPC-H, and IMDB as data sets.

We use [Python3]() and [jupyter notebook]() to implement the experiments. The python requirements are listed in `requirements.txt` (for ML experiments only) and `requirements_engine.txt` (for connecting DB engines). Run the following to install the packages needed for ML experiments.

```python3 -m pip install -r requirements.txt``` 

The experiments requires the codes in `core/`. Make sure that your python lib path includes `core/`. For example, 

```export PYTHONPATH="$PYTHONPATH:/YOUR_REPO_DIR/"```


## Sampling random queries and collect training data
Since we include the pregenerated queries and their corresponding training data, you can safely skip this phase. The generated data are under `sample_results/`. 

We use `core/QuerySampler.py` to sample the queries and obtain their featurized vector.  Usage: 

```
python3 core/QuerySampler.py -d LIST OF DATA SETS -e LIST OF DBMS  -n NUM OF SAMPLES
```

* LIST OF DATA SETS: a list of data sets from [ssb, tpch, tpch_10, tpch_100, imdb].
* LIST OF DBMS: a list of DBMS (engines) from [postgres, mssql].
* NUM OF SAMPLES: the number of samples per binary join.

For example, to sample 2000 random queries for each binary join for all the datasets on all engines,

```
python3 core/QuerySampler.py -e postgres mssql -d ssb tpch tpch_10 tpch_100 imdb  -n 2000
```

## Experimental evaluation 

### Training and evaluating ML characterization models
* This corresponds to Section 4.3 in the paper. The experiments are implemented in [exp1-ML-characterization.ipynb](experiments/exp1-ML-characterization.ipynb).

### Applying ML models to full queries
* This corresponds to Section 4.4 in the paper. The queries with hints are under `experiment/benchmark/queries/` (`postgres_original` and `postgres_model`). To run the queries, use `experiments/run_generalized_model_query.sh`. This requires Postgres being deployed on your systems and SSB, TPC-H, IMDB loaded. You can safely skip this and the run time results are in [exp1-ML-generalization.ipynb](experiments/exp1-ML-generalization.ipynb).

### Feature correlations
* This corresponds to Section 5.1 (feature correlation part) in the paper. You can find our code in [exp2-feature-corr.ipynb](experiments/exp2-feature-corr.ipynb).

### Ablation study
* This corresponds to Section 5.1 (ablation study part) in the paper. You can find our code in [exp2-abalation-study-postgres.ipynb](experiments/exp2-abalation-study-postgres.ipynb) for Postgres and [exp2-abalation-study-mssql.ipynb](experiments/exp2-abalation-study-mssql.ipynb) for Microsoft SQL Server.

### Training ML models with two key features
* This corresponds to Section 5.2 in the paper. You can find our code in [exp2-ML-with-key-features.ipynb](experiments/exp2-ML-with-key-features.ipynb).


### Visualizing decision spaces
* This corresponds to Section 5.3 in the paper. You can find the visualization in [exp2-plot-decision-space.ipynb](experiments/exp2-plot decision space.ipynb). The effect of sample weight is shown in [exp2-effect-of-sample-weights.ipynb](experiments/exp2-effect-of-sample-weights.ipynb).

### Transferring query optimizers
* This corresponds to Section 6 in the paper. The queries with hints are under `experiment/benchmark/queries/` (`couchbase_original` for Couchbase with CBO, `couchbase_trans_postgres` for transferring Postgres and `couchbase_trans_mssql` for transferring MSSQL). To run all the queries, use `experiments/run_couchbase_query.sh`. This requires Couchbase being deployed with SSB, TPC-H, IMDB loaded to it. The compare with the CBO in Couchbase, the Couhbase should be Enterprise Edition and the preview mode enabled. The results are listed in [exp3-transfter-learning-plot.ipynb](experiments/exp3-transfter-learning-plot.ipynb).
