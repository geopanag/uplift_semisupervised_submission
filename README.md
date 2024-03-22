# Uplift Learning under Limited Supervision

The code to reproduce the analysis for "Uplift Modeling under Limited Supervision".

Start by installing the requirements.
```bash
pip install -r requirements.txt
```

## Data
Download the [RetailHero](https://ods.ai/competitions/x5-retailhero-uplift-modeling/data) into the folder data/RetailHero and the [Movielens25](https://grouplens.org/datasets/movielens/25m/) into folder data/Movielens25.
Then run:

```bash
python prepare_data.py
```

## Run benchmarkcs

### Causal ML Benchmarks
The benchmarks utilize the respective config files, so the paths have to be adjusted accordingly.

```bash
python benchmarks.py config_RetailHero.json
```

### Netdeconf
We adjusted the code from the paper's [repository](https://github.com/rguo12/network-deconfounder-wsdm20) to address the current task and evaluation setting.

```bash
python netdeconf/main.py 
```

### Run
To run the experiments with the default settings and the default config (attached):

## UMGNN

```bash
python umgn.py
python umgn_test_models.py
python umgn_movielens.py
python umgn_dragon.py
```

## Active learning
```bash
python al_umgn_retail.py
python al_umgn_movielens.py
```
