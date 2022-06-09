# Rnorm

Code for the experimental component of the paper "Intrinsic dimensionality and generalization properties of the $\mathcal{R}$-norm inductive bias" **TODO: Add arxiv link** by Clayton Sanford*, Navid Ardeshir*, and Daniel Hsu. 
We repeat similar experiments to ["Finding the Needle in the Haystack with Convolutions: on the benefits of architectural bias"](https://arxiv.org/abs/1906.06766) by Stéphane d'Ascoli, Levent Sagun, Joan Bruna and Giulio Biroli to convert a convolutional network (CNN) to its equivalent locally-connected network (eLCN), at different stages and provide evidence for low norm implicit bias of SGD.
Our repository borrows heavily from and adapts their code in [sdascoli/anarchitectural-search](https://github.com/sdascoli/anarchitectural-search).

![plot](CNN_vs_LCN_body.png)

# Requirements

To install requirements :
```pip install -r requirements.txt```

Our results were obtained in Python 3.9.7 environment.

# Generate Results
Python files can be used to generate the experiments. It stores networks weights at designated epochs into ```./results``` directory structured as the following:
- CNN weights at every epoch are stored in `skinnyprime_<epoch>.pyT`.
- LCN detaching from CNN at certain epochs are stored under 'skinnyprime_lc_version_<split epoch>' directory.  
- Relevant historical data are stored in `evaluation-history.hist` at the end of training of each network.

```python experiments.py --epochs <num epochs> --n_splits <num splits> --cnn_lr <CNN learning rate> --lc_lr <LCN learning rate> --exec <execution flag>```

| Arguments  | Interpretation | Default values |
|------------|----------------|-----------------|
| epochs     | Number of epochs| 150 |
| n_splits   | number of splits used to produce (almost) logarithmically spaced epochs for translation from CNN to eLCN| 20 |
| cnn_lr     | CNN learning rate| 0.05 |
| lcn_lr     | LCN learning rate| 0.01|
| exec       | Execution flag   | 1   |

To run our experiments in parallel (with GPUs) on a Slurm based cluster use:
```./run_all.sh```
or,
```./job.sh```



