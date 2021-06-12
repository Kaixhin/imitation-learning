A Pragmatic Look at Deep Imitation Learning
===========================================
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

Imitation learning algorithms (with PPO [[1]](#references)):

- ~~ABC [[2]](#references)~~
- AIRL [[3]](#references)
- BC [[4]](#references)
- DRIL [[5]](#references)
- FAIRL [[6]](#references)
- GAIL [[7]](#references)
- GMMIL [[8]](#references)
- RED [[11]](#references)


Requirements
------------
The code runs on Python3.7 (AX requires >3.7). You can install most of the requirements by running 
```
pip install -r requirements.txt
```
Notable required packages are PyTorch, OpenAI gym, Hydra with AX,  and [D4RL-pybullet](https://github.com/takuseno/d4rl-pybullet).
if you fail to install d4rl-pybullet, install it with pip directly from git by using the command 
```
pip install git+https://github.com/takuseno/d4rl-pybullet
```
### Note:
For hyperparameter optimization, [Hydra and AX](https://hydra.cc/docs/next/plugins/ax_sweeper/) is used. Ax requires a specific version of PyTorch, 
and therefore might upgrade/downgrade the PyTorch if you install it on existing environment.


Run
---
The training of each imitation learning algorithm can be started with 
```
python main.py algorithm=ALG/ENV
```
`ALG` one of `[AIRL|BC|DRIL|FAIRL|GAIL|GMMIL|PUGAIL|RED]` and `ENV` to be one of `[ant|halfcheetah|hopper|walker2d]`.
example:
```
python main.py algorithm=AIRL/hopper
```
Hyperparameters can be found in `conf/config.yaml` and `conf/algorithm/ALG/ENV.yaml`, 
with the latter containing algorithm & environment specific hyperparameter that was tuned with AX.

The resulting model will saved in `repo_root/outputs/ENV_ALGO/m-d-H-M` with the last subfolder indicating current date (month-day-hour-minute).

### Run hyperparameter optimization
Hyper parameter optimization can be run by adding the `-m` flag. 

example:
```
python main.py -m algorithm=AIRL/hopper hyperparam_opt=AIRL
```
The last argument specifies *which* parameters to optimize. (Default is IL and contains all parameters).
### Run with seeds
You can run each algorithm with different seeds with:
```
python main.py -m algorithm=AIRL/hopper seed=1, 2, 3, 4, 5 hyperparam_opt=empty hydra/sweeper=base
```
or use the existing bash script
```bash
./run_seed_experiments.sh ALG ENV
```
The results will be available in `./output/seed_sweeper_ENV_ALG` folder (note: running this code twice will overwrite the previous result).

Options that can be modified in config include:

- State-only imitation learning: `state-only: true/false`
- Absorbing state indicator [[12]](#references): `absorbing: true/false`
- R1 gradient regularisation [[13]](#references): `r1-reg-coeff: 0.5` (in each algorithm subfolder)

The state only & absorbing is not used in the result.
 
Results
-------

![all_training_result](figures/result_fig.png) 

Acknowledgements
----------------

- [@ikostrikov](https://github.com/ikostrikov) for [https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail)

Citation
--------

If you find this work useful and would like to cite it, the following would be appropriate:

```
@misc{arulkumaran2020pragmatic,
  author = {Arulkumaran, Kai and Ogawa Lillrank, Dan},
  title = {A Pragmatic Look at Deep Imitation Learning},
  url = {https://github.com/Kaixhin/imitation-learning},
  year = {2020}
}
```

References
----------

[1] [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)  
[2] [Adversarial Behavioral Cloning](https://www.tandfonline.com/doi/abs/10.1080/01691864.2020.1729237)  
[3] [Learning Robust Rewards with Adversarial Inverse Reinforcement Learning](https://arxiv.org/abs/1710.11248)  
[4] [Efficient Training of Artificial Neural Networks for Autonomous Navigation](https://www.mitpressjournals.org/doi/abs/10.1162/neco.1991.3.1.88?journalCode=neco)  
[5] [Disagreement-Regularized Imitation Learning](https://openreview.net/forum?id=rkgbYyHtwB)  
[6] [A Divergence Minimization Perspective on Imitation Learning Methods](https://arxiv.org/abs/1911.02256)  
[7] [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476)  
[8] [Imitation Learning via Kernel Mean Embedding](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/16807)  
[9] [Positive-Unlabeled Reward Learning](https://arxiv.org/abs/1911.00459)  
[10] [Primal Wasserstein Imitation Learning](https://arxiv.org/abs/2006.04678)  
[11] [Random Expert Distillation: Imitation Learning via Expert Policy Support Estimation](https://arxiv.org/abs/1905.06750)  
[12] [Discriminator-Actor-Critic: Addressing Sample Inefficiency and Reward Bias in Adversarial Imitation Learning](https://arxiv.org/abs/1809.02925)  
[13] [Which Training Methods for GANs do actually Converge?](https://arxiv.org/abs/1801.04406)  
