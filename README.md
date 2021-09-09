# A Pragmatic Look at Deep Imitation Learning

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

Imitation learning algorithms (with PPO [[1]](#references)):

- AIRL [[2]](#references)
- BC [[3]](#references)
- DRIL [[4]](#references) (without BC)
- FAIRL [[5]](#references)
- GAIL [[6]](#references)
- GMMIL [[7]](#references) (including an optional self-similarity term [[8]](#references))
- nn-PUGAIL [[9]](#references)
- RED [[10]](#references)

Options include:

- State-only imitation learning: `state-only: true/false`
- R1 gradient regularisation [[11]](#references): `r1-reg-coeff: 0.5`

## Requirements

Requirements can be installed with:
```sh
pip install -r requirements.txt
```
Notable required packages are [PyTorch](https://pytorch.org/), [OpenAI Gym](https://gym.openai.com/), [D4RL-PyBullet](https://github.com/takuseno/d4rl-pybullet) and [Hydra](https://hydra.cc/). [Ax](https://ax.dev/) and the [Hydra Ax sweeper plugin](https://hydra.cc/docs/next/plugins/ax_sweeper/) are required for hyperparameter optimisation; if unneeded they can be removed from `requirements.txt`.

## Run

The training of each imitation learning algorithm can be started with:
```sh
python main.py algorithm=ALG/ENV
```
where `ALG` is one of `[AIRL|BC|DRIL|FAIRL|GAIL|GMMIL|PUGAIL|RED]` and `ENV` is one of `[ant|halfcheetah|hopper|walker2d]`. For example:
```sh
python main.py algorithm=AIRL/hopper
```

Hyperparameters can be found in `conf/config.yaml` and `conf/algorithm/ALG/ENV.yaml`, with the latter containing algorithm- and environment-specific hyperparameters that were tuned with Ax.

Results will be saved in `outputs/ENV_ALGO/m-d_H-M-S` with the last subfolder indicating the current datetime.

### Hyperparameter optimisation

Hyperparameter optimisation can be run by adding `-m hydra/sweeper=ax hyperparam_opt=ALG`, for example:
```sh
python main.py -m algorithm=AIRL/hopper hydra/sweeper=ax hyperparam_opt=AIRL 
```
`hyperparam_opt` specifies the hyperparameter search space.

### Seed sweep

A seed sweep can be performed as follows:
```sh
python main.py -m algorithm=AIRL/hopper seed=1,2,3,4,5 
```
or via the existing bash script:
```sh
./scripts/run_seed_experiments.sh ALG ENV
```

The results will be available in `./output/seed_sweeper_ENV_ALG` folder (note that running this code twice will overwrite the previous results).

## Results

![PyBullet results](figures/pybullet.png) 

## Acknowledgements

- [@ikostrikov](https://github.com/ikostrikov) for [https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail)

## Citation

If you find this work useful and would like to cite it, the following would be appropriate:

```tex
@article{arulkumaran2021pragmatic,
  author = {Arulkumaran, Kai and Ogawa Lillrank, Dan},
  title = {A Pragmatic Look at Deep Imitation Learning},
  journal={arXiv preprint arXiv:2108.01867},
  year = {2021}
}
```

## References

[1] [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)  
[2] [Learning Robust Rewards with Adversarial Inverse Reinforcement Learning](https://arxiv.org/abs/1710.11248)  
[3] [Efficient Training of Artificial Neural Networks for Autonomous Navigation](https://www.mitpressjournals.org/doi/abs/10.1162/neco.1991.3.1.88?journalCode=neco)  
[4] [Disagreement-Regularized Imitation Learning](https://openreview.net/forum?id=rkgbYyHtwB)  
[5] [A Divergence Minimization Perspective on Imitation Learning Methods](https://arxiv.org/abs/1911.02256)  
[6] [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476)  
[7] [Imitation Learning via Kernel Mean Embedding](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/16807)  
[8] [A Pragmatic Look at Deep Imitation Learning](https://arxiv.org/abs/2108.01867)  
[9] [Positive-Unlabeled Reward Learning](https://arxiv.org/abs/1911.00459)  
[10] [Random Expert Distillation: Imitation Learning via Expert Policy Support Estimation](https://arxiv.org/abs/1905.06750)  
[11] [Which Training Methods for GANs do actually Converge?](https://arxiv.org/abs/1801.04406)  
