# A Pragmatic Look at Deep Imitation Learning

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

Off-policy imitation learning algorithms (with SAC [[1, 2]](#references)):

- AdRIL [[3]](#references)
- BC [[4]](#references)
- DRIL [[5]](#references) (with BC auxiliary loss; default true)
- GAIL [[6]](#references)
- GMMIL [[7]](#references) (with optional self-similarity term [[8]](#references))
- PWIL [[9]](#references) (nofill variant)
- RED [[10]](#references)
- SQIL [[11]](#references)

General options include:

- State-only imitation learning: `imitation.state-only: true/false`
- Absorbing state indicator [[12]](#references): `imitation.absorbing: true/false`
- BC auxiliary loss: `imitation.bc: true/false`

GAIL options include:

- Reward shaping (AIRL) [[13]](#references): `imitation.model.reward_shaping: true/false`
- Subtract log π(a|s) (AIRL) [[13]](#references): `imitation.model.subtract_log_policy: true/false`
- Reward functions (GAIL/AIRL/FAIRL) [[6, 13, 14]](#references): `imitation.model.reward_function: AIRL/FAIRL/GAIL`
- Gradient penalty [[12, 15]](#references): `imitation.grad_penalty: <float>`
- Spectral normalisation [[16]](#references): `imitation.spectral_norm: true/false`
- Entropy bonus [[17]](#references): `imitation.entropy_bonus: <float>`
- Loss functions (BCE/Mixup/nn-PUGAIL) [[6, 18, 19]](#references): `imitation.loss_function: BCE/Mixup/PUGAIL`

## Requirements

Requirements can be installed with:
```sh
pip install -r requirements.txt
```
Notable required packages are [PyTorch](https://pytorch.org/), [OpenAI Gym](https://gym.openai.com/), [D4RL-PyBullet](https://github.com/takuseno/d4rl-pybullet) and [Hydra](https://hydra.cc/). [Ax](https://ax.dev/) and the [Hydra Ax sweeper plugin](https://hydra.cc/docs/next/plugins/ax_sweeper/) are required for hyperparameter optimisation; if unneeded they can be removed from `requirements.txt`.

## Run

The training of each imitation learning algorithm (or SAC with the real environment reward) can be started with:
```sh
python main.py algorithm=ALG/ENV
```
where `ALG` is one of `[AdRIL|BC|DRIL|GAIL|GMMIL|PWIL|RED|SAC|SQIL]` and `ENV` is one of `[ant|halfcheetah|hopper|walker2d]`. For example:
```sh
python main.py algorithm=GAIL/hopper
```

Hyperparameters can be found in `conf/config.yaml` and `conf/algorithm/ALG/ENV.yaml`, with the latter containing algorithm- and environment-specific hyperparameters that were tuned with Ax.

Results will be saved in `outputs/ENV_ALGO/m-d_H-M-S` with the last subfolder indicating the current datetime.

### Hyperparameter optimisation

Hyperparameter optimisation can be run by adding `-m hydra/sweeper=ax hyperparam_opt=ALG`, for example:
```sh
python main.py -m algorithm=GAIL/hopper hydra/sweeper=ax hyperparam_opt=GAIL
```
`hyperparam_opt` specifies the hyperparameter search space.

### Seed sweep

A seed sweep can be performed as follows:
```sh
python main.py -m algorithm=GAIL/hopper seed=1,2,3,4,5 
```
or via the existing bash script:
```sh
./scripts/run_seed_experiments.sh ALG ENV
```

The results will be available in `./outputs/seed_sweeper_ENV_ALG` folder (note that running this code twice will overwrite the previous results).

## Results

![PyBullet results](figures/pybullet.png) 

## Acknowledgements

- [@ikostrikov](https://github.com/ikostrikov) for [https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail)

## Citation

If you find this work useful and would like to cite it, please use the following:

```tex
@article{arulkumaran2021pragmatic,
  author = {Arulkumaran, Kai and Ogawa Lillrank, Dan},
  title = {A Pragmatic Look at Deep Imitation Learning},
  journal={arXiv preprint arXiv:2108.01867},
  year = {2021}
}
```

## References

[1] [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)  
[2] [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)  
[3] [Of Moments and Matching: A Game-Theoretic Framework for Closing the Imitation Gap](https://arxiv.org/abs/2103.03236)  
[4] [Efficient Training of Artificial Neural Networks for Autonomous Navigation](https://www.mitpressjournals.org/doi/abs/10.1162/neco.1991.3.1.88?journalCode=neco)  
[5] [Disagreement-Regularized Imitation Learning](https://openreview.net/forum?id=rkgbYyHtwB)  
[6] [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476)  
[7] [Imitation Learning via Kernel Mean Embedding](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/16807)  
[8] [A Pragmatic Look at Deep Imitation Learning](https://arxiv.org/abs/2108.01867)  
[9] [Primal Wasserstein Imitation Learning](https://arxiv.org/abs/2006.04678)  
[10] [Random Expert Distillation: Imitation Learning via Expert Policy Support Estimation](https://arxiv.org/abs/1905.06750)  
[11] [SQIL: Imitation Learning via Reinforcement Learning with Sparse Rewards](https://arxiv.org/abs/1905.11108)  
[12] [Discriminator-Actor-Critic: Addressing Sample Inefficiency and Reward Bias in Adversarial Imitation Learning](https://arxiv.org/abs/1809.02925)  
[13] [Learning Robust Rewards with Adversarial Inverse Reinforcement Learning](https://arxiv.org/abs/1710.11248)  
[14] [A Divergence Minimization Perspective on Imitation Learning Methods](https://arxiv.org/abs/1911.02256)  
[15] [Sample-Efficient Imitation Learning via Generative Adversarial Nets](https://arxiv.org/abs/1809.02064)  
[16] [Lipschitzness Is All You Need To Tame Off-policy Generative Adversarial Imitation Learning](https://arxiv.org/abs/2006.16785)  
[17] [What Matters for Adversarial Imitation Learning?](https://arxiv.org/abs/2106.00672)  
[18] [Batch Exploration with Examples for Scalable Robotic Reinforcement Learning](https://arxiv.org/abs/2010.11917)  
[19] [Positive-Unlabeled Reward Learning](https://arxiv.org/abs/1911.00459)  
