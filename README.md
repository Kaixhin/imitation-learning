IL
==
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

Imitation learning algorithms (with PPO [[1]](#references)):

- GAIL [[2]](#references)
- ~~GMMIL [[3]](#references)~~
- AIRL [[4]](#references)
- ~~RED [[5]](#references)~~

```
python main.py --imitation [AIRL|GAIL]
```

Results
-------

**PPO**

Train | Test
:----:|:---:
![ppo_train_returns](figures/ppo_train_returns.png) | ![ppo_test_returns](figures/ppo_test_returns.png)

**GAIL**

Train | Test
:----:|:---:
![gail_train_returns](figures/gail_train_returns.png) | ![gail_test_returns](figures/gail_test_returns.png)

**AIRL**

Train | Test
:----:|:---:
![airl_train_returns](figures/airl_train_returns.png) | ![airl_test_returns](figures/airl_test_returns.png)

Acknowledgements
----------------

- [@ikostrikov](https://github.com/ikostrikov) for [https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail)

References
----------

[1] [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)  
[2] [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476)  
[3] [Imitation Learning via Kernel Mean Embedding](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/16807)  
[4] [Learning Robust Rewards with Adversarial Inverse Reinforcement Learning](https://arxiv.org/abs/1710.11248)  
[5] [Random Expert Distillation: Imitation Learning via Expert Policy Support Estimation](https://arxiv.org/abs/1905.06750)
