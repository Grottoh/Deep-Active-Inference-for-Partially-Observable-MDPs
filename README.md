# Deep Active Inference for Partially Observable MDPs

This repository contains the code and results for the algorithms described in https://arxiv.org/abs/2009.03622 <sup>[1]</sup>:

[1] Himst, O., & Lanillos, P. (2020) Deep Active Inference for Partially Observable MDPs. arXiv preprint arXiv:2009.03622.

## Usage
The code contains implementations of an:
- Active inference agent designed for fully observable MDPs (ai_mdp_agent.py)
- Active inference agent designed for partially observable MDPs (ai_pomdp_agent.py)
- Deep Q agent designed for fully observable MDPs (dq_mdp_agent.py)
- Deep Q agent designed for partially observable MDPs (dq_pomdp_agent.py)

Use the following command line argument to run an (active inference POMDP) agent with the default parameters (the default being the parameters used in [1]):

```
	python ai_pomdp_agent.py
```
Note that when using the default parameters, the folders *logs*, *networks*, *results* and their subfolders are required,
 these folders may be empty (except when running default *ai_pomdp_agent.py*, which requires the file *vae_n32_end.pth* stored in the *pre_trained_vae* folder).

Specify custom parameters like:

```
	python ai_pomdp_agent.py device=cuda:0 load_pre_trained_vae=False pre_train_vae=True
```

Each implementation contains a function ```set_parameters(...)``` where its parameters are listed and described.
