# Setup

You can run this code on your own machine or on Google Colab.

1. **Local option:** If you choose to run locally, you will need to install MuJoCo and some Python packages; see [installation.md](../hw1/installation.md) from homework 1 for instructions.

1. **Colab:** The first few sections of the notebook will install all required dependencies. You can try out the Colab option by clicking the badges below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LeCAR-Lab/16831-S24-HW/blob/main/hw3/rob831/scripts/run_hw3_dqn.ipynb) **Part I (Q-learning)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LeCAR-Lab/16831-S24-HW/blob/main/hw3/rob831/scripts/run_hw3_actor_critic.ipynb)     **Part II (Actor-critic)**

For `fatal error: GL/osmesa.h: No such file or directory`, see [this](https://github.com/ethz-asl/reinmav-gym/issues/35).

## Complete the code

<!-- The following files have blanks to be filled with your solutions from homework 1. The relevant sections are marked with `TODO: get this from hw1 or hw2`.

- [infrastructure/rl_trainer.py](rob831/infrastructure/rl_trainer.py)
- [infrastructure/utils.py](rob831/infrastructure/utils.py)
- [policies/MLP_policy.py](rob831/policies/MLP_policy.py) -->

You will then need to implement new routines in the following files for homework 3 part 1 (Q-learning):

- [agents/dqn_agent.py](rob831/agents/dqn_agent.py)
- [critics/dqn_critic.py](rob831/critics/dqn_critic.py)
- [policies/argmax_policy.py](rob831/policies/argmax_policy.py)

and in the following files for part 2 (actor-critic):

- [agents/ac_agent.py](rob831/agents/ac_agent.py)
- [critics/bootstrapped_continuous_critic.py](rob831/critics/bootstrapped_continuous_critic.py)
- [policies/MLP_policy.py](rob831/policies/MLP_policy.py)

The relevant sections are marked with `TODO`.

You may also want to look through [scripts/run_hw3_dqn.py](rob831/scripts/run_hw3_dqn.py) and [scripts/run_hw3_actor_critic.py](rob831/scripts/run_hw3_actor_critic.py) (if running locally) or [scripts/run_hw3_dqn.ipynb](rob831/scripts/run_hw3_dqn.ipynb) and [scripts/run_hw3_actor_critic.ipynb](rob831/scripts/run_hw3_actor_critic.ipynb) (if running on Colab), though you will not need to edit this files beyond changing runtime arguments in the Colab notebook.

See the assignment PDF for more details on what files to edit.
