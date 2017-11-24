import copy
import torch
import torch.nn as nn


class PPO:
    def __init__(self, policy, env, optimizer):
        """ Proximal Policy Optimization algorithm class

        Evaluates a policy over a vectorized environment and
        optimizes over policy, value, entropy objectives.

        Args:
            policy (nn.Module): the policy to optimize
            env (vec_env): the vectorized environment to use
            optimizer (optim.Optimizer): the optimizer to use
        """
        self.policy = policy
        self.policy_old = copy.deepcopy(policy)
        self.env = env
        self.optimizer = optimizer

        self.objective = PPOObjective()

        raise NotImplementedError

    def run(self, total_steps):
        """ Runs PPO

        Args:
            total_steps (int): total number of environment steps to run for
        """
        raise NotImplementedError


class PPOObjective(nn.Module):
    def __init__(self):
        raise NotImplementedError

    def forward(self, inputs):
        raise NotImplementedError
