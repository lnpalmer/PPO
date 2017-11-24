import copy
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from torch.autograd import Variable

class PPO:
    def __init__(self, policy, venv, optimizer, clip=.1, gamma=.99, lambd=.95,
                 worker_steps=128, sequence_steps=32, batch_steps=256):
        """ Proximal Policy Optimization algorithm class

        Evaluates a policy over a vectorized environment and
        optimizes over policy, value, entropy objectives.

        Args:
            policy (nn.Module): the policy to optimize
            venv (vec_env): the vectorized environment to use
            optimizer (optim.Optimizer): the optimizer to use
            clip (float): probability ratio clipping range
            gamma (float): discount factor
            lambd (float): GAE lambda parameter
            worker_steps (int): steps per worker between optimization rounds
            sequence_steps (int): steps per sequence (for backprop through time)
            batch_steps (int): steps per sequence (for backprop through time)
        """
        self.policy = policy
        self.policy_old = copy.deepcopy(policy)
        self.venv = venv
        self.optimizer = optimizer

        self.num_workers = venv.num_envs
        self.worker_steps = worker_steps
        self.sequence_steps = sequence_steps
        self.batch_steps = batch_steps

        self.objective = PPOObjective(clip, gamma, lambd)

        self.last_ob = self.venv.reset()

    def run(self, total_steps):
        """ Runs PPO

        Args:
            total_steps (int): total number of environment steps to run for
        """
        taken_steps = 0

        while taken_steps < total_steps:
            obs, rewards, dones, actions, steps = self.interact()

            print(obs.size())

            taken_steps += steps

    def interact(self):
        """ Interacts with the environment

        Returns:
            obs (torch.FloatTensor): observations shaped [T + 1 x N x ...]
            rewards (torch.FloatTensor): rewards shaped [T x N x 1]
            masks (torch.FloatTensor): continuation masks shaped [T x N x 1]
                zero at done timesteps, one otherwise
            actions (torch.LongTensor): discrete actions shaped [T x N x 1]
            steps (int): total number of steps taken
        """
        N = self.num_workers
        T = self.worker_steps

        # TEMP needs to be generalized, does conv-specific transpose for PyTorch
        obs = torch.zeros(T + 1, N, 4, 84, 84)
        rewards = torch.zeros(T, N, 1)
        masks = torch.zeros(T, N, 1)
        actions = torch.zeros(T, N, 1).long()

        for t in range(T):
            ob = torch.from_numpy(self.last_ob.transpose((0, 3, 1, 2))).float()
            ob = Variable(ob / 255.)
            obs[t] = ob.data

            pi, v = self.policy(ob)
            prob = Fnn.softmax(pi)
            action = prob.multinomial().data
            actions[t] = action

            self.last_ob, reward, done, _ = self.venv.step(action.cpu().numpy())
            rewards[t] = torch.from_numpy(reward).unsqueeze(1)
            mask = torch.from_numpy((1. - done)).unsqueeze(1)
            masks[t] = mask

        ob = torch.from_numpy(self.last_ob.transpose((0, 3, 1, 2))).float()
        ob = Variable(ob / 255.)
        obs[T] = ob.data

        steps = N * T

        return obs, rewards, masks, actions, steps

class PPOObjective(nn.Module):
    def __init__(self, clip, gamma, lambd):
        self.clip = clip
        self.gamma = gamma
        self.lambd = lambd

    def forward(self, inputs):
        raise NotImplementedError
