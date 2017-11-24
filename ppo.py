import copy
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from torch.autograd import Variable

from utils import gae


class PPO:
    def __init__(self, policy, venv, optimizer, clip=.1, gamma=.99, lambd=.95,
                 worker_steps=128, sequence_steps=32, batch_steps=256, opt_epochs=3):
        """ Proximal Policy Optimization algorithm class

        Evaluates a policy over a vectorized environment and
        optimizes over policy, value, entropy objectives.

        Assumes discrete action space.

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
        self.opt_epochs = opt_epochs

        self.objective = PPOObjective(clip)
        self.gamma = gamma
        self.lambd = lambd

        self.last_ob = self.venv.reset()

    def run(self, total_steps):
        """ Runs PPO

        Args:
            total_steps (int): total number of environment steps to run for
        """
        taken_steps = 0

        N = self.num_workers
        T = self.worker_steps
        E = self.opt_epochs

        while taken_steps < total_steps:
            obs, rewards, masks, actions, steps = self.interact()

            # compute advantages, returns with GAE
            # TEMP upgrade to support recurrence
            obs = obs.view(((T + 1) * N,) + obs.size()[2:])
            obs = Variable(obs)
            _, values = self.policy(obs)
            values = values.view(T + 1, N, 1)
            advantages, returns = gae(rewards, masks, values, self.gamma, self.lambd)

            for e in range(E):
                raise NotImplementedError

            taken_steps += steps

    def interact(self):
        """ Interacts with the environment

        Returns:
            obs (FloatTensor): observations shaped [T + 1 x N x ...]
            rewards (FloatTensor): rewards shaped [T x N x 1]
            masks (FloatTensor): continuation masks shaped [T x N x 1]
                zero at done timesteps, one otherwise
            actions (LongTensor): discrete actions shaped [T x N x 1]
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
    def __init__(self, clip):
        self.clip = clip

    def forward(self, pi, v, pi_old, v_old, action, advantage, returns):
        """ Computes PPO objectives

        Assumes discrete action space.

        Args:
            pi (Variable): discrete action logits, shaped [N x num_actions]
            v (Variable): value predictions, shaped [N x 1]
            pi_old (Variable): old discrete action logits, shaped [N x num_actions]
            v_old (Variable): old value predictions, shaped [N x 1]
            action (Variable): discrete actions, shaped [N x 1]
            advantage (Variable): action advantages, shaped [N x 1]
            returns (Variable): discounted returns, shaped [N x 1]

        Returns:
            policy_loss (Variable): policy surrogate loss, shaped [1]
            value_loss (Variable): value loss, shaped [1]
            entropy_loss (Variable): entropy loss, shaped [1]
        """
        prob = Fnn.softmax(pi)
        log_prob = Fnn.log_softmax(pi)
        action_prob = prob.gather(1, action)

        prob_old = Fnn.softmax(pi_old)
        action_prob_old = prob_old.gather(1, action)

        ratio = action_prob / (action_prob_old + 1e-10)

        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)

        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, min=1. - self.clip, max=1. + self.clip) * advantage

        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = (.5 * (values - returns) ** 2.).mean()
        entropy_loss = (prob * log_prob).sum(1).mean()

        return policy_loss, value_loss, entropy_loss
