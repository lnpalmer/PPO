import argparse
import torch.optim as optim
from baselines.common.vec_env.vec_frame_stack import VecFrameStack

from envs import make_env, RenderSubprocVecEnv
from models import AtariCNN
from ppo import PPO


parser = argparse.ArgumentParser(description='PPO', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('env_id', type=str, help='Gym environment id')
parser.add_argument('--num-workers', type=int, default=8, help='number of parallel actors')
parser.add_argument('--arch', type=str, default='cnn', help='policy architecture, {lstm, cnn}')
parser.add_argument('--lr', type=float, default=2.5e-4, help='learning rate')
parser.add_argument('--seed', type=int, default=0, help='random seed')
args = parser.parse_args()

env_fns = []
for rank in range(args.num_workers):
    env_fns.append(lambda: make_env(args.env_id, rank, args.seed + rank))
env = RenderSubprocVecEnv(env_fns)
env = VecFrameStack(env, 4)

policy = {'cnn': AtariCNN}[args.arch](env.action_space.n)

optimizer = optim.Adam(policy.parameters(), lr=args.lr)

algorithm = PPO(policy, env, optimizer)
algorithm.run()
