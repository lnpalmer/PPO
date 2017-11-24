import argparse
import torch.optim as optim
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack

from envs import make_env, RenderSubprocVecEnv
from models import AtariCNN
from ppo import PPO


parser = argparse.ArgumentParser(description='PPO', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('env_id', type=str, help='Gym environment id')
parser.add_argument('--arch', type=str, default='cnn', help='policy architecture, {lstm, cnn}')
parser.add_argument('--num-workers', type=int, default=8, help='number of parallel actors')
parser.add_argument('--opt-epochs', type=int, default=3, help='optimization epochs between environment interaction')
parser.add_argument('--total-steps', type=int, default=int(10e6), help='total number of environment steps to take')
parser.add_argument('--worker-steps', type=int, default=128, help='steps per worker between optimization rounds')
parser.add_argument('--sequence-steps', type=int, default=32, help='steps per sequence (for backprop through time)')
parser.add_argument('--batch-steps', type=int, default=256, help='steps per optimization minibatch')
parser.add_argument('--lr', type=float, default=2.5e-4, help='learning rate')
parser.add_argument('--clip', type=float, default=.1, help='probability ratio clipping range')
parser.add_argument('--gamma', type=float, default=.99, help='discount factor')
parser.add_argument('--lambd', type=float, default=.95, help='GAE lambda parameter')
parser.add_argument('--value-coef', type=float, default=1., help='value loss coeffecient')
parser.add_argument('--entropy-coef', type=float, default=.01, help='entropy loss coeffecient')
parser.add_argument('--max-grad-norm', type=float, default=.5, help='grad norm to clip at')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--render', action='store_true', help='render training environments')
parser.add_argument('--render-interval', type=int, default=4, help='steps between environment renders')
args = parser.parse_args()

env_fns = []
for rank in range(args.num_workers):
    env_fns.append(lambda: make_env(args.env_id, rank, args.seed + rank))
if args.render:
    venv = RenderSubprocVecEnv(env_fns, args.render_interval)
else:
    venv = SubprocVecEnv(env_fns)
venv = VecFrameStack(venv, 4)

policy = {'cnn': AtariCNN}[args.arch](venv.action_space.n)

optimizer = optim.Adam(policy.parameters(), lr=args.lr)

algorithm = PPO(policy, venv, optimizer, clip=args.clip, gamma=args.gamma,
                lambd=args.lambd, worker_steps=args.worker_steps,
                sequence_steps=args.sequence_steps,
                batch_steps=args.batch_steps,
                value_coef=args.value_coef, entropy_coef=args.entropy_coef,
                max_grad_norm=args.max_grad_norm)
algorithm.run(args.total_steps)
