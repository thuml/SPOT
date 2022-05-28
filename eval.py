import numpy as np
import gym
from log import Logger
from tqdm import trange

from utils import VideoRecorder

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment


def eval_policy(args, iter, video: VideoRecorder, logger: Logger, policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + seed_offset)

    lengths = []
    returns = []
    last_rewards = []
    avg_reward = 0.
    for episode in trange(eval_episodes):
        video.init(enabled=(args.save_video and _ == 0))
        state, done = eval_env.reset(), False
        video.record(eval_env)
        steps = 0
        episode_return = 0
        while not done:
            state = (np.array(state).reshape(1, -1) - mean) / std
            action = policy.select_action(state)
            state, reward, done, _ = eval_env.step(action)
            video.record(eval_env)
            avg_reward += reward
            episode_return += reward
            steps += 1
        lengths.append(steps)
        returns.append(episode_return)
        last_rewards.append(reward)
        video.save(f'eval_s{iter}_e{episode}_r{str(episode_return)}.mp4')
        if 'antmaze' in args.env:
            print("\tsuccess", float(steps != eval_env._max_episode_steps), "\tlast reward", reward)

    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward)

    logger.log('eval/lengths_mean', np.mean(lengths), iter)
    logger.log('eval/lengths_std', np.std(lengths), iter)
    logger.log('eval/returns_mean', np.mean(returns), iter)
    logger.log('eval/returns_std', np.std(returns), iter)
    logger.log('eval/d4rl_score', d4rl_score, iter)
    if 'antmaze' in args.env:
        logger.log('eval/success_rate', 1 - np.mean(np.array(lengths) == eval_env._max_episode_steps), iter)
        if 'dense' in args.env:
            logger.log('eval/last_reward_mean', np.mean(last_rewards), iter)
            logger.log('eval/last_reward_std', np.std(last_rewards), iter)

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {d4rl_score:.3f}")
    print("\tepisode returns:", *['%.2f' % x for x in returns])
    print("\tepisode lengths", lengths)
    if 'antmaze' in args.env:
        print("\tsuccess rate", 1 - np.mean(np.array(lengths) == eval_env._max_episode_steps))
        if 'dense' in args.env:
            print("\tlast reward", *['%.2f' % x for x in last_rewards])
    print("---------------------------------------")
    return d4rl_score
