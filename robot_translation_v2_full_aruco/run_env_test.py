

from main_rl_env_translation_v2 import RL_ENV

num_episodes     = 10
episode_horizont = 6


def main_run():
    env = RL_ENV()

    for episode in range(1, num_episodes+1):
        env.reset_env()
        for step in range(1, episode_horizont+1):
            print(f"-------Episode:{episode} Step:{step}---------")
            state, _ = env.state_space_function()
            action   = env.generate_sample_act()
            env.env_step(action)  # take the action
            next_state, image_state = env.state_space_function()
            reward, done = env.calculate_reward()
            env.env_render(image=image_state, episode=episode, step=step, done=done)


if __name__ == '__main__':
    main_run()
'''

from MBPO_TD3 import *
from MBPO_DDPG import *


if __name__ == '__main__':
    main_run("MFRL")
    main_run_ddpg("MFRL")

'''