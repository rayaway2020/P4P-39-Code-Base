"""
Author: David Valencia
Date:   03/ 05 /2022
Update: 10/05/2022

Describer:
            This file runs the robot test bed RL environment very straight forward, following the same concept of
            a Gym-OpenAi env environment
            The core functions are in the main_rl_env.py file

            If the number of step in each episode finished the environment will reset so the robot will
            move to the home position and a new goal/target point will be generated

            File for testing the environment, take random actions without RL algorithm
"""
from main_rl_env import *

num_episodes     = 10
episode_horizont = 3


def main_run():

    env = RL_ENV()

    for episode in range(num_episodes):
        env.reset_env()
        for step in range(episode_horizont):
            print(f"-------Episode:{episode + 1} Step:{step + 1}---------")
            state, _ = env.state_space_funct()
            action = env.generate_sample_act()
            env.env_step(action)  # take the action
            next_state, image_state = env.state_space_funct()
            reward, done = env.calculate_reward()
            env.env_render(image_state, done, step)
        print(f"******* -----Episode {episode+1} Ended----- ********* ")
    env.close_env()  # close the usb port and disable torque


if __name__ == '__main__':
    main_run()






