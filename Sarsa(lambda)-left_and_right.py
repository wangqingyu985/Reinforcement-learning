"""
A simple example for Reinforcement Learning using table lookup Sarsa method.
An agent "o" is on the left of a 1 dimensional world, the treasure is on the rightmost location.
Run this program and to see how the agent will improve its strategy of finding the treasure.

View more on my tutorial page: https://www,github.com/wangqingyu985
"""

# Author: Qingyu WANG
# Contact: 120153710@qq.com
# Date: 26 Feb. 2021

import numpy as np
import pandas as pd
import time

from pandas import DataFrame

np.random.seed(2)  # reproducible

N_STATES = 6  # the length of the 1 dimensional world
ACTIONS = ['left', 'right']  # available actions
EPSILON = 0.9  # greedy police
ALPHA = 0.1  # learning rate
GAMMA = 0.9  # discount factor
LAMBDA = 1  # decay rate
MAX_EPISODES = 13  # maximum episodes
FRESH_TIME = 0.3  # fresh time for one move


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),  # q_table initial values
        columns=actions,  # actions's name
    )
    return table


def build_e_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),  # e_table initial values
        columns=actions,  # actions's name
    )
    return table


def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):  # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)
    else:  # act greedy
        action_name = state_actions.idxmax()  # replace argmax to idxmax as argmax means a different function in newer version of pandas
    return action_name


def get_env_feedback(S, A):
    # This is how agent will interact with the environment
    if A == 'right':  # move right
        if S == N_STATES - 2:  # terminate
            S_ = 'terminal'
            R = 1
        else:  # on the way
            S_ = S + 1
            R = 0
    else:  # move left
        if S == 0:
            S_ = S  # reach the wall
            R = -1
        else:  # on the way
            S_ = S - 1
            R = 0
    return S_, R


def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-'] * (N_STATES - 1) + ['T']  # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    # main part of RL loop
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        e_table = build_e_table(N_STATES, ACTIONS)
        step_counter = 0
        S = 0  # Initialize S
        is_terminated = False
        update_env(S, episode, step_counter)
        A = choose_action(S, q_table)  # Initialize A
        while not is_terminated:

            S_, R = get_env_feedback(S, A)  # take action A and observe S_, R

            q_predict = q_table.loc[S, A]
            if S_ != 'terminal':
                # choose A_ from S_ using policy derived from Q(epsilon-greedy)
                A_ = choose_action(S_, q_table)
                q_target = R + GAMMA * q_table.loc[S_, A_]
                delta = q_target - q_predict
            else:
                q_target = R  # next state is terminal
                is_terminated = True  # terminate this episode

            e_table.loc[S, A] += 1
            for s in range(N_STATES):
                for a in list(ACTIONS):
                    q_table.loc[s, a] += ALPHA * delta * e_table.loc[s, a]
                    e_table.loc[s, a] = GAMMA * LAMBDA * e_table.loc[s, a]

            S = S_  # move to next state
            A = A_  # move to next action

            update_env(S, episode, step_counter + 1)
            step_counter += 1
        rubbish = 0
    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
