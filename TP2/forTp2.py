#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 11:39:44 2020

@author: berar
"""
import numpy as np


GRID_SIZE = 4
TERMINAL_STATES = [0, GRID_SIZE*GRID_SIZE-1]
states = np.arange(GRID_SIZE*GRID_SIZE)
actions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
discount = 1.

def next_state(grid_size, state, action):
    i,j = np.unravel_index(state, (grid_size, grid_size))
    if action == 'UP':
        i = np.maximum(0,i-1)
    elif action == 'DOWN':
        i = np.minimum(i+1,grid_size-1)
    elif action == 'RIGHT':
        j = np.minimum(j+1,grid_size-1)
    elif action == 'LEFT':
        j = np.maximum(0,j-1)    
    new_state = np.ravel_multi_index((i,j), (grid_size, grid_size))
    return new_state
    
def is_done(state, terminal_states):
    return state in terminal_states

# The unifom policy            
uniform_policy = {s : { a : 1/len(actions) for a in actions } for s in states}


# Transition is coded as a dictionary of dictionary
P = {}
for s in range(len(states)):
    P[s] = {a : () for a in actions}
    if s in TERMINAL_STATES:
        # if terminal state, stay where you are
        # instead of next_state
        reward = 0.
        for action in actions:
            P[s][action] = (s, reward, True)
    else:
        # transition
        reward = -1.
        for action in actions:
            next_s = next_state(GRID_SIZE, s, action)
            P[s][action] = (next_s,reward,is_done(next_s, TERMINAL_STATES))
            
            
def policy_evaluation_1(policy, P, discount=1.0, theta=1e-6, max_iterations=300):
    
    state_len = len(P.keys())
    actions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
    action_len = len(actions)
    states = np.arange(GRID_SIZE*GRID_SIZE)

    # Number of evaluation iterations
    evaluation_iterations = 1
    # Initialize a value function for each state as zero
    V = np.zeros(state_len)
    # Repeat until change in value is below the threshold
    for i in range(int(max_iterations)):
        # Initialize a change of value function as zero
        delta = 0
        # Iterate though each state
        for state in range(state_len):
            # Initial a new value of current state
            v = 0
            # Try all possible actions which can be taken from this state
            for action_state, action in enumerate(policy[state]):
                action_probability = policy[state][action]
                # Check how good next state will be
                #print(P[state][action])
                #for next_state, reward, terminated in P[state][action]:
                    # Calculate the expected value
                for action in actions:
                    next_state, reward, done = P[state][action]
                    v += uniform_policy[s][action] * action_probability * (reward + discount * V[next_state])

            # Calculate the absolute change of value function
            delta = max(delta, np.abs(V[state] - v))
            # Update value function
            V[state] = v
            
        evaluation_iterations += 1

        # Terminate if value change is insignificant
        if delta < theta:
            print(f'Policy evaluated in {evaluation_iterations} iterations.')
            return V

def one_step_lookahead(P, state, V, discount_factor):
        action_values = np.zeros(4)
        probability = 0.25
        actions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        uniform_policy = {s : { a : 1/len(actions) for a in actions } for s in states}
        
        for s in range(len(P.keys())):
            # we will get new state value
            new_s = 0
            # for all actions
            for a in range(len(actions)):
                # for all transitions from currect state
                for action in actions:
                    #print("i->", i)
                    #print("uniform_policy ->", uniform_policy)
                    next_state, reward, done = P[s][action]
                    print(action_values[a])
                    action_values[a] += uniform_policy[s][action] * (reward + discount_factor * V[next_state])
        return action_values

def policy_iteration(P, discount_factor=1.0, max_iterations=1e9):
        
        state_len = len(P.keys())
        action_len = 4 # up right down left
        states = np.arange(GRID_SIZE*GRID_SIZE)
        actions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        transition_prob = 1/4
        
        # Start with a random policy
        #num states x num actions / num actions
        #policy = np.ones([state_len, action_len]) / action_len
        policy = {s : { a : 1/len(actions) for a in actions } for s in states}
        #print(policy)
        # Initialize counter of evaluated policies
        evaluated_policies = 1
        # Repeat until convergence or critical number of iterations reached
        for i in range(int(max_iterations)):
                stable_policy = True
                # Evaluate current policy
                #V = policy_evaluation(policy, environment, discount_factor=discount_factor)
                V = policy_evaluation(P, state_values, GRID_SIZE)
                # Go through each state and try to improve actions that were taken (policy Improvement)
                for state in range(state_len):
                        # Choose the best action in a current state under current policy
                        current_action = np.argmax(policy[state])
                        # Look one step ahead and evaluate if current action is optimal
                        # We will try every possible action in a current state
                        action_value = one_step_lookahead(P, state, V, discount_factor)
                        # Select a better action
                        best_action = np.argmax(action_value)
                        # If action didn't change
                        if current_action != best_action:
                                stable_policy = True
                                # Greedy policy update
                                policy[state] = np.eye(action_len)[best_action]
                evaluated_policies += 1
                # If the algorithm converged and policy is not changing anymore, then return final policy and value function
                if stable_policy:
                        print(f'Evaluated {evaluated_policies} policies.')
                        return policy, V

