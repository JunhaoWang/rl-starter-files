#Script to get the array containing the state-indexed trajectories.

import torch
import numpy as np

def getIndexedArrayFromTrajectory(obs):
    #states that are not real states
    badStates=[2,5,6]

    trajTensor = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)
    trajTensor = np.array(trajTensor)

    #get dimensions of the tensor of trajectory
    shapeParams = trajTensor.shape
    steps = shapeParams[0]
    x_values = shapeParams[1]
    y_values = shapeParams[2]

    state_indexer = 0

    #The two dictionaries
    stateToIndex = {}
    indexToState = {}

    #to get state to tensor mappings
    one_observation = trajTensor[0]
    for x in range(0,x_values):
        for y in range(0,y_values):
            one_state = one_observation[x][y]
            state_code = one_state[0]
            if state_code not in badStates:
                stateToIndex[(x,y)] = state_indexer
                indexToState[state_indexer] = (x,y)
                state_indexer += 1

    state_sequence=[]

    for i in range(0,steps):
        one_observation = trajTensor[i]

        #find where the agent is
        for x in range(0,x_values):
            for y in range(0,y_values):
                one_state = one_observation[x][y]
                state_code = one_state[0]
                if state_code == 10:
                    x_agent = x
                    y_agent = y

        pos = (x_agent,y_agent)
        stateIndex = stateToIndex[pos]
        state_sequence = state_sequence.append(stateIndex)

    return state_sequence, stateToIndex, indexToState




