# miscellaneous util functions

import torch

def grid_seq_2_idx_seq(grid_seq):
    """
    Convert full grid sequence (temporal order is asumed to be preserved) to discrete state sequence

    :param grid_seq: torch.Tensor of size (num_frame x width x height x num_channel)
    :return:
        idx_seq: Python list of integers of size num_frame denoting flattened index of agent
        location (encoded by 10)

        idx_max: Python integer denoting the size of discrete state space (equal to width x height)

        grid_shape: Python list containing [width, height] of grid
    """
    grid_shape = list(grid_seq.shape[1:3])
    flat_grid_seq = grid_seq[:, :, :, 0].reshape(grid_seq.shape[0], -1)
    idx_max = flat_grid_seq.shape[-1]
    idx_seq = torch.argmax(flat_grid_seq, dim=-1).tolist()
    return idx_seq, idx_max, grid_shape

# def getIndexedArrayFromTrajectory(obs):

    #states that are not real states
    # badStates=[0,1,2,5,6,7]

    # trajTensor = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)
    # trajTensor = trajTensor.cpu()
    # trajTensor = np.array(trajTensor)
    #
    # #get dimensions of the tensor of trajectory
    # shapeParams = trajTensor.shape
    # steps = shapeParams[0]
    # x_values = shapeParams[1]
    # y_values = shapeParams[2]
    #
    # state_indexer = 0
    #
    # #The two dictionaries
    # stateToIndex = {}
    # indexToState = {}
    #
    # #to get state to tensor mappings
    # one_observation = trajTensor[0]
    # for x in range(0,x_values):
    #     for y in range(0,y_values):
    #         one_state = one_observation[x][y]
    #         state_code = one_state[0]
    #         if state_code not in badStates:
    #             stateToIndex[(x,y)] = state_indexer
    #             indexToState[state_indexer] = (x,y)
    #             state_indexer += 1
    #
    # state_sequence=[]
    #
    # for i in range(0,steps):
    #     one_observation = trajTensor[i]
    #
    #     #find where the agent is
    #     for x in range(0,x_values):
    #         for y in range(0,y_values):
    #             one_state = one_observation[x][y]
    #             state_code = one_state[0]
    #             if state_code == 10:
    #                 x_agent = x
    #                 y_agent = y
    #                 pos = (x_agent,y_agent)
    #                 stateIndex = stateToIndex[pos]
    #                 state_sequence = state_sequence.append(stateIndex)

    # return state_sequence, stateToIndex, indexToState
