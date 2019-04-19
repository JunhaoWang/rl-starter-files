# miscellaneous util functions

import torch
import torch
import numpy as np
from collections import Counter


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
    idx_seq = torch.argmax(flat_grid_seq, dim=-1)
    row_idx = torch.div(idx_seq, grid_shape[1])
    col_idx = torch.remainder(idx_seq,grid_shape[1])
    coord_idx = torch.transpose(torch.cat((row_idx.unsqueeze(0),col_idx.unsqueeze(0)),0),0,1)

    return coord_idx


def getSSRepHelperEveryVisit(vectorTraj, numState):
   """
   Every visit estimate of stationary state distribution
     :param vectorTraj: list representing a length-T state trajectory containing integers from [0 .. |States|-1]
     :param numState: integer denoting |States|, the cardinality of set of unqiue discrete states
     :return: |States|-length list representing the stationary distribution of states from available N trajectories
   """

   collect_s = []
   for s in range(numState):
       if s in vectorTraj:
           start_idx_set = [i for i, x in enumerate(vectorTraj) if x == s]
           for j in start_idx_set:
               collect_s += vectorTraj[j:]
   collect_s = dict(Counter(collect_s))
   result = []
   for s in range(numState):
       if s not in collect_s:
           result.append(0)
       else:
           result.append(collect_s[s])
   return (np.array(result) / np.sum(result)).tolist()


def getSSRepHelperFirstVisit(vectorTraj, numState):
    """
    First visit estimate of stationary state distribution
     :param vectorTraj: list representing a length-T state trajectory containing integers from [0 .. |States|-1]
     :param numState: integer denoting |States|, the cardinality of set of unqiue discrete states
     :return: |States|-length list representing the stationary distribution of states from available N trajectories
    """

    collect_s = []
    for s in range(numState):
        if s in vectorTraj:
            collect_s += vectorTraj[vectorTraj.index(s):]
    collect_s = dict(Counter(collect_s))
    result = []
    for s in range(numState):
        if s not in collect_s:
            result.append(0)
        else:
            result.append(collect_s[s])
    return (np.array(result) / np.sum(result)).tolist()

def getSSRepHelperVanilla(matrixTraj, numState):
    collect_s = [item for sublist in matrixTraj for item in sublist]
    collect_s_count = Counter(collect_s)
    result = []
    for s in range(numState):
        if s not in collect_s:
            result.append(0)
        else:
            result.append(collect_s[s])
    return (np.array(result) / np.sum(result)).tolist()

def getSSRepHelperMeta(matrixTraj, numState, aggregator, method='vanilla'):
    """
    Estimate of stationary state distribution using a set of trajectories and specified aggregator
     :param matrixTraj: N x T numpy array representing N number of length-T state trajectories, with each state denoted by
        integer from [0 .. |States|-1], where |States| denotes the cardinality of set of discrete states
     :param numState: integer denoting |States|, the cardinality of set of unqiue discrete states
     :param aggregator: lambda function that takes in list of list of float and return |States|-length array
     :param method: 'first' or 'every' or 'vanilla'
     :return: |States|-length array representing the stationary distribution of states from available N trajectories
    """
    if method == 'first':
        collect_SSRep = list(map(lambda x: getSSRepHelperFirstVisit(x, numState), matrixTraj))
        collect_SSRep = np.array(collect_SSRep)
        return aggregator(collect_SSRep)
    elif method == 'every':
        collect_SSRep = list(map(lambda x: getSSRepHelperEveryVisit(x, numState), matrixTraj))
        collect_SSRep = np.array(collect_SSRep)
        return aggregator(collect_SSRep)
    elif method == 'vanilla':
        collect_SSRep = getSSRepHelperVanilla(matrixTraj, numState)
        return np.array(collect_SSRep)
    else:
        raise Exception('not implemented')

# This is wrong, need to be replaced with David code
def getSSRep(traj_info_set, start_idx, end_idx, aggregator, method='vanilla'):
    subset_traj_info = traj_info_set[start_idx:end_idx]
    matrixTraj = list(map(lambda x: x[0], subset_traj_info))
    numState = subset_traj_info[0][1]
    return getSSRepHelperMeta(matrixTraj, numState, aggregator, method)

