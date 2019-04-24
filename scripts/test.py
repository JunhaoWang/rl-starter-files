import pickle
from utils.misc import getSSRep
from utils.aggregators import aggregateAverage, aggregateVAE
from utils.getIndexedArrayFromTrajectory import getIndexedArrayFromTrajectory, getStateIndexTraj

fileObject = open("scripts/optimal_trajs_MiniGrid-Empty-8x8-v0.pkl",'rb')
optimal_trajs = pickle.load(fileObject)

print(optimal_trajs)
stateToIndex, indexToState = getIndexedArrayFromTrajectory(optimal_trajs[0])

print(indexToState)

stateOccupancyList = []

for i in range(len(optimal_trajs)):
    indexedTraj = getIndexedArrayFromTrajectory(optimal_trajs[i],stateToIndex, indexToState)
    print(indexedTraj)
    stateOccupancyList.append(indexedTraj)

stateOccupancyList = getSSRepHelperMeta(stateOccupancyList,len(stateToIndex),aggregateAverage,method='every')

print(stateOccupancyList)