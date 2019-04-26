import numpy as np
import torch

def getIndexedArrayFromTrajectory(obs):
	#states that are not real states
	badStates=[2,3,4,5,6,7]
	
	trajTensor = np.array(obs)
	
	#get dimensions of the tensor of trajectory
	shapeParams = trajTensor.shape
	steps = shapeParams[0]
	x_values = shapeParams[2]
	y_values = shapeParams[3]
	dir_values = shapeParams[4]

	state_indexer = 0

	#The two dictionaries
	stateToIndex = {}
	indexToState = {}
	
	#to get state to tensor mappings
	one_observation = trajTensor[0]
	one_observation=one_observation[0]

	for x in range(0,x_values-1):
		for y in range(0,y_values-1):
			state_code = one_observation[x][y][0]
			if state_code not in badStates:
				stateToIndex[(x,y,0)] = state_indexer
				indexToState[state_indexer] = (x,y,0)
				state_indexer += 1
				stateToIndex[(x,y,1)] = state_indexer
				indexToState[state_indexer] = (x,y,1)
				state_indexer += 1
				stateToIndex[(x,y,2)] = state_indexer
				indexToState[state_indexer] = (x,y,2)
				state_indexer += 1
				stateToIndex[(x,y,3)] = state_indexer
				indexToState[state_indexer] = (x,y,3)
				state_indexer += 1

	            
	return stateToIndex, indexToState
	            
def getStateIndexTraj(obs,stateToIndex,indexToState):

	trajTensor = np.array(obs)
	
	shapeParams = trajTensor.shape
	steps = shapeParams[0]
	x_values = shapeParams[2]
	y_values = shapeParams[3]
	dir_values = shapeParams[4]

	state_sequence=[]
	
	for i in range(0,steps-1):
		one_observation = trajTensor[i]
		one_observation=one_observation[0]
		for x in range(0,x_values-1):
			for y in range(0,y_values-1):
				state_code = one_observation[x][y][0]
				if state_code == 10:
					dir = one_observation[x][y][1]
					x_agent = x
					y_agent = y
					dir_agent = dir
	
		pos = (x_agent,y_agent,dir_agent)
		stateIndex = stateToIndex[pos]
		state_sequence.append(stateIndex)
	return state_sequence