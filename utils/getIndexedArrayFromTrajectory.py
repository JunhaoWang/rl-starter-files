import numpy as np
import torch

def getIndexedArrayFromTrajectory(obs):
	#states that are not real states
	badStates=[2,3,4,5,6,7]
	
	trajTensor = torch.transpose(torch.transpose(obs, 1, 3), 2, 3)
	trajTensor = np.array(trajTensor)
	
	#get dimensions of the tensor of trajectory
	shapeParams = trajTensor.shape
	print(shapeParams)
	steps = shapeParams[0]
	x_values = shapeParams[2]
	y_values = shapeParams[3]
	dir_values = shapeParams[1]

	state_indexer = 0
	
	#The two dictionaries
	stateToIndex = {}
	indexToState = {}
	
	#to get state to tensor mappings
	one_observation = trajTensor[0]

	for x in range(0,x_values-1):
		for y in range(0,y_values-1):
			state_code = one_observation[0][x][y]
			print(state_code)
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

	trajTensor = torch.transpose(torch.transpose(obs, 1, 3), 2, 3)
	trajTensor = np.array(trajTensor)
	
	shapeParams = trajTensor.shape
	steps = shapeParams[0]
	x_values = shapeParams[2]
	y_values = shapeParams[3]
	dir_values = shapeParams[1]

	state_sequence=[]
	
	for i in range(0,steps-1):
		one_observation = trajTensor[i]
		for x in range(0,x_values-1):
			for y in range(0,y_values-1):
				state_code = one_observation[0][x][y]
				if state_code == 10:
					dir = one_observation[1][x][y]
					x_agent = x
					y_agent = y
					dir_agent = dir
	
		pos = (x_agent,y_agent,dir_agent)
		stateIndex = stateToIndex[pos]
		state_sequence.append(stateIndex)
	return state_sequence