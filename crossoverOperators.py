import numpy as np
from copy import deepcopy

#@profile
def singlePointCrossover(indiv1, indiv2, numNodes):
	crossedIndividual1 = deepcopy(indiv1)
	crossedIndividual2 = deepcopy(indiv2)
	cut = np.random.randint(1, numNodes-1)
	for i in range(cut,numNodes):
		crossedIndividual1.vertexColors[i], crossedIndividual2.vertexColors[i] = indiv2.vertexColors[i], indiv1.vertexColors[i]
	return crossedIndividual1, crossedIndividual2
