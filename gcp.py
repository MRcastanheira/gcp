import fileinput
import sys
import numpy as np
import time
from copy import deepcopy

DEBUG = 0

def readFileInstance(file):
	nodes = 0
	edges = 0
	inputEdges = []
	population = []

	for line in fileinput.input(file):
		if(line[0] == 'p'):
			params = line.split()
			nodes = int(params[2])
			edges = int(params[3])
		elif (line[0] == 'e'):
			params = line.split()
			fromVertex = int(params[1])
			toVertex = int(params[2])
			inputEdges.append([fromVertex, toVertex])

	graph = [0] * nodes
	for j in range(nodes):
		graph[j] = [0] * nodes

	for x in inputEdges:
		j = x[0] - 1
		i = x[1] - 1
		graph[j][i] = 1
		graph[i][j] = 1

	return graph

class Individual:
	global graph
	global edgeList
	global numNodes
	global numEdges

	def __init__(self, mutationRate):
		self.vertexColors = np.random.randint(1, numNodes+1, size=numNodes) #creates an assortment of random colors
		self.mutationRate = mutationRate

	def fitness(self):
		score = 0
		edgeViolationScore = numEdges
		for edge in edgeList:
			if(self.vertexColors[edge[0]] == self.vertexColors[edge[1]]):
				#print("Warning: {0} ({1}) to {2} ({3})".format(i, self.vertexColors[i], j, self.vertexColors[j]))
				edgeViolationScore -= 1

		normalizedColors = numNodes - self.validColors() + 1
		vertexColoringScore = normalizedColors * 10

		if(self.isValidSolution()):
			score += edgeViolationScore + vertexColoringScore
		else:
			score += edgeViolationScore + (vertexColoringScore / 2)
		return score

	def validColors(self):
		return len(np.unique(self.vertexColors))

	def mutate(self):
		r = np.random.random()
		if(self.mutationRate > r):
			position = np.random.randint(0, numNodes)
			self.vertexColors[position] = np.random.randint(1, numNodes+1)

	def isValidSolution(self):
		for edge in edgeList:
			if(self.vertexColors[edge[0]] == self.vertexColors[edge[1]]):
				return False
		return True

	def __str__(self):
		return self.visual()

	def visual(self):
		return str(self.vertexColors) + " (" + str(self.fitness()) + ")"

class Population:
	global graph
	global numNodes
	global numEdges

	def __init__(self, size):
		numNodes = len(graph[0])
		self.size = size
		self.population = self.initialize()

		for i, individual in enumerate(self.population):
			print("Node {0} colors: {1}".format(i, individual.vertexColors))
			individual.fitness()

	def __str__(self):
		pop = ""
		for i in self.population:
			pop += i.visual() + "\n"
		return pop

	def initialize(self):
		population = []
		for i in range(self.size):
			individual = Individual(0.2) # 0.2 = mutation rate
			population.append(individual)
		return population

	def crossover(self, indiv1, indiv2):
		crossedIndividual1 = deepcopy(indiv1)
		crossedIndividual2 = deepcopy(indiv2)
		cut = np.random.randint(1, numNodes-1)
		for i in range(cut,numNodes):
			crossedIndividual1.vertexColors[i], crossedIndividual2.vertexColors[i] = indiv2.vertexColors[i], indiv1.vertexColors[i]
		return crossedIndividual1, crossedIndividual2

	def beautifulGraph(self):
		for i in range(numNodes):
			if(i == 0):
				sys.stdout.write("  ")
				for x in range(numNodes):
					sys.stdout.write(chr(x + 65) + " ")
				print()
			for j in range(numNodes):
				if(j == 0):
					sys.stdout.write(chr(i + 65) + " ")
				sys.stdout.write(str(graph[i][j]) + " ")
			print()

	def nextGen(self):
		totalScore = 0
		scores = [0] * self.size
		accumulated = [0] * self.size

		for i in range(self.size):
			scores[i] = self.population[i].fitness()

		# Sort score population pairs list based on the score
		scores, sortedPopulation = list(zip(*sorted(zip(scores, self.population),
		 	key=lambda x: x[0])))
			
#===================================== PRINTS =====================================
		print("-------- Best so far -------")
		#print("Colors: {0}".format(sortedPopulation[self.size-1].vertexColors))
		print("Number of colors: {0}".format(sortedPopulation[self.size-1].validColors()))
		print("Is valid solution: {0}".format("yes" if
			sortedPopulation[self.size-1].isValidSolution() else "no"))
		print("Best = {0}".format(scores[self.size-1]))
#==================================================================================

		# compute cumulative score
		for i in range(self.size):
			totalScore += scores[i]
			accumulated[i] = totalScore

		probs = [0] * self.size
		for i in range(self.size):
			probs[i] = (scores[i] / totalScore) * 100

#===================================== DEBUG ======================================
		if DEBUG == 1:
			print("Input population (sorted):")
			for i in range(self.size):
				validity = "valid" if sortedPopulation[i].isValidSolution() else "invalid"
				print(str(sortedPopulation[i].vertexColors) +
				" (" + str(sortedPopulation[i].validColors()) + ")" +
				" " + str(validity) +
				" - " + str(scores[i]) + " / " + str(totalScore) +
				" (" + str(round(probs[i], 2)) + "%)")
#==================================================================================

		# generate a new population
		newPopulation = []
		# newPopulation.append(sortedPopulation[-1])
		# newPopulation.append(sortedPopulation[-2])
		while(len(newPopulation) < self.size-2):
			# first random individual
			firstRandomRange = np.random.randint(0, totalScore+1)
			firstIndex = 0
			while(accumulated[firstIndex] < firstRandomRange):
				firstIndex += 1

			firstIndividual = sortedPopulation[firstIndex]

			# second random individual
			secondRandomRange = np.random.randint(0, totalScore+1)
			secondIndex = 0
			while(accumulated[secondIndex] < secondRandomRange):
				secondIndex += 1

			secondIndividual = sortedPopulation[secondIndex]

			# do crossover
			firstCrossed, secondCrossed = self.crossover(firstIndividual, secondIndividual)

			# add to new population
			newPopulation.append(firstCrossed)
			newPopulation.append(secondCrossed)

		if DEBUG == 1:
			print("Crossover population:")
			for i in range(self.size-2):
				print(newPopulation[i])

		# do mutation
		for i in range(self.size-2):
			newPopulation[i].mutate()
		
		newPopulation.append(sortedPopulation[-1])
		newPopulation.append(sortedPopulation[-2])

		self.population = newPopulation
		
#===================================== DEBUG ======================================
		if DEBUG == 1:
			print("Crossover + Mutated population: \n{0}".format(self))
			print("----------------------------")
#==================================================================================

graph = readFileInstance('flat1000_76_0.col') # flat1000_76_0 simple complicated
numNodes = len(graph[0])
numEdges = 0
for i in range(numNodes):
	for j in range(i+1, numNodes):
		if(graph[i][j] == 1):
			numEdges += 1

edgeList = []
for i in range(numNodes):
	for j in range(i+1, numNodes):
		if(graph[i][j] == 1):
			edgeList.append([i,j])

population = Population(50)
for i in range(100):
	print("Generation {0}:".format(i))
	population.nextGen()
	time.sleep(0)
