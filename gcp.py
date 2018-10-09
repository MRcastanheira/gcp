import fileinput
import sys
import numpy as np
import time
import math
import random
from copy import deepcopy

import crossoverOperators

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
	edgeList = []
	for j in range(nodes):
		graph[j] = [0] * nodes

	for x in inputEdges:
		j = x[0] - 1
		i = x[1] - 1
		graph[j][i] = 1
		graph[i][j] = 1
		edgeList.append([i,j])

	return graph, edgeList, nodes, edges

class Individual:
	global graph
	global edgeList
	global numNodes
	global numEdges

	def __init__(self, mutationRate):
		self.vertexColors = np.random.randint(1, numNodes+1, size=numNodes) #creates an assortment of random colors
		self.mutationRate = mutationRate

	#@profile
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
		#print(self.vertexColors)
		for position in range(numNodes):
			r = np.random.random()
			if(self.mutationRate > r):
				self.vertexColors[position] = np.random.randint(1, numNodes+1)
		# print(self.vertexColors)
		# print("-------------")
		# if(self.mutationRate > r):
			# for i in range(np.random.randint(0, numNodes+1)):
				# position = np.random.randint(0, numNodes)
				# self.vertexColors[position] = np.random.randint(1, numNodes+1)

	#@profile
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

	def __init__(self, size, mutationRate, crossoverRate, crossoverMethod):
		numNodes = len(graph[0])
		self.size = size
		self.crossoverRate = crossoverRate
		self.population = self.initialize(mutationRate)
		self.crossover = crossoverMethod

	def __str__(self):
		pop = ""
		for i in self.population:
			pop += i.visual() + "\n"
		return pop

	def initialize(self, mutationRate):
		population = []
		for i in range(self.size):
			individual = Individual(mutationRate)
			population.append(individual)
		return population

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
		best = deepcopy(sortedPopulation[-1])

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

#===================================== DEBUG ======================================
		if DEBUG == 1:
			probs = [0] * self.size
			for i in range(self.size):
				probs[i] = (scores[i] / totalScore) * 100
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
		numberOfElites = 1
		numPointers = self.size - numberOfElites

		pointerDistance = math.floor(totalScore / numPointers)

		start = np.random.randint(0, pointerDistance)

		pointers = np.arange(start, totalScore, pointerDistance)

		mating = []

		pointerCount = 0
		i = 0
		while(i < self.size and pointerCount < numPointers):
			if accumulated[i] > pointers[pointerCount]:
				mating.append(i)
				pointerCount += 1
				i -= 1
			i += 1

			if i < 0:
				i = 0

		random.shuffle(mating)

		newPopulation = []
		for i in range(0, numPointers, 2):
			firstIndividual = sortedPopulation[mating[i]]
			if i+1 >= numPointers:
				firstIndividual = sortedPopulation[mating[i]]
			else:
				secondIndividual = sortedPopulation[mating[i+1]]

			r = np.random.random()
			if self.crossoverRate > r:
				firstCrossed, secondCrossed = self.crossover(firstIndividual, secondIndividual, numNodes)
			else:
				firstCrossed = deepcopy(firstIndividual)
				secondCrossed = deepcopy(secondIndividual)

			# add to new population
			newPopulation.append(firstCrossed)
			if(len(newPopulation) < self.size-1):
				newPopulation.append(secondCrossed)

#===================================== DEBUG ======================================
		if DEBUG == 1:
			print("Crossover population:")
			for i in range(self.size-1):
				print(newPopulation[i])
#==================================================================================

		# do mutation
		for i in range(len(newPopulation)):
			newPopulation[i].mutate()

		newPopulation.append(best)

		self.population = newPopulation

#===================================== DEBUG ======================================
		if DEBUG == 1:
			print("Crossover + Mutated population: \n{0}".format(self))
			print("----------------------------")
#==================================================================================

graph, edgeList, numNodes, numEdges = readFileInstance('complicated.col') # flat1000_76_0 simple complicated

populationSize = 20
generations = 100
mutationRate = 0.15
crossoverRate = 0.7

population = Population(populationSize, mutationRate, crossoverRate, crossoverOperators.crossover)
for i in range(1, generations+1):
	print("Generation {0}:".format(i))
	population.nextGen()
