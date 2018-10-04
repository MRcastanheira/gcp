import fileinput
import sys
import numpy as np

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
	def __init__(self, graph):
		self.graph = graph
		self.numNodes = len(graph[0])
		self.vertexColors = np.random.randint(1, self.numNodes, size=self.numNodes)

	def fitness(self):
		score = 0
		for i in range(self.numNodes):
			for j in range(self.numNodes):
				if(self.graph[i][j] == 1):
					if(self.vertexColors[i] == self.vertexColors[j]):
						print("Warning: {0} ({1}) to {2} ({3})".format(i, self.vertexColors[i], j, self.vertexColors[j]))
						score -= 1
		return score

class Population:
	def __init__(self, graph, size):
		self.numNodes = len(graph[0])
		self.graph = graph
		self.size = size
		self.population = self.initialize()

		for i, individual in enumerate(self.population):
			print("Node {0} colors: {1}".format(i, individual.vertexColors))
			individual.fitness()

	def initialize(self):
		population = []
		for i in range(self.size):
			individual = Individual(self.graph)
			population.append(individual)
		return population

	def beautifulGraph(self):
		for i in range(self.numNodes):
			if(i == 0):
				sys.stdout.write("  ")
				for x in range(self.numNodes):
					sys.stdout.write(chr(x + 65) + " ")
				print()
			for j in range(self.numNodes):
				if(j == 0):
					sys.stdout.write(chr(i + 65) + " ")
				sys.stdout.write(str(graph[i][j]) + " ")
			print()

graph = readFileInstance('simple.col')
population = Population(graph, 1)
population.beautifulGraph()
