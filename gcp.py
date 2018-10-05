import fileinput
import sys
import numpy as np
import time

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
	def __init__(self, graph, mutationRate):
		self.graph = graph
		self.numNodes = len(graph[0])
		self.vertexColors = np.random.randint(1, self.numNodes, size=self.numNodes)
		self.mutationRate = mutationRate

	def fitness(self):
		score = 0
		for i in range(self.numNodes):
			for j in range(self.numNodes):
				if(self.graph[i][j] == 1):
					if(self.vertexColors[i] == self.vertexColors[j]):
						#print("Warning: {0} ({1}) to {2} ({3})".format(i, self.vertexColors[i], j, self.vertexColors[j]))
						score -= 1
		return score
		
	def mutation(self):
		r = np.random.random()
		if(self.mutationRate > r):
			position = np.random.randint(0, self.numNodes-1)
			#print("New vertexColors: {0}".format(self.vertexColors))
			#print("Vertex color at {0} before = {1}".format(position+1, self.vertexColors[position]))
			self.vertexColors[position] = np.random.randint(1, self.numNodes)
			#print("New vertexColors: {0}".format(self.vertexColors))
		

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
			individual = Individual(self.graph, 0.5) # 0.5 = mutation rate
			population.append(individual)
		return population
		
	def crossover(self, indiv1, indiv2):
		cut = np.random.randint(1, self.numNodes-1)
		#print("Candidates before crossover: \n {0} \n {1}".format(vertexColors1, vertexColors2))
		for i in range(cut,self.numNodes):
			indiv1.vertexColors[i], indiv2.vertexColors[i] = indiv2.vertexColors[i], indiv1.vertexColors[i]
		#print("Candidates after crossover: \n {0} \n {1}".format(vertexColors1, vertexColors2))
		return indiv1,indiv2

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
			
	def nextGen(self):	
		for i in range(self.size):
			print(self.population[i].vertexColors)
		totalScore = 0
		maxScore = self.numNodes * self.numNodes
		scores = [0] * self.size
		accumulated = [0] * self.size
		for i in range(self.size):
			scores[i] = maxScore + self.population[i].fitness()
			
		scores, self.population = (list(x) for x in zip(*sorted(zip(scores, self.population)))) # This sorts the population list based on the order that scores got sorted
		print("Best = {0}".format(scores[self.size-1]))
		
		for i in range(self.size):
			totalScore += scores[i]
			accumulated[i] = totalScore
		
		newPopulation = []
		while(len(newPopulation) < self.size):
			j = 0
			r = np.random.randint(0, totalScore)	
			while(accumulated[j] < r):
				j += 1
			k = 0
			r = np.random.randint(0, totalScore)	
			while(accumulated[k] < r):
				k += 1		
			first = self.population[j]	
			second = self.population[k]
			crossed1, crossed2 = self.crossover(first,second)
			newPopulation.append(crossed1)
			newPopulation.append(crossed2)
		self.population = newPopulation
			
		
		

graph = readFileInstance('flat1000_76_0.col')
population = Population(graph, 6)
for i in range(100):
	print("Generation {0}:".format(i))
	population.nextGen()
	time.sleep(3)
#population.crossover(population.population[0].vertexColors,population.population[1].vertexColors)
# population.beautifulGraph()

# test = Individual(graph,1)
# test.mutation()
