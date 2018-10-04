import fileinput
import sys

def populationSize 500

def fitness(graph, colors, nodes):
	# for c in range(colors):
		# for j in range(graph[j]):
			# connected = graph[c][j]
			# if(connected):
				# if(colors[c] == colors[graph[c][j]]):
					# print("VIOLACAO");
	score = 0;
					
	for i in range(nodes):
		for j in range(nodes):
			if(graph[i][j] == 1):
				if(colors[i] == colors[j]):
					print("Warning: {0} ({1}) to {2} ({3})".format(i, colors[i], j, colors[j]));
					score -= 1;
		
	return score

nodes = 0
edges = 0
inputEdges = list()
population = list()

for line in fileinput.input('simple.col'):
	if(line[0] == 'p'):
		params = line.split()
		nodes = int(params[2]);
		edges = int(params[3]);	
	elif (line[0] == 'e'):
		params = line.split()
		fromVertex = int(params[1]);
		toVertex = int(params[2]);
		inputEdges.append([fromVertex, toVertex])


graph = [0] * nodes;
for j in range(nodes):
	graph[j] = [0] * nodes;

for x in inputEdges:
	j = x[0] - 1;
	i = x[1] - 1;
	graph[j][i] = 1 
	graph[i][j] = 1
	#print("{0} - {1}", x[0], x[1]);
	
print("nodes: {0}, edges: {1}".format(nodes, edges));

for j in range(nodes):
	for i in range(nodes):
		#print("{0}".format(graph[j][i]));
		sys.stdout.write(str(graph[j][i]) + " ")
	print()
	

# vertex color representation
graphColors = [0] * nodes;
#for i in range(populationSize)
	#graphColors = np.random.randint(1, nodes, size=nodes) #Create random coloration for the graph
	#population.append(graphColors)
print(graphColors);

score = fitness(graph, graphColors, nodes)

print(score);
