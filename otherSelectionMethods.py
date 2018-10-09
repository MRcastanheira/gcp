#newPopulation = []
#while(len(newPopulation) < self.size-1):
#========================= Roulette Wheel Selection ===============================
  # # first random individual
  # firstRandomRange = np.random.randint(0, totalScore+1)
  # firstIndex = 0
  # while(accumulated[firstIndex] < firstRandomRange):
    # firstIndex += 1

  # firstIndividual = sortedPopulation[firstIndex]

  # # second random individual
  # secondRandomRange = np.random.randint(0, totalScore+1)
  # secondIndex = 0
  # while(accumulated[secondIndex] < secondRandomRange):
    # secondIndex += 1

  # secondIndividual = sortedPopulation[secondIndex]
#==================================================================================

#========================= Stochastic Universal Sampling ==========================
  # first random individual
  # firstRandomRange = np.random.randint(0, totalScore+1)
  # firstIndex = 0
  # while(accumulated[firstIndex] < firstRandomRange):
  # 	firstIndex += 1
  #
  # firstIndividual = sortedPopulation[firstIndex]
  #
  # # second random individual
  # secondIndex = int((self.size / 2 + firstIndex) % self.size)
  # secondIndividual = sortedPopulation[secondIndex]
  #
  # # do crossover
  # firstCrossed, secondCrossed = self.crossover(firstIndividual, secondIndividual, numNodes)
#==================================================================================

  # add to new population
  # newPopulation.append(firstCrossed)
  # if(len(newPopulation) < self.size-1):
  # 	newPopulation.append(secondCrossed)
