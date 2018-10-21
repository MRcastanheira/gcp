#py gcp.py -i flat1000_76_0.col -o output.csv -g 100 -p 5 -m 0.8 -c 0.8 -e 0.1

import os
import pathlib

def run(file, input, generations, population, mutationRate, crossoverRate, elitesRate, fitnessValidFactor, upgradeMutationRate):
    global path
    global fixPath
    global rootDir

    instanceName = input.split("/")
    instanceName = instanceName[len(instanceName)-1]
    dir = path + rootDir + "/" + instanceName

    #print(dir)

    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

    output = fixPath + rootDir + "\\" + instanceName + "\\"
    filename = ""
    filename += instanceName
    filename += "_g" + generations
    filename += "_p" + population
    filename += "_m" + mutationRate
    filename += "_c" + crossoverRate
    filename += "_e" + elitesRate
    filename += "_f" + fitnessValidFactor
    filename += "_u" + upgradeMutationRate

    output += filename

    #print(output)

    outputname = filename + ".csv"

    command = "py " + file
    command += " -i " + "\"" + input + ".col" + "\""
    command += " -o " + "\"" + output.replace("/", "\\") + ".csv" + "\""
    command += " -g " + generations
    command += " -p " + population
    command += " -m " + mutationRate
    command += " -c " + crossoverRate
    command += " -e " + elitesRate
    command += " -f " + fitnessValidFactor
    command += " -u " + upgradeMutationRate
    #print(command)
    os.system(command)

    columns = [1, 2, 3, 4]
    graphCommand = "py plotGraph.py"
    graphCommand += " -i " + "\"" + dir + "/" + filename + ".csv" + "\""
    for c in columns:
        graphCommandColumn = " -c " + str(c)
        #print(graphCommand + graphCommandColumn)
        os.system(graphCommand + graphCommandColumn)

# params ---
rootDir = "outputsRelatorio1" # -- root dir to save outputs
allowedUsers = ["Diego", "Matheus"] # should be your system username
allowedUsersPath = ["C:/Users/Diego/Desktop/Computacao evolutiva/gcp/", "D:/Matheus/Documents/Google Drive/UFRGS Estudo/Semestre 11/Computação Evolutiva/Trabalho Prático I/"]
useRelativePath = 0 # 0 -- relative, 1 -- use allowed users

file = "gcp.py"
#inputList = ["queen7_7"]
inputList = ["input/anna", "david", "fpsol2.i.1", "games120", "homer", "huck", "jean", "miles250", "miles1000", "myciel3", "myciel3", "myciel4", "myciel5", "queen5_5", "queen6_6", "queen7_7", "queen8_8"]
#inputList = ["simple", "complicated", "dsjc500.1", "flat1000_76_0"]
#inputList = ["simple"]
generationsList = ["500", "10000"]
populationList = ["5", "20"]
mutationRateList = ["0.001", "0.01", "0.05", "0.1"]
crossoverRateList = ["0.8", "0.6"]
elitesRateList = ["0.1", "0.2"]
fitnessValidFactorList = ["10", "1", "100"]
upgradeMutationRateList = ["0.8", "0.5"]

# end of params ---

# setup
if not useRelativePath:
    path = ""
    local = os.path.abspath("")
    for i, user in enumerate(allowedUsers):
        if user in local:
            path = allowedUsersPath[i]
else:
    path = "" # relative

fixPath = path.replace("/", "\\")

# run
for input in inputList:
    for generations in generationsList:
        for population in populationList:
            for mutationRate in mutationRateList:
                for crossoverRate in crossoverRateList:
                    for elitesRate in elitesRateList:
                        for fitnessValidFactor in fitnessValidFactorList:
                            for upgradeMutationRate in upgradeMutationRateList:
                                run(file, input, generations, population, mutationRate, crossoverRate, elitesRate, fitnessValidFactor, upgradeMutationRate)
