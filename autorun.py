#py gcp.py -i flat1000_76_0.col -o output.csv -g 100 -p 5 -m 0.8 -c 0.8 -e 0.1

import os
import pathlib

def run(file, input, generations, population, mutationRate, crossoverRate, elitesRate):
    global path
    global fixPath
    global rootDir

    dir = path + rootDir + "/" + input

    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

    output = fixPath + rootDir + "\\" + input + "\\"
    output += input
    output += "_g" + generations
    output += "_p" + population
    output += "_m" + mutationRate
    output += "_c" + crossoverRate
    output += "_e" + elitesRate

    command = "py " + file
    command += " -i " + input + ".col"
    command += " -o " + "\"" + output + ".csv" + "\""
    command += " -g " + generations
    command += " -p " + population
    command += " -m " + mutationRate
    command += " -c " + crossoverRate
    command += " -e " + elitesRate
    print(command)
    os.system(command)

# params ---
rootDir = "outputs" # -- root dir to save outputs
allowedUsers = ["Diego", "Matheus"] # should be your system username
allowedUsersPath = ["C:/Users/Diego/Desktop/Computacao evolutiva/gcp/", "D:/Matheus/Documents/Google Drive/UFRGS Estudo/Semestre 11/Computação Evolutiva/Trabalho Prático I/"]
useRelativePath = 0 # 0 -- relative, 1 -- use allowed users

file = "gcp.py"
inputList = ["simple", "complicated", "dsjc500.1", "flat1000_76_0"]
generationsList = ["100", "500", "1000"]
populationList = ["5", "20", "50"]
mutationRateList = ["0.8"]
crossoverRateList = ["0.8", "0.6"]
elitesRateList = ["0.1", "0.2"]
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
                        run(file, input, generations, population, mutationRate, crossoverRate, elitesRate)
