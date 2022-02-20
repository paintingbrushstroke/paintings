from Colour_Painting_Pillow import *
import math
from random import random
import copy
import time
from collections import Counter
import sys
import pickle
import argparse
import glob
from datetime import datetime


def initPopulation(populationSize, strokeCount, imagePath, mutationStrength):
    # initialize population
    population = []
    initStrokes = []
    for i in range(populationSize):
        individual = Painting(imagePath, False, mutationStrength)
        individual.init_strokes(strokeCount)
        initStrokes.append(individual.strokes)
        population.append(individual)

    return population, initStrokes


def calcPopulationMSEPAR(population):
    # calculate MSE for the entire population
    errors = []
    imgs = []
    for i in range(len(population)):
        error, img = population[i].calcError(population[i].strokes)
        errors.append(error)
        imgs.append(img)
        population[i].current_error = errors[i]
        population[i].current_pheno = imgs[i]

    process_list = []
    nProc = 4
    for i in range(nProc):
        p = multiprocessing.Process(target=population[i].calcError, args=[population[i].strokes])
        p.start()
        process_list.append(p)

    for process in process_list:
        process.join()

    return errors, population


def calcPopulationMSE(population):
    # calculate MSE for the entire population
    errors = []
    imgs = []
    for i in range(len(population)):
        error, img = population[i].calcError(population[i].strokes)
        errors.append(error)
        imgs.append(img)
        population[i].current_error = errors[i]
        population[i].current_pheno = imgs[i]

    return errors, population

def sortPopulation(population, fitnessList, populationSize):
    # Sort list by fitness score
    sortedFitnessList = Sort(fitnessList)

    sortedPopulation = []

    # loop over the sorted fitness list and extract the correct indiviual from the population to sort the population
    for idx, _ in sortedFitnessList:
        # Add one to cycle alive count, to see how long the best indiviual stays alive
        population[idx].cycles_alive = population[idx].cycles_alive + 1
        sortedPopulation.append(population[idx])

    # set the sorted population as the true population and return the population to its original size.
    return sortedPopulation[:populationSize]


def Sort(list):
    list.sort(reverse=True, key=lambda x: x[1])
    return list


def generateOffspring(parents):
    # generate offspring per indidivual of the population
    children = []
    offSpringCount = len(parents) - 1

    for offspring in range(offSpringCount):
        # copy the parent, no fitness based selection  here yet
        newChild = copy.deepcopy(parents[offspring])
        newChild.strokes = newChild.mutate()
        children.append(newChild)

    return children


def strokeAnalyze(individual):
    strokeTypes = []
    for stroke in individual.strokes:
        strokeTypes.append(stroke.brush_type)
    countedStrokes = Counter(strokeTypes)

    return countedStrokes


def writeTolog(f, evalCount, error, offSpringCount, countedStrokes, mutateCount, cycles_alive):
    t = time.localtime()
    t = time.strftime("%H:%M:%S", t)
    # Add evaluations
    log = str(t) + "," + str(evalCount)
    log = log + "," + str(error)
    log = log + "," + str(offSpringCount)
    log = log + "," + str(countedStrokes[1]) + "," + str(countedStrokes[2]) + "," + str(countedStrokes[3]) + "," + str(countedStrokes[4])
    log = log + "," + str(mutateCount)
    log = log + "," + str(cycles_alive)

    f.write(log + "\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('argfilename', metavar='N', nargs='+',
                        help='painting name')
    args = parser.parse_args()

    print("start GA")

    # setup parameters
    filename = str(args.argfilename[0])
    imagePath = "imgs/" + filename

    try:
        os.mkdir("output_dir")
    except Exception:
        print("Dir exists")
    try:
        os.mkdir("output_dir/"+filename)
    except Exception:
        print("Dir exists")

    populationSize = 25
    strokeCount = 125
    evaluations = 10000
    mutationStrength = 0.1

    nGenerations = math.ceil(evaluations/populationSize) - 1  # Subtract one for initial population
    print("Number of generations: " + str(nGenerations))

    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    today = str(dt_string)

    population, initStrokes = initPopulation(populationSize, strokeCount, imagePath, mutationStrength)
    for i in range(nGenerations):
        # Get fitness
        errors, population = calcPopulationMSE(population)
        # print(errors)
        # Sort population
        sorted = np.argsort(errors)
        population = np.take(population, sorted)
        errors = np.take(errors, sorted)

        # Elitism
        newpopulation = []
        newpopulation.append(population[0])
        print("# Evals: " + str(i*populationSize) + " - Gen: " + str(i) + " - Best fitness: " + str(errors[0]))
        population[0].current_pheno.save("output_dir/" + filename + "/GA-intermediate-" + str(strokeCount) + "-" + today + ".png", "PNG")

        # Create children (mutation) crossover later?
        #   Tournament selection (save 1 for elite-copy)
        parentCandidateIDs = np.random.randint(populationSize - 1, high=None, size=[populationSize,2])
        tournament = np.take(errors, parentCandidateIDs)
        parentIDs = []
        for i in range(tournament.shape[0]):
            if tournament[i,0] < tournament[i,1]:
                parentIDs.append(parentCandidateIDs[i,0])
            else:
                parentIDs.append(parentCandidateIDs[i,1])

        #   Mutate parents
        children = generateOffspring(population[parentIDs])
        population = newpopulation + children

        # logger = "output_dir/"+ filename + "/log-PPA-" + str(strokeCount) + "-" + str((i+1)*populationSize) + "-" + today
        # logger = logger + "-v" + str(len(glob.glob(logger)))
        # f = open(logger, "w")
        #     start = time.time()
        #     # loggingx
        #     writeTolog(f, evalCount, population[0].current_error, len(offspring), countedStrokes, population[0].mutateCount, population[0].cycles_alive)

        #     end = time.time()
        #     f.flush()
        #     sys.stdout.flush()
