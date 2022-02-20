from Colour_Painting_Pillow import *
import math
import random
import copy
import time
from collections import Counter
import sys
import pickle
import argparse
import glob
from datetime import datetime

from colorthief import ColorThief

def initPopulation(populationSize, strokeCount, imagePath, mutationStrength):
    # initialize population
    population = []
    initStrokes = []
    for i in range(populationSize):
        individual = Painting(imagePath, False, mutationStrength)
        individual.init_strokes(strokeCount)
        # initStrokes.append(individual.strokes)
        population.append(individual)

    return population


def calcPopulationMSEPAR(population):
    # calculate MSE for the entire population
    errors = []
    imgs = []
    for i in range(len(population)):
        error, img = population[i].calcError(population[i].strokes)
        errors.append(error)
        imgs.append(img)
        population[i].current_error = errors[i]
        population[i].current_pheno = img

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


def generateOffspring(parents, errors, recombinationThreshold):
    # generate offspring per indidivual of the population
    children = []
    offSpringCount = int(len(parents)/2) - 1
    for offspring in range(offSpringCount):
        # Recombination
        parentA = copy.deepcopy(parents[offspring])
        parentB = copy.deepcopy(parents[offspring+offSpringCount])
        errorA = errors[offspring]
        errorB = errors[offspring+offSpringCount]
        newChild = copy.deepcopy(parentA)
        for strokeID in range(len(parentA.strokes)):
            randNr = random.uniform(0.0, 1.0)
            if randNr < recombinationThreshold and errorA < errorB:
                selectGene = parentA.strokes[strokeID]
            elif randNr > recombinationThreshold and errorA > errorB:
                selectGene = parentA.strokes[strokeID]
            else:
                selectGene = parentB.strokes[strokeID]
            newChild.strokes[strokeID] = selectGene

        # Mutation
        newChild.strokes = newChild.mutate()
        children.append(newChild)

    return children

def assignColor(population, palette):
    for i in range(len(population)):
        for j in range(len(population[i].strokes)):
            population[i].strokes[j].color = list(palette[j])

    return population

def strokeAnalyze(individual):
    strokeTypes = []
    for stroke in individual.strokes:
        strokeTypes.append(stroke.brush_type)
    countedStrokes = Counter(strokeTypes)

    return countedStrokes


def writeTolog(f, evalCount, error):
    t = time.localtime()
    t = time.strftime("%H:%M:%S", t)
    # Add evaluations
    log = str(t) + "," + str(evalCount)
    log = log + "," + str(error)

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

    populationSize = 32
    strokeCount = 125
    evaluations = 100000
    mutationStrength = 0.1
    recombinationThreshold = 0.6

    nGenerations = math.ceil(evaluations/populationSize) - 1  # Subtract one for initial population
    print("Number of generations: " + str(nGenerations))

    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    today = str(dt_string)

    logger = "output_dir/"+ filename + "/log-GA-" + str(strokeCount) + "-" + str(populationSize) + "-" + today
    logger = logger + "-v" + str(len(glob.glob(logger)))
    f = open(logger, "w")

    color_thief = ColorThief(imagePath)
    palette = color_thief.get_palette(color_count=strokeCount+1)
    population = initPopulation(populationSize, strokeCount, imagePath, mutationStrength)
    population = assignColor(population, palette)
    for i in range(nGenerations):
        # Get fitness
        errors, population = calcPopulationMSE(population)

        # Sort population
        sorted = np.argsort(errors)
        population = np.take(population, sorted)
        errors = np.take(errors, sorted)

        # Elitism
        newpopulation = []
        newpopulation.append(population[0])
        if i%5 == 0:
            print("# Evals: " + str(i*populationSize) + " - Gen: " + str(i) + " - Best fitness: " + str(errors[0]))
            population[0].current_pheno.save("output_dir/" + filename + "/GA-intermediate-" + str(strokeCount) + "-" + today + ".png", "PNG")
            # cv2.imwrite("output_dir/" + filename + "/GA-intermediate-" + str(strokeCount) + "-" + today + ".png", population[0].current_pheno)

        # Create children (mutation) crossover later?
        #   Tournament selection (save 1 for elite-copy)
        parentCandidateIDs = np.random.randint(populationSize - 1, high=None, size=[populationSize*2,2])
        tournament = np.take(errors, parentCandidateIDs)
        parentIDs = []
        for i in range(tournament.shape[0]):
            if tournament[i,0] < tournament[i,1]:
                parentIDs.append(parentCandidateIDs[i,0])
            else:
                parentIDs.append(parentCandidateIDs[i,1])

        #   Mutate parents
        children = generateOffspring(population[parentIDs], errors[parentIDs], recombinationThreshold)
        population = newpopulation + children

        writeTolog(f, i*populationSize, errors[0])
        f.flush()
        sys.stdout.flush()
        #     start = time.time()
        #     # loggingx
        #     writeTolog(f, evalCount, population[0].current_error, len(offspring), countedStrokes, population[0].mutateCount, population[0].cycles_alive)

        #     end = time.time()
