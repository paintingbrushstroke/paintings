import cv2
import numpy as np
from Colour_Painting import *
import math
from random import random
import copy
import time
from collections import Counter
from datetime import date
import sys
import pickle
import argparse
import glob
from datetime import datetime

def initPopulation(populationSize, strokeCount, imagePath):
    # initilize population
    population = []
    for i in range(populationSize):
        individual = Painting(imagePath)
        individual.init_strokes(strokeCount)

        population.append(individual)

    return population

def calcPopulationMSE(population, evalCount):
     # calculate MSE for the entire population
    minMSE = 1000000000
    maxMSE = 0
    outputGif = False
    for individual in population:
        if individual.MSE_calced == False:
            error, image = individual.calcError(individual.strokes)
            individual.current_error = error
            individual.canvas_memory = image

            # the MSE if now recalculated which counts as one evaluation
            individual.MSE_calced = True
            if evalCount == 0 or evalCount == 250000 or evalCount == 500000 or evalCount == 750000:
                outputGif = True
            evalCount = evalCount + 1


        # find min and max MSE to calculate fitness
        if minMSE > individual.current_error: minMSE = individual.current_error
        if maxMSE < individual.current_error: maxMSE = individual.current_error
    
    return (minMSE, maxMSE, evalCount, outputGif)

def calcPopulationFitness(population, minMSE, maxMSE):
    # calulate fitness for entire population. Also keep track of all fitness scores per index for sorting later
    fitnessList = []
    for idx, individual in enumerate(population):
        fitness = (maxMSE - individual.current_error)/(maxMSE-minMSE)
        individual.fitness = fitness
        fitnessList.append((idx, fitness))

        # also calculate normalized fitness 
        individual.norm_fitness = 0.5 * (math.tanh(4*fitness-2)+1)
    return fitnessList

def sortPopulation(population, fitnessList, populationSize):
    # Sort list by fitness score
    sortedFitnessList = Sort(fitnessList)

    sortedPopulation = []

    # loop over the sorted fitness list and extract the correct indiviual from the population to sort the population
    for idx , _ in sortedFitnessList:
        # Add one to cycle alive count, to see how long the best indiviual stays alive
        population[idx].cycles_alive = population[idx].cycles_alive + 1 
        sortedPopulation.append(population[idx])


    # set the sorted population as the true population and return the population to its original size.
    return sortedPopulation[:populationSize]

def Sort(list):
  
    list.sort(reverse=True, key = lambda x: x[1])
    return list

def generateOffspring(population, maxOffspring):
    # generate offspring per indidivual of the population
    childs = []
    for individual in population:
        offSpringCount = math.ceil(maxOffspring * individual.norm_fitness * random())
        mutationCount = math.ceil(100 * (1/maxOffspring) * (1-individual.norm_fitness) * random())
       
        # generate x amount of childs based on the offspring count        
        for offspring in range(offSpringCount):
            # copy the parent 
            newChild = copy.deepcopy(individual)

            # # make sure that after the mutations the MSE is recalculated
            newChild.MSE_calced = False
            newChild.cycles_alive = 0
            newChild.mutateCount = mutationCount

            # mutate the offspring based on the mutation count
            for mutations in range(mutationCount):
                newChild.strokes = newChild.mutate()
            childs.append(newChild)

    return childs

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

   
    print("start PPA")

    # setup parameters 
    populationSize = 30
    filename = str(args.argfilename[0])
    imagePath = "imgs/" + filename

    strokeCount = 250
    maxOffspring = 5
    evaluations = 1000000

    population = initPopulation(populationSize, strokeCount, imagePath)

    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    today = str(dt_string)
    
    evalCount = 0
    logger  = "output_dir/"+ filename + "/log-PPA-" + str(strokeCount) + "-" + str(evaluations) + "-" + today
    logger = logger + "-v" + str(len(glob.glob(logger)))   
    f = open(logger, "w")
    while evalCount < evaluations:

        start = time.time()

        # calculations
        minMSE, maxMSE, evalCount, outputPickle = calcPopulationMSE(population, evalCount)
        fitnessList = calcPopulationFitness(population, minMSE, maxMSE)
        
        # population operations
        population = sortPopulation(population, fitnessList, populationSize)
        offspring = generateOffspring(population, maxOffspring)
        population = population + offspring   

        # loggingx
        countedStrokes = strokeAnalyze(population[0])
        writeTolog(f, evalCount, population[0].current_error, len(offspring), countedStrokes, population[0].mutateCount, population[0].cycles_alive )

        if outputPickle:
            pickle.dump( population[0], open( "output_dir/" + filename + "/population-" + str(strokeCount) + "-"+ today+ "-" + str(evalCount) +".p", "wb" ) )

        end = time.time()
        print(evalCount, population[0].current_error, len(offspring), "Full duration " + str(end - start))

        f.flush()
        sys.stdout.flush()

    
    cv2.imwrite("output_dir/"+ filename +"/PPA-final-" + str(strokeCount) + "-" + today + ".png" , population[0].canvas_memory)
    pickle.dump( population[0], open( "output_dir/" + filename + "/population-" + str(strokeCount) + "-"+ today+ "-final.p", "wb" ) )
