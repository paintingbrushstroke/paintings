from Colour_Painting_Pillow import *
import math
import random
import copy
import time
import sys
import argparse
import glob
from datetime import datetime

from colorthief import ColorThief
from PIL import ImageDraw
from multiprocessing import Process, Queue, current_process, freeze_support

def initPopulation(populationSize, strokeCount, imagePath, palette):
    # initialize population
    population = []
    # initStrokes = []
    for i in range(populationSize):
        individual = Painting(imagePath, False, palette)
        individual.init_strokes(strokeCount)
        # initStrokes.append(individual.strokes)
        population.append(individual)

    return population


def worker(input, output):
    for func, args in iter(input.get, 'STOP'):
        result = calculate(func, args)
        output.put(result)

def calculate(func, args):
    result = func(*args)
    #return '%s says that %s%s = %s' % \
    #    (current_process().name, func.__name__, args, result)
    return result

def calcPopulationMSEPAR(population, task_queue, done_queue):
    # calculate MSE for the entire population (parallel execution)
    # errors = []
    # imgs = []
    # for i in range(len(population)):
    #     error, img = population[i].calcError(population[i].strokes)
    #     errors.append(error)
    #     imgs.append(img)
    #     population[i].current_error = errors[i]
    #     population[i].current_pheno = img

    NUMBER_OF_PROCESSES = len(population)
    TASKS1 = [(population[0].calcErrorForParpool, (population[i].strokes, i)) for i in range(len(population))]

    # Submit tasks
    for task in TASKS1:
        task_queue.put(task)

    # Start worker processes
    for i in range(NUMBER_OF_PROCESSES):
        Process(target=worker, args=(task_queue, done_queue)).start()

    # Get and print results
    errors = [None]*len(TASKS1)
    imgs = [None]*len(TASKS1)
    for i in range(len(TASKS1)):
        res = done_queue.get()
        errors[res[2]] = res[0]
        imgs[res[2]] = res[1]

    for i in range(NUMBER_OF_PROCESSES):
        task_queue.put('STOP')

    for i in range(len(population)):
        population[i].current_error = errors[i]
        population[i].current_pheno = imgs[i]

    return errors, population


def calcPopulationMSE(population):
    # calculate MSE for the entire population
    errors = []
    imgs = []
    for i in range(len(population)):
        error, img = population[i].calcError(population[i].strokes)
        errors.append(error)
        imgs.append(img)
        population[i].current_error = error
        population[i].current_pheno = img

    return errors, population


def generateOffspring(parents, errors, mutPerc, mutationStrength, recombinationThreshold):
    # generate offspring per indidivual of the population
    children = []
    nMutations = int(mutPerc*len(parents[0].strokes))
    offSpringCount = int(len(parents)/2)
    for offspring in range(offSpringCount):
        # Recombination
        parentA = parents[offspring]
        parentB = parents[offspring+offSpringCount]
        errorA = errors[offspring]
        errorB = errors[offspring+offSpringCount]
        newChild = copy.deepcopy(parentA)
        if recombinationThreshold is not None:
            for strokeID in range(len(parentA.strokes)):
                randNr = random.random()
                if randNr < recombinationThreshold:
                    if errorA > errorB:
                        newChild.strokes[strokeID] = copy.deepcopy(parentB.strokes[strokeID])
                else:
                    if errorA <= errorB:
                        newChild.strokes[strokeID] = copy.deepcopy(parentB.strokes[strokeID])

        # Mutation
        for i in range(nMutations):
            newChild.strokes = newChild.mutate(mutationStrength)
        children.append(newChild)

    return children


def getConcatenation(images):
    count = sum(1 for e in images if e)
    if count < 2:
        print("None or only one image input for concatenation.")
        return
    nImageRows = int(math.sqrt(len(images)))
    imSize = images[0].width
    catImage = Image.new('RGBA', (imSize*nImageRows, imSize*nImageRows))
    for num, im in enumerate(images, start=0):
        row = int(num / nImageRows)
        col = num % nImageRows
        if im is None:
            print("Image empty, replacing with blank canvas")
            im = Image.new('RGB', (imSize, imSize))
        catImage.paste(im, (row*imSize, col*imSize))
    return catImage


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
    color_count = 32
    evaluations = 100000
    mutationSigma = 0.1
    mutPerc = 0.01
    recombinationThreshold = 0.5 # 0.6  # 0.6  # None

    nGenerations = math.ceil(evaluations/populationSize) - 1  # Subtract one for initial population
    print("Number of generations: " + str(nGenerations))

    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    today = str(dt_string)

    logger = "output_dir/"+ filename + "/log-GA-" + str(strokeCount) + "-" + str(populationSize) + "-" + today
    logger = logger + "-v" + str(len(glob.glob(logger)))
    f = open(logger, "w")

    color_thief = ColorThief(imagePath)
    palette = None
    # palette = color_thief.get_palette(color_count=color_count)

    # Create queues
    task_queue = Queue()
    done_queue = Queue()

    population = initPopulation(populationSize, strokeCount, imagePath, palette)

    for i in range(nGenerations):
        # Get fitness
        errors, population = calcPopulationMSEPAR(population,task_queue,done_queue)

        # Sort population
        sorted = np.argsort(errors)
        population = np.take(population, sorted)
        errors = np.take(errors, sorted)

        # Elitism
        newpopulation = []
        newpopulation.append(copy.deepcopy(population[0]))

        # Reporting
        if i%5 == 0:
            print("# Evals: " + str(i*populationSize) + " - Gen: " + str(i) + " - Best error: " + str(errors[0]))
            imgs = []
            for m in range(len(population)):
                img = copy.deepcopy(population[m].current_pheno)
                draw = ImageDraw.Draw(img)
                draw.text((0, 0),"Error: " + str(int(errors[m])),(255,255,255))  # ,font=font
                imgs.append(img)
            concatImage = getConcatenation(imgs)
            concatImage.save("output_dir/" + filename + "/GA-intermediate-" + str(strokeCount) + "-" + today + ".png", "PNG")

        #   Tournament selection (save 1 for elite-copy)
        parentCandidateIDs = np.random.randint(populationSize, high=None, size=[(populationSize-1)*2,2])
        tournament = np.take(errors, parentCandidateIDs)
        parentIDs = []
        for i in range(tournament.shape[0]):
            if tournament[i,0] < tournament[i,1]:
                parentIDs.append(parentCandidateIDs[i,0])
            else:
                parentIDs.append(parentCandidateIDs[i,1])

        children = generateOffspring(population[parentIDs], errors[parentIDs], mutPerc, mutationSigma, recombinationThreshold)
        population = newpopulation + children

        writeTolog(f, i*populationSize, errors[0])
        f.flush()
        sys.stdout.flush()
        #     start = time.time()
        #     # loggingx
        #     writeTolog(f, evalCount, population[0].current_error, len(offspring), countedStrokes, population[0].mutateCount, population[0].cycles_alive)

        #     end = time.time()
