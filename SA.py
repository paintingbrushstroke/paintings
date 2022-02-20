from Colour_Painting_Pillow import *
from collections import Counter
import math
import sys
import pickle
import argparse
import glob
from datetime import datetime
import cv2
import time
import cProfile


def simulated_annealing(painting, evaluations, filename):
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    today = str(dt_string)

    logger  = "output_dir/"+filename+"/log-SA-" + str(len(painting.strokes))+ "-" + str(evaluations) + "-" + today
    f = open(logger, "w")
    for i in range(evaluations):
        mutatedStrokes = painting.mutate()

        # calculate error
        error, img = painting.calcError(mutatedStrokes)

        # if the MSE is lowered accept the mutated stroke
        if error < painting.current_error:
            painting.current_error = error
            painting.canvas_memory = img
            painting.strokes = mutatedStrokes

            painting.current_best_error = error
            painting.current_best_canvas = img

            print(i, ". new best:", painting.current_error)
            writeTolog(f, i, painting.current_error, strokeAnalyze(painting), 0)

        elif error > painting.current_error:
            # Calculate the probability to accept a worse stroke
            probability = calcProb(error, painting.current_error, i)

            # if the random value is within the probability accept the new stroke
            if random.random() < probability:
                painting.current_error = error
                painting.canvas_memory = img
                painting.strokes = mutatedStrokes
                print(i, ". SA Accept new best:", painting.current_error)
                writeTolog(f, i, painting.current_error, strokeAnalyze(painting), 1)
        sys.stdout.flush()
        f.flush()

        # if i == 250000 or i == 0 or i == 500000 or i == 750000:
            # pickle.dump( painting, open( "output_dir/" + filename + "/SA-" + str(len(painting.strokes)) + "-"+ today+ "-" + str(i) +".p", "wb" ) )
        if i%1000 == 0:
            painting.canvas_memory.save("output_dir/" + filename + "/SA-intermediate-" + str(len(painting.strokes)) + "-" + today + ".png", "PNG")

    # Final image
    # pickle.dump( painting, open( "output_dir/" + filename + "/SA-" + str(len(painting.strokes)) + "-"+ today+ "-final.p", "wb" ) )
    painting.canvas_memory.save("output_dir/" + filename + "/SA-final-" + str(len(painting.strokes)) + "-" + today + ".png", "PNG")


def calcProb(new_error, old_error, i):
    # Calculate the probability using the cooling function
    c = 1
    temperature = c / (math.log(i+1))
    delta_MSE = new_error - old_error
    prob = math.e ** (-delta_MSE/temperature)

    return prob


def writeTolog(f, evalCount, error, countedStrokes, SA):
    t = time.localtime()
    t = time.strftime("%H:%M:%S", t)
    # Add evaluations

    log = str(t) + "," + str(evalCount)
    log = log + "," + str(error)
    log = log + "," + str(countedStrokes[1]) + "," + str(countedStrokes[2]) + "," + str(countedStrokes[3]) + "," + str(countedStrokes[4])
    log = log + "," + str(SA)
    f.write(log + "\n")


def strokeAnalyze(individual):
    strokeTypes = []
    for stroke in individual.strokes:
        strokeTypes.append(stroke.brush_type)
    countedStrokes = Counter(strokeTypes)

    return countedStrokes


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('argfilename', metavar='N', nargs='+', help='painting name')
    args = parser.parse_args()

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

    strokeCount = 125
    evaluations = 100000
    mutationStrength = 0.2

    for j in range(1):
        canvas = Painting(imagePath, False, mutationStrength)
        canvas.init_strokes(strokeCount)
        strokes = canvas.strokes

        oldMutation = True
        canvas = Painting(imagePath, oldMutation, mutationStrength)
        canvas.init_strokes(strokeCount)
        canvas.strokes = strokes
        simulated_annealing(canvas, evaluations, filename)

        oldMutation = False
        canvas = Painting(imagePath, oldMutation, mutationStrength)
        canvas.init_strokes(strokeCount)
        canvas.strokes = strokes
        simulated_annealing(canvas, evaluations, filename)
