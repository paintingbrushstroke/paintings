from Colour_Painting import *
from collections import Counter
import argparse
import pickle
import sys
import glob
from datetime import datetime 
import cv2
import time



def hillclimber(painting, evaluations, filename):
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    today = str(dt_string)
    logger  = "output_dir/"+filename+"/log-HC-" + str(len(painting.strokes))+ "-" + str(evaluations) + "-" + today 

    f = open(logger, "w")
    for i in range(evaluations):
        mutatedStrokes = painting.mutate()

        # calculate error
        error, img = painting.calcError(mutatedStrokes)

        # if the error is lowered set the mutated version as the new version
        if error < painting.current_error:
            print(i, ". new best:", painting.current_error)
            painting.current_error = error
            painting.canvas_memory = img 
            painting.strokes = mutatedStrokes

            t = time.localtime()
            t = time.strftime("%H:%M:%S", t)
            writeTolog(t, f, i, painting.current_error, strokeAnalyze(painting))

        sys.stdout.flush()
        f.flush()

        # output pickle 
        if i == 250000 or i == 0 or i == 500000 or i == 750000:
            pickle.dump( painting, open( "output_dir/" + filename + "/HC-" + str(len(painting.strokes)) + "-"+ today+ "-" + str(i) +".p", "wb" ) )

    # Final image
    cv2.imwrite("output_dir/" + filename + "/HC-final-" + str(len(painting.strokes)) + "-" + today + ".png" , painting.canvas_memory)
    pickle.dump( painting, open( "output_dir/" + filename + "/HC-" + str(len(painting.strokes)) + "-"+ today+ "-final.p", "wb" ) )

def writeTolog(t, f, evalCount, error, countedStrokes):

    # Add evaluations
    log = str(t) + "," + str(evalCount)
    log = log + "," + str(error)
    log = log + "," + str(countedStrokes[1]) + "," + str(countedStrokes[2]) + "," + str(countedStrokes[3]) + "," + str(countedStrokes[4])

    f.write(log + "\n")


def strokeAnalyze(individual):
    strokeTypes = []
    for stroke in individual.strokes:
        strokeTypes.append(stroke.brush_type)
    countedStrokes = Counter(strokeTypes)

    return countedStrokes

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('argfilename', metavar='N', nargs='+',
                    help='painting name')
    args = parser.parse_args()

     # setup parameters 
    filename = str(args.argfilename[0])
    imagePath = "imgs/" + filename

    strokeCount = 250
    evaluations = 1000000

    canvas = Painting(imagePath)
    canvas.init_strokes(strokeCount)
    hillclimber(canvas, evaluations, filename)


 
