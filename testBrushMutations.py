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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('argfilename', metavar='N', nargs='+',
                        help='painting name')
    args = parser.parse_args()

    print("start GA")

    # setup parameters
    filename = str(args.argfilename[0])
    imagePath = "imgs/" + filename

    strokeCount = 100
    evaluations = 100000
    mutationStrength = 0.2

    individual = Painting(imagePath, False, mutationStrength)
    individual.init_strokes(strokeCount)
    for i in range(100):
        mutatedStroke = individual.strokes[strokeCount-1]
        mutatedStroke.posY, mutatedStroke.posX = mutatedStroke.mut_positions(individual.bound, individual.padding, [mutatedStroke.posY, mutatedStroke.posX], individual.mutationStrength, mutatedStroke.size)
        error, img = individual.calcError(individual.strokes)
        time.sleep(0.2)
        img.show()
        # newChild.strokes = newChild.mutate()
