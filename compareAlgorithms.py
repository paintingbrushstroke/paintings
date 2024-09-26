from Colour_Painting_Pillow import *
import argparse
import glob
import os
from datetime import datetime

from PIL import ImageDraw, ImageFont
import GA as ga
import SA as sa

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('argfilename', metavar='N', nargs='+',
                        help='painting name')
    args = parser.parse_args()

    # setup parameters
    filename = str(args.argfilename[0])
    imagePath = "imgs/" + filename
    evaluations = 50000
    mutationSigma = 0.1
    strokeCount = 125
    colorCount = 32
    nReplicates = 5

    for rep in range(nReplicates):
        try:
            os.mkdir("output_dir")
        except Exception:
            print("Dir exists")
        mainoutputfolder = "output_dir/" + filename
        try:
            os.mkdir(mainoutputfolder)
        except Exception:
            print("Dir exists")

        outputfolder = "output_dir/" + filename + '/split_0_old_1'
        try:
            os.mkdir(outputfolder)
        except Exception:
            print("Dir exists")
        initColorsMedSplit = False
        oldMutation = True
        ga.genetic_algorithm(imagePath, outputfolder, evaluations, strokeCount, mutationSigma, oldMutation, initColorsMedSplit, colorCount)
        sa.simulated_annealing(imagePath, outputfolder, evaluations, strokeCount, mutationSigma, oldMutation, initColorsMedSplit, colorCount)

        outputfolder = "output_dir/" + filename + '/split_0_old_0'
        try:
            os.mkdir(outputfolder)
        except Exception:
            print("Dir exists")
        initColorsMedSplit = False
        oldMutation = False
        ga.genetic_algorithm(imagePath, outputfolder, evaluations, strokeCount, mutationSigma, oldMutation, initColorsMedSplit, colorCount)
        sa.simulated_annealing(imagePath, outputfolder, evaluations, strokeCount, mutationSigma, oldMutation, initColorsMedSplit, colorCount)

        outputfolder = "output_dir/" + filename + '/split_1_old_0'
        try:
            os.mkdir(outputfolder)
        except Exception:
            print("Dir exists")
        initColorsMedSplit = True
        oldMutation = False
        ga.genetic_algorithm(imagePath, outputfolder, evaluations, strokeCount, mutationSigma, oldMutation, initColorsMedSplit, colorCount)
        sa.simulated_annealing(imagePath, outputfolder, evaluations, strokeCount, mutationSigma, oldMutation, initColorsMedSplit, colorCount)
