# from Colour_Painting import *
import Colour_Painting as pCV
import Colour_Painting_Pillow as pPillow
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

from PIL import Image


def editPainting(canvas, evaluations):
    for j in range(evaluations):
        canvas.draw(canvas.strokes)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('argfilename', metavar='N', nargs='+',
                    help='painting name')
    args = parser.parse_args()

    # setup parameters
    filename = str(args.argfilename[0])
    imagePath = "imgs/" + filename

    strokeCount = 100  # 0000
    evaluations = 100
    mutationStrength = 0.1

    canvas = pCV.Painting(imagePath, False, mutationStrength)
    canvas.init_strokes(strokeCount)
    canvas2 = pPillow.PaintingPillow(imagePath, False, mutationStrength)
    canvas2.init_strokes(strokeCount)
    canvas2.strokes = canvas.strokes

    start_time = time.time()
    # editPainting(canvas, evaluations)
    elapsed_time = time.time() - start_time
    print(elapsed_time)
    myImg = canvas.draw(canvas.strokes)
    myImg = Image.fromarray(myImg.astype('uint8'), 'RGB')
    myImg.show()

    start_time = time.time()
    # cProfile.run('editPainting(canvas2, evaluations)')
    editPainting(canvas2, evaluations)
    elapsed_time = time.time() - start_time
    print(elapsed_time)

    myImg = canvas2.draw(canvas2.strokes)
    myImg.show()
