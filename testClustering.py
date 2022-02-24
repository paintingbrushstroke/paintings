from __future__ import print_function

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

import binascii
import struct
from PIL import Image
import numpy as np
import scipy
import scipy.misc
import scipy.cluster

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

    NUM_CLUSTERS = 32

    ar = np.asarray(individual.original_img)
    shape = ar.shape
    ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

    print('finding clusters')
    codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
    print('cluster centres:\n', codes)

    vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
    counts, bins = np.histogram(vecs, len(codes))    # count occurrences

    print("Counts: " + str(counts))
    index_max = scipy.argmax(counts)                    # find most frequent
    peak = codes[index_max]
    colour = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')
    print('most frequent is %s (#%s)' % (peak, colour))
