import pickle
from collections import Counter
from Colour_Painting import *
import glob
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches


def logReader(logname):
   
    f = open(logname, "r")
    logX = []
    logY = []

    for line in f:
        splitted = line.strip("\n").split(",")
        logX.append(int(splitted[1]))
        logY.append(float(splitted[2]))

    return [logX,logY] 

def calcXY(path):
    x = []
    y = []
    y_new = []

    # get original data
    for data in glob.glob(path):

        newdata = logReader(data)
        x.append(newdata[0])
        y.append(newdata[1])

    # create new y and x that have data at every eval
    new_x = np.linspace(0, 1000000, 1000000)
    y_min = np.zeros_like(new_x)
    y_max = np.zeros_like(new_x)
    y_mean = np.zeros_like(new_x)
    new_y = []

    # Loop over all evaluations to find a corresponding y value
    for item_x in range(0,len(x)):
        y_list = []
        current_y_value = y[item_x][0]
        current_x_index = 0
        for i in range(0,len(new_x)):
            if int(i) == x[item_x][current_x_index]:
                # select only improvments, mostly for SA
                if current_y_value > y[item_x][current_x_index]:
                    current_y_value = y[item_x][current_x_index]

                # dont go out of bounce
                if current_x_index < len(x[item_x])-1:
                    current_x_index = current_x_index + 1

            # find max value of y for the upper bound
            if y_max[i] < current_y_value or y_max[i] == 0:
                y_max[i] = current_y_value

            # find min values of y for the lower bound
            if y_min[i] > current_y_value or y_min[i] == 0:
                y_min[i] = current_y_value

            y_list.append(current_y_value)
        new_y.append(y_list)

    for index_x in range(0, len(new_x)):
        value = 0
        for index_y in range(0, len(new_y)):
            value += new_y[index_y][index_x]
        y_mean[index_x] = value/len(new_y)

    return new_x, y_mean, y_min, y_max


if __name__ == "__main__":

    names = ["mona", "bach", "dali", "klimt", "mondriaan", "pollock", "starrynight"]
    x = [4, 25, 75, 125, 175, 250]
    titles = ["Mona Lisa","Johann Sebastian Bach", "The Persistence of Memory", "The kiss"
     , "Red, Yellow and Blue", "Convergence", "The Starry Night"]
    colors = ["lightcoral", "orchid", "royalblue", "mediumseagreen", "yellowgreen", "deepskyblue", "gray"]

    y = []

    for i in range(0, len(names)):
        paintinglist = []
        name  = names[i]
        
        for count in x:
            path = "output_dir/"+name+".png/SA-"+str(count)+"*-final.p"
            current = 0
            for data in glob.glob(path):
                canvas =  pickle.load(open(data, "rb"))
                current += canvas.current_error
            paintinglist.append(current/5)

        y.append(paintinglist)

    fig, ax = plt.subplots()
    for i in range(0, len(names)):
        name  = names[i]
        ax.plot(x, y[i], '-ok', label=titles[i], color=colors[i])

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)

    plt.xlabel("Brush stroke counts")
    plt.ylabel("Mean squared error")
    ax.set_ylim(230, 39052)

    ax.set_yscale('log')
    plt.title("Simulated Annealing")
    plt.show()



            







