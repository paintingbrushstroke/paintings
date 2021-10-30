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
                # if y_max[i-10000] == y_max[i]:
                #     print(path + ":" + str(current_y_value))

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

    for alg in  ["SA", "HC", "PPA"]:

        names = ["mona", "bach", "dali", "klimt", "mondriaan", "pollock", "starrynight"]
        limits = [1200,1200,3000,6000,750,25000,3000]
        titles = ["Mona Lisa","Johann Sebastian Bach", "The Persistence of Memory", "The kiss"
        , "Red, Yellow and Blue", "Convergence", "The Starry Night"]

        for i in range(0, len(names)):
            name  = names[i]

            fig, ax = plt.subplots()

            path = "output_dir/"+name+".png/log-"+alg+"-250*"

            new_x, y_mean, y_min, y_max = calcXY(path)

            ax.plot(new_x, y_mean, label="250 brush strokes", color="red")
            ax.fill_between(new_x,y_min, y_max, alpha=0.2, color="red")

            path = "output_dir/"+name+".png/log-"+alg+"-175*"

            new_x, y_mean, y_min, y_max = calcXY(path)

            ax.plot(new_x, y_mean, label="175 brush strokes", color="blue")
            ax.fill_between(new_x,y_min, y_max, alpha=0.2, color="blue")

            path = "output_dir/"+name+".png/log-"+alg+"-125*"

            new_x, y_mean, y_min, y_max = calcXY(path)


            ax.plot(new_x, y_mean, label="125 brush strokes", color="black")
            ax.fill_between(new_x,y_min, y_max, alpha=0.2, color="black")

            path = "output_dir/"+name+".png/log-"+alg+"-75*"

            new_x, y_mean, y_min, y_max = calcXY(path)

            ax.plot(new_x, y_mean, label="75 brush strokes", color="yellow")
            ax.fill_between(new_x,y_min, y_max, alpha=0.2, color="yellow")

            path = "output_dir/"+name+".png/log-"+alg+"-25*"

            new_x, y_mean, y_min, y_max = calcXY(path)

            ax.plot(new_x, y_mean, label="25 brush strokes", color="purple")
            ax.fill_between(new_x,y_min, y_max, alpha=0.2, color="purple")


            path = "output_dir/"+name+".png/log-"+alg+"-4*"

            new_x, y_mean, y_min, y_max = calcXY(path)

            ax.plot(new_x, y_mean, label="4 brush strokes", color="green")
            ax.fill_between(new_x,y_min, y_max, alpha=0.2, color="green")

            ax.legend()

            plt.xlabel("Evaluations")
            plt.ylabel("Mean squared error")
            plt.ylim(0,limits[i])
            plt.title(titles[i] )

            plt.savefig("Results/MSE/"  +alg+ "-"+name+ "-combined.png")



            







