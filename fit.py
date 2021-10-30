import pickle
from collections import Counter
from Colour_Painting import *
import glob
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import scipy.optimize
from sklearn import metrics

def MSE(Y_true, Y_pred):
    mse = np.square(np.subtract(Y_true,Y_pred)).mean()
    return mse

def exp_new(x,a,b,c,d):
    y = a*np.power((x-b),-c) + d
    return y

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
    # Loop over a log file and return the corresponding x and y values

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

    limits = [1500,6000,6000,6000,3000,25000,6000]
    titles = ["Mona Lisa","Johann Sebastian Bach", "The Persistence of Memory", "The kiss"
     , "Red, Yellow and Blue", "Convergence", "The Starry Night"]

    for i in range(0, len(names)):
        name  = names[i]

        fig, ax = plt.subplots()

        # Boundary lines
        plt.vlines(x=1500, ymin=0, ymax=2000, linestyles= "dotted", color="lightgrey")
        plt.vlines(x=1000000, ymin=0, ymax=2000, linestyles="dotted",color="lightgrey")

        # Get xy values for HC
        path = "output_dir/"+name+".png/log-HC-250*"
        new_x, y_mean, y_min, y_max = calcXY(path)
        ax.plot(new_x, y_mean, label="Hillclimber", color="orchid")
        ax.fill_between(new_x,y_min, y_max, alpha=0.2, color="orchid")

        # Create fit for HC
        p0 = (917416, -7701, 0.63, 676) # start with values near those we expect
        params, cv = scipy.optimize.curve_fit(exp_new, xdata=new_x[1500:], ydata = y_mean[1500:], p0=p0)
        m, t, b, d = params
        HC_fit = exp_new(np.arange(start=1500, stop=5000000, step=100), m, t, b, d)
        plt.plot(np.arange(start=1500, stop=5000000, step=100), exp_new(np.arange(start=1500, stop=5000000, step=100), m, t, b, d), '--', color="purple")


        # Get xy value for PPA
        path = "output_dir/"+name+".png/log-PPA-250*"
        new_x, y_mean, y_min, y_max = calcXY(path)
        ax.plot(new_x, y_mean, label="Plant propagation", color="mediumseagreen")
        ax.fill_between(new_x,y_min, y_max, alpha=0.2, color="mediumseagreen")

        # Create fit for PPA
        p0 = (4138007,288,0.69,658) # start with values near those we expect
        params, cv = scipy.optimize.curve_fit(exp_new, xdata=new_x[1500:], ydata = y_mean[1500:], p0=p0)
        m, t, b, d = params
        PPA_fit = exp_new(np.arange(start=1500, stop=5000000, step=100), m, t, b, d)
        plt.plot(np.arange(start=1500, stop=5000000, step=100), exp_new(np.arange(start=1500, stop=5000000, step=100), m, t, b, d), '--', color="green")

        # Get xy values for SA
        path = "output_dir/"+name+".png/log-SA-250*"
        new_x, y_mean, y_min, y_max = calcXY(path)
        ax.plot(new_x, y_mean, label="Simulated annealing", color="deepskyblue")
        ax.fill_between(new_x,y_min, y_max, alpha=0.2, color="deepskyblue")

        # Create fit for SA
        p0 = (949172,-10244,0.61,505) # start with values near those we expect
        params, cv = scipy.optimize.curve_fit(exp_new, xdata=new_x[1500:], ydata = y_mean[1500:], p0=p0)
        m, t, b, d = params
        SA_fit = exp_new(np.arange(start=1500, stop=5000000, step=100), m, t, b, d)
        plt.plot(np.arange(start=1500, stop=5000000, step=100), exp_new(np.arange(start=1500, stop=5000000, step=100), m, t, b, d), '--', color="blue")


        ax.legend()
        plt.xlabel("Evaluations")
        plt.ylabel("Mean squared error")
        plt.ylim(500,1200)
        plt.title("Predicted Mean squared error of " + titles[i])

        plt.show()



            







