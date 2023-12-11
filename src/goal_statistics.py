import os
import numpy as np
import matplotlib.pyplot as plt
from compute_goal_surface import ACCURACY_FILE


def show_statistics():
    if not os.path.exists(ACCURACY_FILE):
        print("Accuracy file does not exist.")
        raise FileNotFoundError
    with open(ACCURACY_FILE, "r") as file:
        lines = file.readlines()
        jaccard = []
        intersection_error = []
        for line in lines:
            line = line.split(",")
            jaccard.append(float(line[1]))
            intersection_error.append(float(line[2]))
        jaccard = np.array(jaccard)
        intersection_error = np.array(intersection_error)
        print("Jaccard index - Intersection Over Union (%): ", np.mean(jaccard), " +/- ", np.std(jaccard))
        print("Surface intersection error (%): ", np.mean(intersection_error), " +/- ", np.std(intersection_error))
        x = np.arange(max(len(jaccard), len(intersection_error)))
        plt.plot(x, jaccard, label='Jaccard index - Intersection Over Union')
        plt.plot(x, intersection_error, label='Intersection error')
        plt.xlabel('image number')
        plt.ylabel('Percentage (%)')
        plt.ylim(bottom=0)
        plt.legend()
        plt.show()
    return


if __name__ == "__main__" :
    show_statistics()