from tkinter import *
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import math


TITLE = "K-Means Visualization- Thomas Body - U1101544"

# Creates a list of colours
colours = ['r', 'g', 'c', 'y', 'b', 'm', 'k', 'w']

# Imports the dataset
df = pd.read_csv('haberman.csv')


class Interface:
    # This function contains all of the GUI code.
    def __init__(self, master):
        # Sets root title and min window size
        self.master = master
        master.title(TITLE)
        master.minsize(width=400, height=40)

        # Select number of clusters
        self.clusters_title = Label(master, text="Number of Clusters", width=20)
        self.clusters_title.grid(row=0, column=0)

        self.clusters_num = IntVar(master)
        self.clusters_num.set(2)
        self.clusters_ddlst = OptionMenu(master, self.clusters_num, *list(range(2, 7)))
        self.clusters_ddlst.grid(row=1, column=0, padx=5, pady=5, sticky=EW)

        # Select first attribute initialization method
        self.att_1_title = Label(master, text="X-axis", width=20)
        self.att_1_title.grid(row=0, column=1)

        self.att_1_lst = ['Age', 'year', 'Nodes']
        self.att_1 = StringVar(master)
        self.att_1.set('Age')
        self.att_1_ddlst = OptionMenu(master, self.att_1, *self.att_1_lst)
        self.att_1_ddlst.grid(row=1, column=1, padx=5, pady=5, sticky=EW)

        # Select first attribute 2 initialization method
        self.att_2_title = Label(master, text="Y-axis", width=20)
        self.att_2_title.grid(row=0, column=2)

        self.att_2_lst = ['Age', 'year', 'Nodes']
        self.att_2 = StringVar(master)
        self.att_2.set('Nodes')
        self.att_2_ddlst = OptionMenu(master, self.att_2, *self.att_2_lst)
        self.att_2_ddlst.grid(row=1, column=2, padx=5, pady=5, sticky=EW)

        # Initialize Button
        self.initialize_button = Button(master, text="Initialize", width=15,
                                        command=lambda: self.initialize(self.att_1.get(),
                                                                        self.att_2.get(),
                                                                        self.clusters_num.get(),
                                                                        self.points(self.att_1.get(),
                                                                                    self.att_2.get())))
        self.initialize_button.grid(row=1, column=3, padx=5, pady=5, sticky=EW)

        # Iterate button
        self.iterate_button = Button(master, text="Iterate", width=15,
                                     command=lambda: self.iterate(self.centroids,
                                                                  self.points_list))
        self.iterate_button.grid(row=1, column=4, padx=1, pady=5, sticky=EW)

        # Placeholder for drawing area and plot
        self.f = Figure(figsize=(10, 8), dpi=100)
        self.ax = None

        canvas = FigureCanvasTkAgg(self.f, master=master)
        canvas.draw()
        canvas.get_tk_widget().grid(row=2, column=0, columnspan=5)

        # Holds the centroids values
        self.centroids = []

        # Holds the set of points
        self.points_list = []

    # These functions used for the K-means algorithm
    # Sets the data points
    def points(self, att_1, att_2):
        points = np.array(list(zip(df[att_1], df[att_2])))
        self.points_list = points
        return points

    # Random centroid function. Returns separate x and y centroid values and array of both
    def random_centroids(self, att_1, att_2, clusters_num):
        centroids_x = np.random.randint(np.min(df[att_1]), np.max(df[att_1]), size=clusters_num)
        centroids_y = np.random.randint(np.min(df[att_2]), np.max(df[att_2]), size=clusters_num)
        centroids = np.array(list(zip(centroids_x, centroids_y)))
        return centroids_x, centroids_y, centroids

    # Plotting function
    def cluster_plot(self, points, min_indexes, centroids_x, centroids_y, clusters_num):

        if self.ax is None:
            # Classifies which point belongs to which cluster and colour codes it
            self.ax = self.f.add_subplot(111)
            for i in range(clusters_num):
                plot_points = np.array([points[j] for j in range(len(points)) if min_indexes[j] == i])
                self.ax.scatter(plot_points[:, 0], plot_points[:, 1], s=5, c=colours[i])

            # Plots the centroids
            self.ax.scatter(centroids_x, centroids_y, marker="x", s=50)
            self.ax.set_xlabel(self.att_1.get())
            self.ax.set_ylabel(self.att_2.get())
        else:
            # Clears the plot before re-plotting
            self.ax.cla()
            for i in range(clusters_num):
                plot_points = np.array([points[j] for j in range(len(points)) if min_indexes[j] == i])
                self.ax.scatter(plot_points[:, 0], plot_points[:, 1], s=5, c=colours[i])

            # Plots the centroids
            self.ax.scatter(centroids_x, centroids_y, marker="x", s=50)
            self.ax.set_xlabel(self.att_1.get())
            self.ax.set_ylabel(self.att_2.get())

        # Configures the drawing area
        canvas = FigureCanvasTkAgg(self.f, master=self.master)
        canvas.draw()
        canvas.get_tk_widget().grid(row=2, column=0, columnspan=6)

    # Euclidean distance calculator for vectors a and b
    def dist(self, a, b):
        distance = math.sqrt(sum([(x - y) ** 2 for x, y in zip(a, b)]))
        return distance

    # Assign each data point to the nearest centroid by euclidean distance
    # Returns the indexes of each points closest cluster
    def dist_from_centroids(self, centroids, points):
        distances = {}
        min_indexes = []

        # Finds the euclidean distance between each point and each cluster
        for i in range(len(centroids)):
            for j in range(len(points)):
                dist_from_clust = self.dist(centroids[i], points[j])
                if j in distances:
                    distances[j].append(dist_from_clust)
                else:
                    distances[j] = [dist_from_clust]

        # Finds which cluster each point is closest to
        for i in range(len(distances)):
            min_indexes.append(np.argmin(distances[i]))

        return min_indexes

    # Find the new centroid position by taking the average of all the points assigned to the cluster
    def calculate_centroids(self, min_indexes, clusters_num, points):
        centroids = []
        centroids_x = []
        centroids_y = []

        for i in range(clusters_num):
            plot_points_x = np.array([points[j][0] for j in range(len(points)) if min_indexes[j] == i])
            plot_points_y = np.array([points[j][1] for j in range(len(points)) if min_indexes[j] == i])

            centroids_x.append(np.mean(plot_points_x))
            centroids_y.append(np.mean(plot_points_y))
            centroids = np.array(list(zip(centroids_x, centroids_y)))

        centroids_x = np.array(centroids_x)
        centroids_y = np.array(centroids_y)
        self.centroids = centroids

        return centroids_x, centroids_y, centroids

    # The following functions are required for the GUI application.
    # Initializes/runs the first iteration
    def initialize(self, att_1, att_2, clusters_num, points):
        # Randomly assigns the first centroid
        centroids_x, centroids_y, centroids = self.random_centroids(att_1, att_2, clusters_num)

        # Assigns the class variable to centroids
        self.centroids = centroids

        # Calculates the distance from each point to the centroids and finds the closest centroid
        min_indexes = self.dist_from_centroids(centroids, points)

        # Plots the current clusters
        self.cluster_plot(points, min_indexes, centroids_x, centroids_y, clusters_num)

    # Iterates the k-means algorithm
    def iterate(self, centroids, points):
        # Recalculates the centroid positions
        centroids_x, centroids_y, centroids = self.calculate_centroids(self.dist_from_centroids(centroids, points),
                                                                       self.clusters_num.get(), points)

        # Recalculates the closest cluster
        min_indexes = self.dist_from_centroids(centroids, points)

        # Plots the current clusters
        self.cluster_plot(points, min_indexes, centroids_x, centroids_y, self.clusters_num.get())


# Runs the main root display. Part of the GUI application.
if __name__ == "__main__":
    root = Tk()
    interface = Interface(root)
    root.mainloop()
