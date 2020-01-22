#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Third script of the practical session. Neighborhoods in a point cloud
#
# ------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 13/12/2017
#


# ------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
import numpy as np
import matplotlib.pyplot as plt

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply

# Import time package
import time
from tqdm import tqdm

# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#


def brute_force_spherical(queries, supports, radius):
    # YOUR CODE

    neighborhoods = []
    for querie in tqdm(queries):
        neighborhood = []
        for point in tqdm(supports,leave=False):
            distance = np.linalg.norm(querie-point)
            if distance <= radius:
                neighborhood.append(point)
        neighborhoods.append(neighborhood)

    return neighborhoods


def brute_force_KNN(queries, supports, k):

    # YOUR CODE
    neighborhoods = []
    for querie in tqdm(queries):
        distances = []
        for point in tqdm(supports,leave=False):
            distance = np.linalg.norm(querie - point)
            distances.append(distance)

        distances = np.asarray(distances)
        sorted = np.argsort(distances)

        neighborhoods.append(supports[sorted][0:k])

    return neighborhoods





# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # Load point cloud
    # ****************
    #
    #   Load the file '../data/indoor_scan.ply'
    #   (See read_ply function)
    #

    # Path of the file
    file_path = '../data/indoor_scan.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T

    # Brute force neighborhoods
    # *************************
    #

    # If statement to skip this part if you want
    if False:

        # Define the search parameters
        neighbors_num = 1000
        radius = 0.2
        num_queries = 10

        # Pick random queries
        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]

        # Search spherical
        t0 = time.time()
        neighborhoods = brute_force_spherical(queries, points, radius)
        t1 = time.time()

        # Search KNN      
        neighborhoods = brute_force_KNN(queries, points, neighbors_num)
        t2 = time.time()

        # Print timing results
        print('{:d} spherical neighborhoods computed in {:.3f} seconds'.format(num_queries, t1 - t0))
        print('{:d} KNN computed in {:.3f} seconds'.format(num_queries, t2 - t1))

        # Time to compute all neighborhoods in the cloud
        total_spherical_time = points.shape[0] * (t1 - t0) / num_queries
        total_KNN_time = points.shape[0] * (t2 - t1) / num_queries
        print('Computing spherical neighborhoods on whole cloud : {:.0f} hours'.format(total_spherical_time / 3600))
        print('Computing KNN on whole cloud : {:.0f} hours'.format(total_KNN_time / 3600))



    # KDTree neighborhoods
    # ********************
    #

    # If statement to skip this part if wanted
    if True:

        # Define the search parameters
        num_queries = 1000

        # YOUR CODE
        radius = 0.2

        # Pick random queries
        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]

        if False:
            best_time = 100.0
            opt_leaf_size = 1
            for i in range(1,100):
                #TREE
                tree = KDTree(points, leaf_size=i)
                
                # Search spherical
                t0 = time.time()
                neighborhoods = tree.query_radius(queries, r=radius) 
                t1 = time.time()
                timedif = t1-t0
                if timedif < best_time:
                    print("timing improved from {:.3f} to {:.3f}, opt leaf size is now {}".format(best_time,timedif,i))
                    best_time = timedif
                    opt_leaf_size = i
            
            # Print timing results
            print('{:d} spherical neighborhoods with leaf size of {} computed in {:.3f} seconds'.format(num_queries,opt_leaf_size, best_time))

        if True:
            opt_leaf_size = 67
            tree = KDTree(points, leaf_size=opt_leaf_size)
            t0 = time.time()
            neighborhoods = tree.query_radius(queries, r=radius) 
            t1 = time.time()
            
        # Time to compute all neighborhoods in the cloud
        total_spherical_time = points.shape[0] * (t1 - t0) / num_queries
        print('Computing spherical neighborhoods with radius 20cm on whole cloud : {:.0f} seconds'.format(total_spherical_time))

        # Timings results on different radius with 1000 queries

        radius = [0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4]
        timings = []
        for rad in radius:
            random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
            queries = points[random_indices, :]
            t2 = time.time()
            neighborhoods = tree.query_radius(queries, r=rad)
            t3 = time.time()
            timings.append(t3-t2)
            
            print('{:d} spherical neighborhoods with radius {}cm computed in {:.3f} seconds'.format(num_queries,rad, t3 - t2))

        plt.figure()
        plt.plot(radius,timings)
        plt.xlabel("radius")
        plt.ylabel("timings")
        plt.show()
