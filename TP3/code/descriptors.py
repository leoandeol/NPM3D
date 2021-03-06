#
#
#      0=============================0
#      |    TP4 Point Descriptors    |
#      0=============================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Script of the practical session
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

# Import library to plot in python
from matplotlib import pyplot as plt

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
#   Here you can define usefull functions to be used in the main
#

def local_PCA(points):
    assert points.shape[1] in [2,3]
    
    bary = np.mean(points,axis=0).reshape((1,-1))
    cov = (points-bary).T@(points-bary)/points.shape[0]

    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    return eigenvalues, eigenvectors


def neighborhood_PCA(query_points, cloud_points, radius):
    
    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points
    all_eigenvalues = np.zeros((query_points.shape[0], 3))
    all_eigenvectors = np.zeros((query_points.shape[0], 3, 3))
    
    for i,point in enumerate(tqdm(query_points)):
        point = point.reshape((1,-1))
        diff = cloud_points - point
        norm_diff = np.linalg.norm(diff,axis=1)
        pca_points = cloud_points[norm_diff<radius]
        eigvals, eigvecs = local_PCA(pca_points)
        all_eigenvalues[i] = eigvals
        all_eigenvectors[i] = eigvecs

    return all_eigenvalues, all_eigenvectors


def compute_features(query_points, cloud_points, radius):

    # Compute the features for all query points in the cloud
    eigvals, eigves = neighborhood_PCA(query_points, cloud_points, radius)

    lbd3, lbd2, lbd1 = eigvals[:,0], eigvals[:,1], eigvals[:,2]
    lbd1 = lbd1 + 1e-7
    normals = eigves[:,0,:]
    vert_unit = np.array([[0,0,1]])

    verticality = 2*np.arcsin(np.abs(normals@vert_unit.T))/np.pi
    linearity = -1*(lbd2/lbd1) + 1
    planarity = (lbd2 - lbd3) / lbd1
    sphericity = lbd3 / lbd1

    return verticality, linearity, planarity, sphericity


# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # PCA verification
    # ****************
    #

    if False:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        eigenvalues, eigenvectors = local_PCA(cloud)

        # Print your result
        print(eigenvalues)

        # Expected values :
        #
        #   [lambda_3; lambda_2; lambda_1] = [ 5.25050177 21.7893201  89.58924003]
        #
        #   (the convention is always lambda_1 >= lambda_2 >= lambda_3)
        #

    # Normal computation
    # ******************
    #

    if False:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        query = cloud[:,:]
        
        #np.random.shuffle(query)
        #query = query[:2000,:]

        # YOUR CODE        
        eigenvalues, eigenvectors = neighborhood_PCA(query,cloud,1)
        normals = eigenvectors[:,0,:]
        write_ply('../results/normals.ply', [query,normals], ['x', 'y', 'z', 'nx', 'ny', 'nz'])

    # Features computation
    # ********************
    #

    if True:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
        
        # YOUR CODE
        
        query = cloud[:,:]
        np.random.shuffle(query)
        query = query[:200,:]
        vert, lin, plan, spher = compute_features(cloud, cloud, 1)
        write_ply('../results/features.ply',[cloud,vert,lin,plan,spher],['x', 'y', 'z', 'vert', 'lin', 'plan', 'spher'])
