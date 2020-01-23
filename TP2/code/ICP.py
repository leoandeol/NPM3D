#
#
#      0===================================0
#      |    TP2 Iterative Closest Point    |
#      0===================================0
#
#
#------------------------------------------------------------------------------------------
#
#      Script of the practical session
#
#------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 17/01/2018
#


#------------------------------------------------------------------------------------------
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
from utils.visu import show_ICP


#------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#

def RMS(X, Y):
    '''
    Computes the root mean square error between two point sets X and Y
    '''
    return np.sqrt(np.sum(np.linalg.norm(X-Y,axis=0)**2)/X.shape[1])


def best_rigid_transform(data, ref):
    '''
    Computes the least-squares best-fit transform that maps corresponding points data to ref.
    Inputs :
        data = (d x N) matrix where "N" is the number of points and "d" the dimension
         ref = (d x N) matrix where "N" is the number of points and "d" the dimension
    Returns :
           R = (d x d) rotation matrix
           T = (d x 1) translation vector
           Such that R * data + T is aligned on ref
    '''

    # YOUR CODE
    
    Pm = np.mean(ref,axis=1).reshape((-1,1))
    Pmp = np.mean(data,axis=1).reshape((-1,1))
    
    
    Q = ref - Pm
    Qp = data - Pmp
    
    H = Qp @ Q.T
    
    U,S,Vt = np.linalg.svd(H)
    
    R = Vt.T @ U.T
    
    T = Pm - R@Pmp

    return R, T


def icp_point_to_point(data, ref, max_iter, RMS_threshold):
    '''
    Iterative closest point algorithm with a point to point strategy.
    Inputs :
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
    Returns :
        data_aligned = data aligned on reference cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each data point in
        the ref cloud and this obtain a (1 x N_data) array of indices. This is the list of those
        arrays at each iteration
           
    '''

    # Variable for aligned data
    data_aligned = np.copy(data)

    # Initiate lists
    R_list = []
    T_list = []
    neighbors_list = []
    rms_list = [np.inf]
    tree = KDTree(ref.T,leaf_size=10)
    
    k = 0
    while k<max_iter and rms_list[-1] > RMS_threshold:
        _,neighbors = tree.query(data_aligned.T)
        neighbors = neighbors.ravel()
        #todo matching
        R,T = best_rigid_transform(data, ref[:,neighbors])
        data_aligned = R@data + T
        
        neighbors_list.append(neighbors)
        R_list.append(R)
        T_list.append(T)
        rms_list.append(RMS(ref[:,neighbors],data_aligned))
        k+=1
        
    return data_aligned, R_list, T_list, neighbors_list, rms_list[1:]


#------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
#
#   Here you can define the instructions that are called when you execute this file
#


if __name__ == '__main__':
   
    # Transformation estimation
    # *************************
    #

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        bunny_o_path = '../data/bunny_original.ply'
        bunny_r_path = '../data/bunny_returned.ply'

        # Load clouds

        # Find the best transformation

        # Apply the tranformation

        # Save cloud

        # Compute RMS

        # Print RMS
   

    # Test ICP and visualize
    # **********************
    #

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        ref2D_path = '../data/ref2D.ply'
        data2D_path = '../data/data2D.ply'

        # Load clouds
        ref2D_cloud = read_ply(ref2D_path)
        data2D_cloud = read_ply(data2D_path)

        ref2D_cloud = np.asarray([np.asarray(list(x)) for x in ref2D_cloud]).T
        data2D_cloud = np.asarray([np.asarray(list(x)) for x in data2D_cloud]).T

        # Apply ICP
        data_aligned, R_list, T_list, neighbors_list, rms_list = icp_point_to_point(data2D_cloud,
                                                                          ref2D_cloud,
                                                                          15,
                                                                          1e-7)
        
        # Show ICP
        show_ICP(data2D_cloud, ref2D_cloud, R_list, T_list, neighbors_list)

        plt.plot(rms_list)
        plt.xlabel('iteration')
        plt.ylabel('RMSE')
        plt.show()

    # If statement to skip this part if wanted
    if True:

        # Cloud paths
        bunny_o_path = '../data/bunny_original.ply'
        bunny_p_path = '../data/bunny_perturbed.ply'

        # Load clouds
        bunny_o_cloud = read_ply(bunny_o_path)
        bunny_p_cloud = read_ply(bunny_p_path)

        bunny_o_cloud = np.asarray([np.asarray(list(x)) for x in bunny_o_cloud]).T
        bunny_p_cloud = np.asarray([np.asarray(list(x)) for x in bunny_p_cloud]).T

        # Apply ICP
        data_aligned, R_list, T_list, neighbors_list, rms_list = icp_point_to_point(bunny_p_cloud,
                                                                          bunny_o_cloud,
                                                                          50,
                                                                          1e-7)
        
        # Show ICP
        plt.plot(rms_list)
        plt.xlabel('iteration')
        plt.ylabel('RMSE')
        plt.show()

        show_ICP(bunny_p_cloud, bunny_o_cloud, R_list, T_list, neighbors_list)
    # Fast ICP
    # ********
    #

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        NDDC_1_path = '../data/Notre_Dame_Des_Champs_1.ply'
        NDDC_2_path = '../data/Notre_Dame_Des_Champs_2.ply'

        # Load clouds
        NDDC_1_cloud = read_ply(NDDC_1_path)
        NDDC_2_cloud = read_ply(NDDC_2_path)

        # Apply fast ICP for different values of the sampling_limit parameter

        # Plot RMS
        #
        # => To plot something in python use the function plt.plot() to create the figure and 
        #    then plt.show() to display it
