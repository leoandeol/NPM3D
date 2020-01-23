#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Second script of the practical session. Subsampling of a point cloud
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

# Import functions from scikit-learn
from sklearn.preprocessing import label_binarize

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply

# Import time package
import time


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#


def cloud_decimation(points, colors, labels, factor):
    

    # YOUR CODE
    decimated_points = points[::factor]
    decimated_colors = colors[::factor]
    decimated_labels = labels[::factor]

    return decimated_points, decimated_colors, decimated_labels


def grid_subsampling(points, voxel_size):

    #   Tips :
    #       > First compute voxel indices in each direction for all the points (can be negative).
    #       > Sum and count the points in each voxel (use a dictionaries with the indices as key).
    #         Remember arrays cannot be dictionary keys, but tuple can.
    #       > Divide the sum by the number of point in each cell.
    #       > Do not forget you need to return a numpy array and not a dictionary.
    #

    # YOUR CODE
    amin = np.amin(points,axis=0)
    amax = np.amax(points,axis=0)
    xmin = amin[0]
    ymin = amin[1]
    zmin = amin[2]
    xmax = amax[0]
    ymax = amax[1]
    zmax = amax[2]
    
    subsampled_points = np.zeros((np.ceil((xmax-xmin)/voxel_size).astype(np.int64),np.ceil((ymax-ymin)/voxel_size).astype(np.int64),np.ceil((zmax-zmin)/voxel_size).astype(np.int64),3))
    
    points_count = np.zeros((np.ceil((xmax-xmin)/voxel_size).astype(np.int64),np.ceil((ymax-ymin)/voxel_size).astype(np.int64),np.ceil((zmax-zmin)/voxel_size).astype(np.int64),1))
    
    points_idxs = ((points-amin.reshape((1,3)))//voxel_size).astype(np.int64)

    for i in range(points.shape[0]):
        x = points_idxs[i][0]
        y = points_idxs[i][1]
        z = points_idxs[i][2]
        points_count[x,y,z] += 1
        
        subsampled_points[x,y,z] += points[i]

    for i in range(subsampled_points.shape[0]):
        for j in range(subsampled_points.shape[1]):
            for k in range(subsampled_points.shape[2]):             
                if(points_count[i,j,k] != 0):
                    subsampled_points[i,j,k] = subsampled_points[i,j,k] / points_count[i,j,k]

    points_count = points_count.astype(np.int64)        

    return subsampled_points.reshape((-1,3))[points_count.reshape((-1))!=0]


def grid_subsampling_colors(points, colors, voxel_size):

    # YOUR CODE
    amin = np.amin(points,axis=0)
    amax = np.amax(points,axis=0)
    xmin = amin[0]
    ymin = amin[1]
    zmin = amin[2]
    xmax = amax[0]
    ymax = amax[1]
    zmax = amax[2]
    
    subsampled_points = np.zeros((np.ceil((xmax-xmin)/voxel_size).astype(np.int64),np.ceil((ymax-ymin)/voxel_size).astype(np.int64),np.ceil((zmax-zmin)/voxel_size).astype(np.int64),3))
    subsampled_colors = np.zeros((np.ceil((xmax-xmin)/voxel_size).astype(np.int64),np.ceil((ymax-ymin)/voxel_size).astype(np.int64),np.ceil((zmax-zmin)/voxel_size).astype(np.int64),3))
    
    points_count = np.zeros((np.ceil((xmax-xmin)/voxel_size).astype(np.int64),np.ceil((ymax-ymin)/voxel_size).astype(np.int64),np.ceil((zmax-zmin)/voxel_size).astype(np.int64),1))
    
    points_idxs = ((points-amin.reshape((1,3)))//voxel_size).astype(np.int64)

    for i in range(points.shape[0]):
        x = points_idxs[i][0]
        y = points_idxs[i][1]
        z = points_idxs[i][2]
        points_count[x,y,z] += 1
        

        subsampled_points[x,y,z] += points[i]
        subsampled_colors[x,y,z] += colors[i]

    for i in range(subsampled_points.shape[0]):
        for j in range(subsampled_points.shape[1]):
            for k in range(subsampled_points.shape[2]):             
                if(points_count[i,j,k] != 0):
                    subsampled_points[i,j,k] = subsampled_points[i,j,k] / points_count[i,j,k]
                    subsampled_colors[i,j,k] = subsampled_colors[i,j,k] / points_count[i,j,k]

    points_count = points_count.astype(np.int64)
    subsampled_colors = subsampled_colors.astype(np.uint8)
    
    return subsampled_points.reshape((-1,3))[points_count.reshape((-1))!=0],subsampled_colors.reshape((-1,3))[points_count.reshape((-1))!=0]




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
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    labels = data['label']    

    # Decimate the point cloud
    # ************************
    #

    # Define the decimation factor
    factor = 300

    # Decimate
    t0 = time.time()
    decimated_points, decimated_colors, decimated_labels = cloud_decimation(points, colors, labels, factor)
    t1 = time.time()
    print('decimation done in {:.3f} seconds'.format(t1 - t0))

    # Save
    write_ply('../results/decimated.ply', [decimated_points, decimated_colors, decimated_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])

    # Subsample the point cloud on a grid
    # ***********************************
    #

    # Define the size of the grid
    voxel_size = 0.2

    # Subsample
    t0 = time.time()
    subsampled_points = grid_subsampling(points, voxel_size)
    t1 = time.time()
    print('Subsampling done in {:.3f} seconds'.format(t1 - t0))

    write_ply('../results/grid_subsampled.ply', [subsampled_points], ['x', 'y', 'z'])

    t0 = time.time()
    subsampled__points, subsampled_colors = grid_subsampling_colors(points, colors, voxel_size)
    t1 = time.time()
    print('Colored subsampling done in {:.3f} seconds'.format(t1 - t0))

    # Save
    write_ply('../results/grid_subsampled_colors.ply', [subsampled_points,subsampled_colors], ['x', 'y', 'z', 'red', 'green', 'blue'])
    
    print('Done')
