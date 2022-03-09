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
from tabnanny import verbose
from tkinter import VERTICAL
import numpy as np

# Import library to plot in python
from matplotlib import pyplot as plt

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#



def PCA(points):
    points -= np.mean(points, axis=0, keepdims=True)
    cov = points.T @ points / points.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    return eigenvalues, eigenvectors



def compute_local_PCA(query_points, cloud_points, radius=0.5, k=30, method='radius'):

    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points

    tree = KDTree(cloud_points, leaf_size=20)
    if method == 'radius':
        idxs = tree.query_radius(query_points, radius)
    elif method == 'knn':
        _, idxs = tree.query(query_points, k=k)

    all_eigenvalues = np.zeros((query_points.shape[0], 3))
    all_eigenvectors = np.zeros((query_points.shape[0], 3, 3))

    for i, _ in enumerate(query_points):
        all_eigenvalues[i], all_eigenvectors[i] = PCA(cloud_points[idxs[i]])
    
    return all_eigenvalues, all_eigenvectors

def compute_local_PCA_2D(query_points, cloud_points, radius=0.5, k=30, method='radius'):

    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points

    tree = KDTree(cloud_points, leaf_size=20)
    if method == 'radius':
        idxs = tree.query_radius(query_points, radius)
    elif method == 'knn':
        _, idxs = tree.query(query_points, k=k)

    all_eigenvalues = np.zeros((query_points.shape[0], 2))
    all_eigenvectors = np.zeros((query_points.shape[0], 2, 2))

    for i, _ in enumerate(query_points):
        all_eigenvalues[i], all_eigenvectors[i] = PCA(cloud_points[idxs[i]][:,:2])
    
    return all_eigenvalues, all_eigenvectors


def compute_features(query_points, cloud_points, radius):

    all_eigenvalues, all_eigenvectors = compute_local_PCA(query_points, cloud_points, radius)
    all_eigenvalues[:,2] += 1e-6 # add epsilon
    normals = all_eigenvectors[:, :, 0]

    verticality = 2 * np.arcsin( np.abs(normals[:,2]) )
    linearity = 1 - all_eigenvalues[:,1] / all_eigenvalues[:,2]
    planarity = (all_eigenvalues[:,1] - all_eigenvalues[:,0]) / all_eigenvalues[:,2]
    sphericity = all_eigenvalues[:,0] / all_eigenvalues[:,2]

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
    if False:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        eigenvalues, eigenvectors = PCA(cloud)

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
    if True:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        all_eigenvalues, all_eigenvectors = compute_local_PCA(cloud, cloud, 0.50, method='radius')
        normals = all_eigenvectors[:, :, 0]

        # Save cloud with normals
        write_ply('../Lille_street_small_normals.ply', (cloud, normals), ['x', 'y', 'z', 'nx', 'ny', 'nz'])
		
    if False:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute features
        verticality, linearity, planarity, sphericity = compute_features(cloud, cloud, 0.50)

        # Save cloud with normals
        write_ply(
            '../Lille_street_small_normals_features.ply',
            (cloud, verticality, linearity, planarity, sphericity),
            ['x', 'y', 'z', 'verticality', 'linearity', 'planarity', 'sphericity']
            )
		
