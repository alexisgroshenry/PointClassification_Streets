#
#
#      0=============================0
#      |    TP4 Point Descriptors    |
#      0=============================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Script of the practical session
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 13/12/2017
#


# ----------------------------------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#

 
# Import numpy package and name it "np"
import numpy as np

# Import functions from scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KDTree
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier
# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

from descriptors import compute_local_PCA, compute_local_PCA_2D
from os import listdir
from os.path import exists, join


# ----------------------------------------------------------------------------------------------------------------------
#
#           Feature Extractor Class
#       \*****************************/
#
#
#   Here you can define useful functions to be used in the main
#

class FeaturesExtractor:
    """
    Class that computes features from point clouds
    """

    # Initiation methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self):
        """
        Initiation method called when an object of this class is created. This is where you can define parameters
        """
        self.num_scales = 9
        self.num_features = 18 * self.num_scales
        # Neighborhood construction
        self.method_tree = 'knn'
        self.radius = 0.5
        self.k = 50

        # Number of training points per class
        self.num_per_class = 5000

        # Classification labels
        self.label_names = {0: 'Unclassified',
                            1: 'Ground',
                            2: 'Building',
                            3: 'Poles',
                            4: 'Pedestrians',
                            5: 'Cars',
                            6: 'Vegetation'}

    # Methods
    # ------------------------------------------------------------------------------------------------------------------

    def compute_features(self, query_points, cloud_points):
        '''
        Compute all features at all scale
        '''
        tot_features = np.empty((query_points.shape[0], 0))
        for _ in range(self.num_scales):
            # Compute features for the points of the chosen indices and place them in a [N, 4] matrix
            features = self.compute_features_scale(query_points, cloud_points).T
            tot_features = np.hstack((tot_features, features))
            # downsample the point cloud by a factor 2
            cloud_points = cloud_points[::2]

        return tot_features


    def compute_features_scale(self, query_points, cloud_points):
        '''
        Compute the features at a fixed scale
        '''
        all_eigenvalues, all_eigenvectors = compute_local_PCA(query_points, cloud_points, self.radius, self.k, self.method_tree)
        all_eigenvalues[:,2] += 1e-6 # add epsilon
        normals = all_eigenvectors[:, :, 0]

        # geometric 3D properties
        tree = KDTree(cloud_points, leaf_size=20)
        dists, idxs = tree.query(query_points, k=self.k)
        neighborhood_3D_heights = cloud_points[idxs][:,:,2]

        # radius encapsulating the kNN
        radius_3D = dists[:,-1]
        # maximum amplitude of heights within the neighborhood
        delta_z = np.max(neighborhood_3D_heights, axis=1) - np.min(neighborhood_3D_heights,axis=1)
        # standard deviation of heights in the neighborhood
        sd_z = np.std(neighborhood_3D_heights, axis=1)
        # local point density
        density_3D = (self.k+1) / ((4/3)*np.pi*radius_3D**3)
        z = query_points[:,2]


        # local 3D shape features
        verticality = 2 * np.arcsin( np.abs(normals[:,2]) )
        linearity = 1 - all_eigenvalues[:,1] / all_eigenvalues[:,2]
        planarity = (all_eigenvalues[:,1] - all_eigenvalues[:,0]) / all_eigenvalues[:,2]
        sphericity = all_eigenvalues[:,0] / all_eigenvalues[:,2]
        omnivariance = np.cbrt(np.prod(all_eigenvalues, axis=1))
        anisotropy = (all_eigenvalues[:,2] - all_eigenvalues[:,0]) / all_eigenvalues[:,2]
        eigenentropy = - np.sum(all_eigenvalues*np.log(np.abs(all_eigenvalues)+1e-6), axis=1)
        sum_3D = np.sum(all_eigenvalues, axis=1)
        change_curvature = all_eigenvalues[:,0] / sum_3D

        # geometric 2D properties
        # radius encapsulating the k NN in 2D (horizontal plane)
        radius_2D = np.max(np.sqrt(np.sum((np.expand_dims(query_points, axis=1)[:,:,:2] - cloud_points[idxs][:,:,:2])**2,axis=2)),axis=1)
        density_2D = (self.k+1) / (np.pi*radius_2D**2)

        # local 2D shape features
        all_eigenvalues_2D, _ = compute_local_PCA_2D(query_points, cloud_points, self.radius, self.k, self.method_tree)
        sum_2D = np.sum(all_eigenvalues_2D, axis=1)
        ratio_2D = all_eigenvalues_2D[:,0] / (all_eigenvalues_2D[:,1]+1e-6)

        features = np.vstack((
            z.ravel(),
            radius_3D.ravel(),
            radius_2D.ravel(),
            delta_z.ravel(),
            sd_z.ravel(),
            density_3D.ravel(),
            density_2D.ravel(),
            verticality.ravel(),
            linearity.ravel(),
            planarity.ravel(),
            sphericity.ravel(),
            omnivariance.ravel(),
            anisotropy.ravel(),
            eigenentropy.ravel(),
            sum_3D.ravel(),
            sum_2D.ravel(),
            change_curvature.ravel(),
            ratio_2D.ravel()
        ))
        return features


    def extract_training(self, path):
        '''
        This method extract features/labels of a subset of the training points. It ensures a balanced choice between
        classes.
        :param path: path where the ply files are located.
        :return: features and labels
        '''
        # Get all the ply files in data folder
        ply_files = [f for f in listdir(path) if f.endswith('.ply')]

        # Initiate arrays
        training_features = np.empty((0, self.num_features))
        training_labels = np.empty((0,))

        val_features = np.empty((0, self.num_features))
        val_labels = np.empty((0,))

        # Loop over each training cloud
        for _, file in enumerate(ply_files):
            
            # Load Training cloud
            cloud_ply = read_ply(join(path, file))
            points = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
            labels = cloud_ply['class']

            # Initiate training indices array
            training_inds = np.empty(0, dtype=np.int32)
            val_inds = np.empty(0, dtype=np.int32)

            # Loop over each class to choose training points
            for label, name in self.label_names.items():

                # Do not include class 0 in training
                if label == 0:
                    continue

                # Collect all indices of the current class
                label_inds = np.where(labels == label)[0]

                # If you have not enough indices, just take all of them
                if len(label_inds) <= self.num_per_class:
                    print('Not enough points for class {}, selecting {} points instead of {}'.format(name, len(label_inds), self.num_per_class))
                    split = int(.8*len(label_inds))
                    training_inds = np.hstack((training_inds, label_inds[:split]))
                    val_inds = np.hstack((val_inds, label_inds[split:]))

                # If you have more than enough indices, choose randomly
                else:
                    random_choice = np.random.choice(len(label_inds), self.num_per_class, replace=False)
                    split = int(.8*self.num_per_class)
                    training_inds = np.hstack((training_inds, label_inds[random_choice[:split]]))
                    val_inds = np.hstack((val_inds, label_inds[random_choice[split:]]))

            # Gather chosen points
            training_points = points[training_inds, :]
            val_points = points[val_inds, :]

            # Compute features for the points of the chosen indices and place them in a [N, 4] matrix
            features = self.compute_features(training_points, points)
            # Concatenate features / labels of all clouds
            training_features = np.vstack((training_features, features))
            training_labels = np.hstack((training_labels, labels[training_inds]))

            features = self.compute_features(val_points, points)
            val_features = np.vstack((val_features, features))
            val_labels = np.hstack((val_labels, labels[val_inds]))

        return training_features, training_labels, val_features, val_labels


    def extract_test(self, path):
        '''
        This method extract features of all the test points.
        :param path: path where the ply files are located.
        :return: features
        '''
        # Get all the ply files in data folder
        ply_files = [f for f in listdir(path) if f.endswith('.ply')]

        # Initiate arrays
        test_features = np.empty((0, self.num_features))

        # Loop over each training cloud
        for _, file in enumerate(ply_files):

            # Load Training cloud
            cloud_ply = read_ply(join(path, file))
            points = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

            # Compute features only one time and save them for further use
            #
            #   WARNING : This will save you some time but do not forget to delete your features file if you change
            #             your features. Otherwise you will not compute them and use the previous ones
            #

            # Name the feature file after the ply file.
            feature_file = file[:-4] + '_features.npy'
            feature_file = join(path, feature_file)

            # If the file exists load the previously computed features
            if exists(feature_file):
                features = np.load(feature_file)

            # If the file does not exist, compute the features (very long) and save them for future use
            else:

                features = self.compute_features(points, points)
                np.save(feature_file, features)

            # Concatenate features of several clouds
            # (For this minichallenge this is useless as the test set contains only one cloud)
            test_features = np.vstack((test_features, features))

        return test_features


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
#
#   Here you can define the instructions that are called when you execute this file
#


if __name__ == '__main__':

    # Parameters
    # **********
    #
    TEST = True
    SELECT_FEATURES = False
    # Path of the training and test files
    training_path = '../data/MiniChallenge/training'
    test_path = '../data/MiniChallenge/test'

    # Collect training features / labels
    # **********************************
    #
    #   For this simple algorithm, we only compute the features for a subset of the training points. We choose N points
    #   per class in each training file. This has two advantages : balancing the class for our classifier and saving a
    #   lot of computational time.
    #

    print('Collect Training Features')
    t0 = time.time()

    # Create a feature extractor
    f_extractor = FeaturesExtractor()

    # Collect training features and labels
    training_features, training_labels, val_features, val_labels = f_extractor.extract_training(training_path)
    sample_weights = compute_sample_weight(
        class_weight='balanced',
        y=training_labels
    )

    t1 = time.time()
    print('Done in %.3fs\n' % (t1 - t0))

    # Train a random forest classifier
    # ********************************
    #
    t0 = time.time()

    # Create and train a random forest with scikit-learn
    # print('Training Random Forest')
    # clf = RandomForestClassifier(class_weight='balanced')

    # Create and train a xgboost classifier
    print('Training XGBoost')
    clf = XGBClassifier()
    clf.fit(training_features, training_labels, sample_weight=sample_weights)

    t1 = time.time()
    print('Done in %.3fs\n' % (t1 - t0))


    # Feature Selection
    # *****************
    #
    if SELECT_FEATURES:
        t0 = time.time()
        print('Selecting 50 best features')
        sfm = SelectFromModel(clf, threshold=-np.inf, max_features=50)

        # Train the selector
        sfm.fit(training_features, training_labels)
        # Keep only the relevant features in the data
        training_features = sfm.transform(training_features)
        val_features = sfm.transform(val_features)
        t1 = time.time()
        print('Done in %.3fs\n' % (t1 - t0))

        # Train a random forest classifier on the best features
        # *****************************************************
        #
        t0 = time.time()

        # Create and train a random forest with scikit-learn
        # print('Training Random Forest')
        # clf = RandomForestClassifier(class_weight='balanced')

        # Create and train a xgboost classifier
        print('Training XGBoost')
        clf = XGBClassifier()
        clf.fit(training_features, training_labels, sample_weight=sample_weights)

        t1 = time.time()
        print('Done in %.3fs\n' % (t1 - t0))

        # Performance on validation set
        # ***************************
        #
        print(classification_report(val_labels, clf.predict(val_features)))


    # Test
    # ****
    #
    if TEST:
        print('Compute testing features')
        t0 = time.time()

        # Collect test features
        test_features = f_extractor.extract_test(test_path)
        if SELECT_FEATURES:
            test_features = sfm.transform(test_features)

        t1 = time.time()
        print('Done in %.3fs\n' % (t1 - t0))

        print('Test')
        t0 = time.time()

        # Test the random forest on our features
        predictions = clf.predict(test_features)

        t1 = time.time()
        print('Done in %.3fs\n' % (t1 - t0))

        # Save prediction for submission
        # ******************************
        #

        print('Save predictions')
        t0 = time.time()
        np.savetxt('../preds/MiniDijon9.txt', predictions, fmt='%d')
        t1 = time.time()
        print('Done in %.3fs\n' % (t1 - t0))

        # Save clouds with predictions for results visualization
        # ******************************************************
        #
        print('Save predicted clouds')
        t0 = time.time()
        ply_files = [f for f in listdir(test_path) if f.endswith('.ply')]
        for file in ply_files:
            cloud_ply = read_ply(join(test_path, file))
            cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T
            write_ply(
                join('../preds', file),
                (cloud, predictions),
                ['x', 'y', 'z', 'predictions']
            )
        t1 = time.time()
        print('Done in %.3fs\n' % (t1 - t0))


