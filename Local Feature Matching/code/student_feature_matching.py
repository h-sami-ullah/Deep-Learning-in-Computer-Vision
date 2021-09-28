import numpy as np


def match_features(features1, features2, x1, y1, x2, y2):
    """
    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).

    To start with, simply implement the "ratio test", equation 4.18 in
    section 4.1.3 of Szeliski. There are a lot of repetitive features in
    these images, and all of their descriptors will look similar. The
    ratio test helps us resolve this issue (also see Figure 11 of David
    Lowe's IJCV paper).

    For extra credit you can implement various forms of spatial/geometric
    verification of matches, e.g. using the x and y locations of the features.

    Args:
    -   features1: A numpy array of shape (n,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
    -   features2: A numpy array of shape (m,feat_dim) representing a second set
            features (m not necessarily equal to n)
    -   x1: A numpy array of shape (n,) containing the x-locations of features1
    -   y1: A numpy array of shape (n,) containing the y-locations of features1
    -   x2: A numpy array of shape (m,) containing the x-locations of features2
    -   y2: A numpy array of shape (m,) containing the y-locations of features2

    Returns:
    -   matches: A numpy array of shape (k,2), where k is the number of matches.
            The first column is an index in features1, and the second column is
            an index in features2
    -   confidences: A numpy array of shape (k,) with the real valued confidence for
            every match

    'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
    """
    #############################################################################
    # TODO: YOUR CODE HERE                                                        #
    #############################################################################

        # this is just an arbitrarily small place holder to avoid division by zero
    nonzero = 0.000001

    confidences = []
    matches = []
    


    # go through features1
    for i, feature1 in enumerate(features1):

        # compute |feature1| that is, the euclidean distance to feature1
        feature1_norm = np.linalg.norm(feature1) #cacluate L2 Norm
        feature1 = feature1 / (feature1_norm + nonzero) #Normalization of feature1

        thisMatches = []


        # go through features2
        for j, feature2 in enumerate(features2):
            # compute |feature2| that is, the euclidean distance to feature2
            feature2_norm = np.linalg.norm(feature2) #cacluate L2 Norm
            feature2 = feature2 / (feature2_norm + nonzero)#Normalization of feature2

            # compute distance between feature1 and feature2
            thisDistance = np.sum(abs(feature1 - feature2))#  distance between each feature

            # add this distance
            thisMatch = [i, j, thisDistance]# adding point and distance
            thisMatches.append(thisMatch)

        # sort matches by their distances in descending order
        thisMatches = np.asarray(thisMatches) # converting to numpy array
        thisMatches = thisMatches[thisMatches[:, 2].argsort()] # sorting in ascending order
         
        # reject all matches with the distance ratio greater than 0.8
        if thisMatches[0, 2] < 0.80 * thisMatches[1, 2]: # select first 2 points and find the ratio of the distance
            confidences.append(thisMatches[0, 2])# if the distacne ratio is less than 0.8 consider the point
            matches.append(thisMatches[0, 0:2]) # if the distacne ratio is less than 0.8 consider the point
            

    # convert from lsit to numpy array
    matches = np.asarray(matches) 
    confidences = np.asarray(confidences)

    # add confidences to matches to sort them together
    confidences = np.expand_dims(confidences, axis=1) # changing the shape to make it a column vector
    matchAndConfidences = np.concatenate((matches, confidences), axis=1) # concatenate the matches and confidence on axis=1 

    # sort the two arrays together by distance ratio
    matchAndConfidences = matchAndConfidences[matchAndConfidences[:, 2].argsort()]

    # separate the two arrays after sorting together
    matches = matchAndConfidences[:, 0:2].astype(int)
    confidences = matchAndConfidences[:, 2]

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return matches, confidences
