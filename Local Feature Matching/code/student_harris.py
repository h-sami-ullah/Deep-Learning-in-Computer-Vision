import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_interest_points(img, feature_width):
    """
    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful in this function in order to (a) suppress boundary interest
    points (where a feature wouldn't fit entirely in the image, anyway)
    or (b) scale the image filters being used. Or you can ignore it.

    By default you do not need to make scale and orientation invariant
    local features.

    The lecture slides and textbook are a bit vague on how to do the
    non-maximum suppression once you've thresholded the cornerness score.
    You are free to experiment. For example, you could compute connected
    components and take the maximum value within each component.
    Alternatively, you could run a max() operator on each sliding window. You
    could use this to ensure that every interest point is at a local maximum
    of cornerness.

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   feature_width: integer representing the local feature width in pixels.

    Returns:
    -   x: A numpy array of shape (N,) containing x-coordinates of interest points
    -   y: A numpy array of shape (N,) containing y-coordinates of interest points
    -   confidences (optional): numpy nd-array of dim (N,) containing the strength
            of each interest point
    -   scales (optional): A numpy array of shape (N,) containing the scale at each
            interest point
    -   orientations (optional): A numpy array of shape (N,) containing the orientation
            at each interest point
    """
    confidences, scales, orientations = None, None, None
    #############################################################################
    # TODO: YOUR HARRIS CORNER DETECTOR CODE HERE                                                      #
    #############################################################################
    alpha = 0.04 # tuning parameter for later stage
   

    m, n = img.shape #finding the shape of image (image is grayscale)

    # finding X and Y derivative using Sobel 
    Iy = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)     # changes in Y direction
    Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3) # changes in X direction


    # computing auto-correaltion matrix entries, REMEMBER this matrix is symetric around off-diagonal enteries, 
    # It means I_XY =I_YX
    
    
    Iyy = Iy * Iy
    Ixx = Ix * Ix
    Ixy = Ix * Iy
    

    # convolve each entry with a guassian blurr
    gaussian_kernel = cv2.getGaussianKernel(ksize=5, sigma=1)# gaussain kernel of size 5
    Iyy = cv2.filter2D(Iyy, cv2.CV_64F, gaussian_kernel)
    Ixy = cv2.filter2D(Ixy, cv2.CV_64F, gaussian_kernel)
    Ixx = cv2.filter2D(Ixx, cv2.CV_64F, gaussian_kernel)
    
    

    # compute R strength which is nothing but the det(A)-alpha* trace(A) where alpha is choosen to be 0.04
    R_strength = (Iyy*Ixx-Ixy**2) - alpha*(Iyy+Ixx)**2 

    # threshold R with 2% of maximum value in R and disgarding those value 
    
    R_strength[R_strength<0.02*np.amax(R_strength)] = 0
    R_rect = np.copy(R_strength)
        
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO: YOUR ADAPTIVE NON-MAXIMAL SUPPRESSION CODE HERE                     #
    # While most feature detectors simply look for local maxima in              #
    # the interest function, this can lead to an uneven distribution            #
    # of feature points across the image, e.g., points will be denser           #
    # in regions of higher contrast. To mitigate this problem, Brown,           #
    # Szeliski, and Winder (2005) only detect features that are both            #
    # local maxima and whose response value is significantly (10%)              #
    # greater than that of all of its neighbors within a radius r. The          #
    # goal is to retain only those points that are a maximum in a               #
    # neighborhood of radius r pixels. One way to do so is to sort all          #
    # points by the response strength, from large to small response.            #
    # The first entry in the list is the global maximum, which is not           #
    # suppressed at any radius. Then, we can iterate through the list           #
    # and compute the distance to each interest point ahead of it in            #
    # the list (these are pixels with even greater response strength).          #
    # The minimum of distances to a keypoint's stronger neighbors               #
    # (multiplying these neighbors by >=1.1 to add robustness) is the           #
    # radius within which the current point is a local maximum. We              #
    # call this the suppression radius of this interest point, and we           #
    # save these suppression radii. Finally, we sort the suppression            #
    # radii from large to small, and return the n keypoints                     #
    # associated with the top n suppression radii, in this sorted               #
    # orderself. Feel free to experiment with n, we used n=1500.                #
    #                                                                           #
    # See:                                                                      #
    # https://www.microsoft.com/en-us/research/wp-content/uploads/2005/06/cvpr05.pdf
    # or                                                                        #
    # https://www.cs.ucsb.edu/~holl/pubs/Gauglitz-2011-ICIP.pdf                 #
    #############################################################################

    # construct 2d-array "R" filled with arrays containing x, y, r-value
    R = []
    m, n = R_rect.shape 
    Rmax = np.amax(R_rect)#finding maximum of feature vector
    for y in range(0, m):
        for x in range(0, n):
            if not R_rect[y, x] == 0:# ignoring the zero values
                R.append([int(x), int(y), 255 * R_rect[y, x] / Rmax]) #list R containing x,y, position with normalize R 
                                                                          #feature

    # sort R by R_rect in descending order
    R = np.asarray(R)# Converting list to numpy array
    R = R[R[:, 2].argsort()] #indirect sorting based on feature values
    R = np.flipud(R) #upside down flippling of R

    numPts = 2000
    radiis = []

    # go through R_rect
    for i, point in enumerate(R): # i is index and point has point with x,y coordinates, and faeture value  
        if i == 0:
            radius = 9999999.0  # arbitrary large number 
            thisPtSet = (point[0], point[1], radius)
            radiis.append(thisPtSet)
            continue

        # these are key R feature vectors to be compared with "point"
        otherPoints = R[0:i, :] # from zero feature vector to i-1 feature vector

        # compute the distance between the current point and all other point
        distances = np.sqrt(((otherPoints[:, 0] - point[0]) ** 2 + (otherPoints[:, 1] - point[1]) ** 2)) 
        # the above distance is eucilidean distance 
        # get the minimum distance
        
        min_dist = np.min(distances) # above all the distance find points with smallest distance

        # get the index of the keypoint with min radius
        thisPtSet = (point[0], point[1], min_dist)
    
        radiis.append(thisPtSet)

    # sort radiis in descending order
    radiis = np.asarray(radiis)#converting to numpy array
    radiis = radiis[radiis[:, 2].argsort()]#indirect sorting based on last column 
    radiis = np.flipud(radiis)#upside down flippling of raddiis

    #print(radiis.shape)
    radiis = radiis[0:numPts, :] #Selected first 2000 points

    y = radiis[:, 1]
    x = radiis[:, 0]
    

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return x,y, confidences, scales, orientations


