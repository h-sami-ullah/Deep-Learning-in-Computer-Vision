import numpy as np
from numpy import *
def calculate_projection_matrix(points_2d, points_3d):
    """
    To solve for the projection matrix. You need to set up a system of
    equations using the corresponding 2D and 3D points:
                                                      [ M11      [ u1
                                                        M12        v1
                                                        M13        .
                                                        M14        .
    [ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1        M21        .
      0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1        M22        .
      .  .  .  . .  .  .  .    .     .      .       *   M23   =    .
      Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn        M24        .
      0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn ]      M31        .
                                                        M32        un
                                                        M33 ]      vn ]
    Then you can solve this using least squares with np.linalg.lstsq() or SVD.
    Notice you obtain 2 equations for each corresponding 2D and 3D point
    pair. To solve this, you need at least 6 point pairs.
    Args:
    -   points_2d: A numpy array of shape (N, 2)
    -   points_3d: A numpy array of shape (N, 3)
    Returns:
    -   M: A numpy array of shape (3, 4) representing the projection matrix
    """


    M = np.asarray([[0.1768, 0.7018, 0.7948, 0.4613],
                    [0.6750, 0.3152, 0.1136, 0.0480],
                    [0.1020, 0.1725, 0.7244, 0.9932]])
    # Placeholder M matrix. It leads to a high residual. Your total residual
    # should be less than 1.
    ###########################################################################
    ###########################################################################
    
    A=np.zeros((int(2*points_3d.shape[0]),12))# Intializing A matrix with zeros 
    row=0    #intialize row
    for i in range(0,points_3d.shape[0]):#run for all points
        
        # The below code compute the values in A matrix
        
        A[row,0:3]=points_3d[i,:] # Updating X1 Y1 Z1 
        A[row,3]=1 # updating A(row,3) 4th column of A which has value 1
        A[row,8:11]=-points_2d[i,0]*points_3d[i,:] # -ui*Xi -ui*Yi -ui*Zi 
        A[row,11]=-points_2d[i,0]
        A[row+1,4:7]=points_3d[i,:] # Updating Xi Yi Zi
        A[row+1,7]=1# updating A(row+1,3) 8th column of A which has value 1
        A[row+1,8:11]=-points_2d[i,1]*points_3d[i,:]#-vi*Xi -vi*Yi -vi*Zi
        A[row+1,11]=-points_2d[i,1]
        row=row+2
    U,S,VT=np.linalg.svd(A) # Singular Value decomposition
    V=VT.T #Transpose of VT to get V
    M_column=V[:,V.shape[1]-1]# A column matrix of M
    
    M=np.zeros((3,4))# Intializing M (3,4) = 0
    M[0,:]=M_column[:4]# First four value in first row
    M[1,:]=M_column[4:8]# Next(4-8,4 inclusive and 8 exclusive) four value in 2nd row
    M[2,:]=M_column[8:12]# Next (8-12,8 inclusive and 12 exclusive) four value in third row
    
    ###########################################################################
    ###########################################################################

    return M

def calculate_camera_center(M):
    """
    Returns the camera center matrix for a given projection matrix.
    The center of the camera C can be found by:
        C = -Q^(-1)m4
    where your project matrix M = (Q | m4).
    Args:
    -   M: A numpy array of shape (3, 4) representing the projection matrix
    Returns:
    -   cc: A numpy array of shape (1, 3) representing the camera center
            location in world coordinates
    """

    # Placeholder camera center. In the visualization, you will see this camera
    # location is clearly incorrect, placing it in the center of the room where
    # it would not see all of the points.
    #cc = np.asarray([1, 1, 1])
    ###########################################################################
    ###########################################################################
    Q=M[0:3,0:3] # Q has first three column of M
    Q_inv=np.linalg.inv(Q)# Inverse of Q
    center=np.matmul(-Q_inv,M[:,3])# Inv(Q)*M(:,3) (Multiplication of 4th column of M with Q inverse.)
    ###########################################################################
    ###########################################################################
    return center # Center of camera

def estimate_fundamental_matrix(points_a, points_b): 
    """
    Calculates the fundamental matrix. Try to implement this function as
    efficiently as possible. It will be called repeatedly in part 3.
    You must normalize your coordinates through linear transformations as
    described on the project webpage before you compute the fundamental
    matrix.
    Args:
    -   points_a: A numpy array of shape (N, 2) representing the 2D points in
                  image A
    -   points_b: A numpy array of shape (N, 2) representing the 2D points in
                  image B
    Returns:
    -   F: A numpy array of shape (3, 3) representing the fundamental matrix
    """
    # Placeholder fundamental matrix
    #F = np.asarray([[0, 0, -0.0004],
     #               [0, 0, 0.0032],
      #              [0, -0.0044, 0.1034]])
    
    
    ###########################################################################
    ###########################################################################
    
    arr_a = np.column_stack((points_a, [1]*points_a.shape[0])) #adding an identity column
    arr_b = np.column_stack((points_b, [1]*points_b.shape[0])) #adding an identity column

    arr_a = np.tile(arr_a, 3) # Construct an array by repeating arr_a three times
    arr_b = arr_b.repeat(3, axis=1) # Repeat column elements of arr_b three times 
    A = np.multiply(arr_a, arr_b) # Element-wise multiplication 

    '''Solve f from Af=0'''
   
    U, s, V = np.linalg.svd(A) # singular value decomposition
    F_matrix = V[-1]
    F_matrix = np.reshape(F_matrix, (3, 3)) 

    '''Resolve det(F) = 0 constraint using SVD'''
    U, S, Vh = np.linalg.svd(F_matrix)
    S[-1] = 0
    F_matrix = U @ np.diagflat(S) @ Vh
    
    ###########################################################################
    ###########################################################################

    return F_matrix
   
    
    

def estimate_fundamental_matrix_with_normalize(points_a, points_b):
    # Try to implement this function as efficiently as possible. It will be
    # called repeatly for part III of the project
    #

    #                                              [f11
    # [u1u1' v1u1' u1' u1v1' v1v1' v1' u1 v1 1      f12     [0
    #  u2u2' v2v2' u2' u2v2' v2v2' v2' u2 v2 1      f13      0
    #  ...                                      *   ...  =  ...
    #  ...                                          ...     ...
    #  unun' vnun' un' unvn' vnvn' vn' un vn 1]     f32      0]
    #                                               f33]
    
    
    ###########################################################################
    ###########################################################################

    
    mean_a = points_a.mean(axis=0) # Calculating mean of points_a
    mean_b = points_b.mean(axis=0) # Calculating mean of points_b
    
    std_a = np.sqrt(np.mean(np.sum((points_a-mean_a)**2, axis=1), axis=0)) #  Calculating Stadard deviation of points_a
    std_b = np.sqrt(np.mean(np.sum((points_b-mean_b)**2, axis=1), axis=0)) #  Calculating Stadard deviation of points_b

    Ta1 = np.diagflat(np.array([np.sqrt(2)/std_a, np.sqrt(2)/std_a, 1]))  #  Create a two-dimensional array with the 
                                                                          #  Flattened input as a diagonal
  
    # Return a 2-D array with ones on the diagonal and zeros elsewhere.
    Ta2 = np.column_stack((np.row_stack((np.eye(2), [[0, 0]])), [-mean_a[0], -mean_a[1], 1])) 

    #  Create a two-dimensional array with the Flattened input as a diagonal
    Tb1 = np.diagflat(np.array([np.sqrt(2)/std_b, np.sqrt(2)/std_b, 1]))
    
    # Return a 2-D array with ones on the diagonal and zeros elsewhere.
    Tb2 = np.column_stack((np.row_stack((np.eye(2), [[0, 0]])), [-mean_b[0], -mean_b[1], 1]))

    Ta = np.matmul(Ta1, Ta2) # Matrix product of two arrays.
    Tb = np.matmul(Tb1, Tb2) # Matrix product of two arrays.

    arr_a = np.column_stack((points_a, [1]*points_a.shape[0])) # Adding an identity column
    arr_b = np.column_stack((points_b, [1]*points_b.shape[0])) # Adding an identity column

    arr_a = np.matmul(Ta, arr_a.T) # Matrix product of two arrays.
    arr_b = np.matmul(Tb, arr_b.T) # Matrix product of two arrays.

    arr_a = arr_a.T # Transpose of arr_a.
    arr_b = arr_b.T # Transpose of arr_b.

    arr_a = np.tile(arr_a, 3) # Construct an array by repeating arr_a three times
    arr_b = arr_b.repeat(3, axis=1)
    A = np.multiply(arr_a, arr_b)

    
    
    '''Solve f from Af=0'''
    U, s, V = np.linalg.svd(A)
    F_matrix = V[-1]
    F_matrix = np.reshape(F_matrix, (3, 3))
    F_matrix /= np.linalg.norm(F_matrix)

    '''Resolve det(F) = 0 constraint using SVD'''
    U, S, Vh = np.linalg.svd(F_matrix)
    S[-1] = 0
    F_matrix = U @ np.diagflat(S) @ Vh # matmul short form 

    F_matrix = Tb.T @ F_matrix @ Ta # matmul short form 
    
    
    ###########################################################################
    ###########################################################################

    return F_matrix

def ransac_fundamental_matrix(matches_a, matches_b):
    """
    Find the best fundamental matrix using RANSAC on potentially matching
    points. Your RANSAC loop should contain a call to
    estimate_fundamental_matrix() which you wrote in part 2.
    If you are trying to produce an uncluttered visualization of epipolar
    lines, you may want to return no more than 100 points for either left or
    right images.
    Args:
    -   matches_a: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from image A
    -   matches_b: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from image B
    Each row is a correspondence (e.g. row 42 of matches_a is a point that
    corresponds to row 42 of matches_b)
    Returns:
    -   best_F: A numpy array of shape (3, 3) representing the best fundamental
                matrix estimation
    -   inliers_a: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from image A that are inliers with
                   respect to best_F
    -   inliers_b: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from image B that are inliers with
                   respect to best_F
    """

    # Placeholder values
    #best_F = estimate_fundamental_matrix(matches_a[:10, :], matches_b[:10, :])
    #inliers_a = matches_a[:100, :]
    #inliers_b = matches_b[:100, :]

    ###########################################################################
    ###########################################################################
    num_iteration =15000
    number_matches=matches_b.shape[0] # Number of matches
    num_sample_rand = 8
    index_rand =np.random.randint(number_matches,size=(num_iteration,num_sample_rand)) 
    
    m=np.ones((3,number_matches)) # unity matrix
    m[0:2,:]=matches_a.T # stacking matches and identity columns for matches a
    
    mdash=np.ones((3,number_matches)) # unity matrix
    mdash[0:2,:]=matches_b.T # stacking matches and identity columns for matches b
    
    count=np.zeros(num_iteration) 
    err=np.zeros(number_matches)
    
    threshold=0.01 # Threshold value
    for i in range(num_iteration):
        
        cost1=np.zeros(8)
        
        #Finding the normalize fundamental estimation matrix (Normalized)
        F=estimate_fundamental_matrix_with_normalize(matches_a[index_rand[i,:],:]
                                                     ,matches_b[index_rand[i,:],:])
        #for each point calculate error
        for j in range(number_matches):
            err[j]=np.dot(np.dot(mdash[:,j].T,F),m[:,j])
        inlie=np.absolute(err)<threshold # only consider if error is less than the threshold 
        count[i]=np.sum(inlie + np.zeros(number_matches),axis=None)
        

    index=np.argsort(-count)
    best=index[0]
    
     #Finding the best normalize fundamental estimation matrix 
    
    best_F = estimate_fundamental_matrix_with_normalize(matches_a[index_rand[best,:],:],
                                                        matches_b[index_rand[best,:],:])
    for j in range(number_matches):
        err[j]=np.dot(np.dot(mdash[:,j].T,best_F),m[:,j])
    confidence=np.absolute(err)
    index=np.argsort(confidence)
    matches_b=matches_b[index]
    matches_a=matches_a[index]

    inliers_a=matches_a[:100,:]
    inliers_b=matches_b[:100,:]

    ###########################################################################
    ###########################################################################

    return best_F, inliers_a, inliers_b