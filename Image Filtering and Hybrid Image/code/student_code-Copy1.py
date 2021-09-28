import numpy as np
import time

import numpy as np
import time

def my_imfilter(image, filter):
    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1
    raise NotImplementedError('`my_imfilter` function in `student_code.py` ' +
                              'needs to be implemented')
    
"""
  Apply a filter to an image. Return the filtered image.

  Args
  - image: numpy nd-array of dim (m, n, c)
  - filter: numpy nd-array of dim (k, k)
  Returns
  - filtered_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You may not use any libraries that do the work for you. Using numpy to work
   with matrices is fine and encouraged. Using opencv or similar to do the
   filtering for you is not allowed.
  - I encourage you to try implementing this naively first, just be aware that
   it may take an absurdly long time to run. You will need to get a function
   that takes a reasonable amount of time to run so that the TAs can verify
   your code works.
  - Remember these are RGB images, accounting for the final image dimension.
  """

"""

    ############################
    ### TODO: YOUR CODE HERE ###
    ############################
    
    ############################
    ###      Paddding      ###
    ############################'
"""
"""When we convolve the image the output shape is different than input and depends on shape of filter and stride. 
     If we suppose we do not want to alter the stride and keep it 1 unit (stepsize of the filter while convolution) 
     the out put shape can be computed as:
     
     out_shape = (input_height-filter_height+1, input_width-filter_width+1, n_channel)
     
     
     Suppose we have image of shape (256,256,3) and we want to apply a filter of size (3,7), then the output shape will be 
     
     out_shape = 256-3+1 ,256-7+1 , 3 = (254,252,3) 
     
     To get the output of the same size as of input we need to pad the input before applying the filter. The padding can be
     adding rows and columns of zeros around the boundary of the image.
     
     There are differnt ways of padding the image. 
     
     (1)One can create a zero valued numpy array with required shape of input before the Convolution and update the alligned
     array out of already declared array with image shape.
     
     (2)The other way is to use for loop to update the values for each pixel value. This will have more complexity as compared
     to previous method.
     
     (3) There are other function like numpy padding, It takes multiple arugments, the one for our concern are the first two.
     As first argument it takes the image array we need to pad and the second argument is the array describing the padding
     dimension. For using this functions we need to define where to put the padding and how many rows and columns we need to
     add. There are some key points to look at
     (1)We required to add even number of rows and column, (as the filter dimension are always odd giving m-1 and n-1 even). 
     
     (2) There is also a choice of selecting the value you want to pad with. The third argumetn in np add serve this purpose. 
     There are different mode you can select from by default it is on constant mode, ypou can also use the edge values as 
     padding values. 
     
     (3) As convolution of image and filter doesn't affect the number of channel so while padding we will make sure not to pad 
     the channels. """

  [h,w,c] = image.shape
  [m,n] =filter.shape


############################
###      First method    #### This method is more efficient for small sized images but not for very big images.
############################  Its time efficieny increase drastically with increase in size of the image.
############################  One can make it more efficient by not declaring a zero valued array instead just the one column
############################  and row which can be concatenated to the orignal images to form a padded image. Decalring 
############################  Declaring numpy array of zeros is more costly. 
"""
  t= time.time()
  padded_image = np.zeros((h+m-1,w+n-1,c))
  padded_image[m-2:-1,n-2:-1]=image
  print('consumed time is using array assignment is' + str(time.time()-t)+ ' seconds')
  
"""
"""
    ############################
    ###     Third method     ### This method is more effeicent as compared to all, one thing to bear in mind is the size    
    ###affects 
    ############################  the time consumed for padding. it almost spents 0.001 seconds 
"""
  t= time.time()
  padded = np.pad(image,((int((m-1)/2),int((m-1)/2)),(int((n-1)/2),int((n-1)/2)),(0,0)))
  print('consumed time is using np.pad is ', time.time()-t)
"""
    #################################################################################################################
    ######### Padding Finished########
    ############################
    ###      Convolution     ###
    ############################
"""
    
"""
   As we are unaware of the channels in the images, there can be 3 or just one (in case of gray scale) chanel.
   The complexity of the algorithm depends on the number of chanels along side the size of the image and filter. If We use the 
   Brute Force algorithm there will be four for loops in case of gray scale and five in case of colour image giving complexity 
   of O(N^4) and O(N^5) respectively for gray and colour image. 
   
   Brute Force Algorithm is very costly and takes very high amount of time. We will implement different techniques to quickly 
   solve the convolution problem. 
   
"""
"""    
    ############################
    ###      BRUTE FORCE     ###
    ############################
"""   
  out_image  = np.zeros_like(image)
    
   




"""
    ### END OF STUDENT CODE ####
    ############################
"""


  return filtered_image

def create_hybrid_image(image1, image2, filter):
  """
  Takes two images and creates a hybrid image. Returns the low
  frequency content of image1, the high frequency content of
  image 2, and the hybrid image.

  Args
  - image1: numpy nd-array of dim (m, n, c)
  - image2: numpy nd-array of dim (m, n, c)
  Returns
  - low_frequencies: numpy nd-array of dim (m, n, c)
  - high_frequencies: numpy nd-array of dim (m, n, c)
  - hybrid_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You will use your my_imfilter function in this function.
  - You can get just the high frequency content of an image by removing its low
    frequency content. Think about how to do this in mathematical terms.
  - Don't forget to make sure the pixel values are >= 0 and <= 1. This is known
    as 'clipping'.
  - If you want to use images with different dimensions, you should resize them
    in the notebook code.
  """

  assert image1.shape[0] == image2.shape[0]
  assert image1.shape[1] == image2.shape[1]
  assert image1.shape[2] == image2.shape[2]

  ############################
  ### TODO: YOUR CODE HERE ###

  raise NotImplementedError('`create_hybrid_image` function in ' + 
    '`student_code.py` needs to be implemented')

  ### END OF STUDENT CODE ####
  ############################

  return low_frequencies, high_frequencies, hybrid_image
