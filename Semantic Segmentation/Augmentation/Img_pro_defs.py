#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
#from keras.utils.np_utils import to_categorical
#from keras.models import Sequential
#from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
#import keras.backend as K
import glob
from PIL import Image
from sklearn.model_selection import train_test_split 
#from keras.optimizers import SGD, Adam, Adadelta, RMSprop
from sklearn.metrics import confusion_matrix
#from keras.callbacks import EarlyStopping
from keras.preprocessing import image

import cv2
import os
import re
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000 
from skimage import io
#import labelme
from scipy import ndimage
PIXEL = 1080 * 1616 
import imgaug as ia
import imgaug.augmenters as iaa
#import shutil

def read_data (path,target_size):
    filelist = glob.glob(path)
    data = np.array([image.img_to_array(image.load_img(fname,target_size = target_size)) for fname in filelist])
    return data


def sorted_nicely( li ):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(li, key = alphanum_key)

def hsv_segment (data,start,path):
    i = start-1
    j = 0
    for i in range (start-1,len(data)+start-1):
        img = data[j]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        ## mask of green (36,0,0) ~ (70, 255,255)
        mask = cv2.inRange(img, (25,0,0), (120, 255,255))
        ## slice the green
        imask = mask>0
        green = np.zeros_like(img, np.uint8)
        green[imask] = img[imask]
        cv2.imwrite(os.path.join(path,'{}.png'.format(i+1)) , green)
        j= j+1
    return i+1

def write_nicely (data,start,path):
    i = start-1
    j = 0
    for i in range (start-1,len(data)+start-1):
        img = data[j]
        cv2.imwrite(os.path.join(path,'{}.png'.format(i+1)) , img)
        j= j+1
    return i+2


def create_collage(files, out,dim):
    target_img = Image.new("RGB", (1616*dim,1080*dim))
    for k, png in enumerate(files):
        row, col = divmod(k,dim)
        img = Image.open(png)
        img.thumbnail((1616, 1080))
        target_img.paste(img, (1616*row, 1080*col))
    target_img.save(out)
    
def image_reconst (files,dest):
    for k,img in enumerate (dest):
        create_collage1(files[k*4:k*4+4],img,2)
    
def create_collage1(files, out,dim):
    target_img = Image.new("RGB", (800*dim,544*dim))
    for k, png in enumerate(files):
        row, col = divmod(k,dim)
        img = Image.open(png)
        img.thumbnail((800,544))
        target_img.paste(img, (800*row, 544*col))
    target_img.save(out)

def image_split(file_names,b_image,dim):
    
    # You can add a check to guarantee that the list of images is not bigger than 64...
    img = image.img_to_array(image.load_img((b_image)))
    k = 0
    print (img.shape)
    for col in range(dim):
        for row in range (dim):
            crop = img[row*1080:row*1080+1080, col*1616:col*1616+1616]
            cv2.imwrite(file_names[k],crop)
            print (file_names[k])
            print (crop.shape)
                              
            k = k+1
            if k == len(file_names):
                break
def image_split1(k,b_image,dim_c,dim_r,width,height,path):
    
    img = image.img_to_array(image.load_img(b_image,target_size =(height*dim_r,width*dim_c)))
    print (img.shape)
    for col in range(dim_c):
        for row in range (dim_r):
            crop = img[row*height:row*height+height, col*width:col*width+width]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(path,'{}.png'.format(k)) , crop)
            k = k+1
    return k
                
def labeler (listdir, path, path1, path2):
    for i in range(len(listdir)):
        json = cv2.imread(path + '/' + listdir[i] + '/label.png')
        json[json==0] = 1
        ml_mask = cv2.imread(path1 + '/' + listdir[i] +'.JPG' )
        ml_mask[ml_mask!=0] = 1
        final = json*ml_mask
        cv2.imwrite(path2 + '/' + listdir[i] +'.png', final)
        
def count_nonblack_np(img):
    """Return the number of pixels in img that are not black.
    img must be a Numpy array with colour values along the last axis.

    """
    return img.any(axis=-1).sum()

def thresholding (image, th):
    image = image[:,:,0] + image[:,:,1] + image[:,:,2]
    image[image<th] = 0
    return image  

def median_filter (image, size):
    return ndimage.median_filter(image,size)

def crop_pixel_counter (files, data):
    files = sorted_nicely(files)
    data = data.set_index('filename')
    data['Crop Leaves Area'] = 0
    data['Weed Leaves Area'] = 0
    data = data.astype({"Crop Leaves Area": float, "Weed Leaves Area": float})
    for k, file in enumerate (files):
        image = cv2.imread (file)
        print (image.shape)
        #image = thresholding(image,thd)
        #image = median_filter(image, filter_size)
        image = image.sum(axis = 2)
        image[image!=0] = 1
        crop_percent = (np.sum(image))/(1616*1080)*100
        data['Crop Leaves Area'][file[31:51] + '.JPG'] = crop_percent
    return data        

      
def background_remove (mask, original):
    mask[mask == 1] = 0
    mask[mask != 0] = 1
    return (mask * original)

def file_copy (files,dst,symlinks=True):
    for k, file in enumerate(files):
        shutil.copy(file, dst,follow_symlinks=symlinks)

def augment_seg( img , seg  ):
    aug_det = seq.to_deterministic() 
    image_aug = aug_det.augment_image( img )
    segmap = ia.SegmentationMapOnImage( seg , nb_classes=2 , shape=img.shape )
    segmap_aug = aug_det.augment_segmentation_maps( segmap )
    segmap_aug = segmap_aug.get_arr_int()
    return image_aug , segmap_aug

def augment_flip (X_train, Y_train, X_test, Y_test):
    # Images Inverted horizontally
    X_train_hflip, Y_train_hflip = iaa.Fliplr(1.0)(images = X_train.astype('uint8'), heatmaps = Y_train)
    X_test_hflip, Y_test_hflip = iaa.Fliplr(1.0)(images = X_test.astype('uint8'), heatmaps = Y_test)
    
    # Images Inverted vertically
    X_train_vflip, Y_train_vflip = iaa.Flipud(1.0)(images = X_train.astype('uint8'), heatmaps = Y_train)
    X_test_vflip, Y_test_vflip = iaa.Flipud(1.0)(images = X_test.astype('uint8'), heatmaps = Y_test)
    
    # Images Inverted vertically and horizontally both
    X_train_vhflip, Y_train_vhflip = iaa.Fliplr(1.0)(images = X_train_vflip.astype('uint8'), heatmaps = Y_train_vflip)
    X_test_vhflip, Y_test_vhflip = iaa.Fliplr(1.0)(images = X_test_vflip.astype('uint8'), heatmaps = Y_test_vflip)
    
    # concatenating arrays--------1
    X_train_flips = np.concatenate((X_train,X_train_vflip,X_train_vhflip,X_train_hflip), axis = 0)
    Y_train_flips = np.concatenate((Y_train,Y_train_vflip,Y_train_vhflip,Y_train_hflip), axis = 0)
    X_test_flips = np.concatenate((X_test,X_test_vflip,X_test_vhflip,X_test_hflip), axis = 0)
    Y_test_flips = np.concatenate((Y_test,Y_test_vflip,Y_test_vhflip,Y_test_hflip), axis = 0)

    return X_train_flips, Y_train_flips, X_test_flips,Y_test_flips


def augment_scale (X_train_flips, Y_train_flips, X_test_flips,Y_test_flips):
    # Image scaling
    X_train_scale, Y_train_scale = iaa.Affine(scale  = {"x" : (0.8,1.2), "y" : (0.8,1.2)})(images = X_train_flips.astype('uint8'),heatmaps = Y_train_flips)
    X_test_scale, Y_test_scale = iaa.Affine(scale  = {"x" : (0.8,1.2), "y" : (0.8,1.2)})(images = X_test_flips.astype('uint8'),heatmaps = Y_test_flips)
    
    # concatenated original, fliped and scaled (ofs)----2
    X_train_ofs = np.concatenate((X_train_flips,X_train_scale), axis = 0)
    Y_train_ofs = np.concatenate((Y_train_flips,Y_train_scale), axis = 0)
    X_test_ofs = np.concatenate((X_test_flips,X_test_scale), axis = 0)
    Y_test_ofs = np.concatenate((Y_test_flips,Y_test_scale), axis = 0)
    
    return X_train_ofs, Y_train_ofs, X_test_ofs, Y_test_ofs

def augment_rotate (X_train_ofs, Y_train_ofs, X_test_ofs, Y_test_ofs):
    # concatenated original, fliped, scaled and rotated (ofsr)----3
    X_train_rot, Y_train_rot = iaa.Affine(rotate  = (-45,45))(images = X_train_ofs.astype('uint8'),heatmaps = Y_train_ofs)
    X_test_rot, Y_test_rot = iaa.Affine(rotate  = (-45,45))(images = X_test_ofs.astype('uint8'),heatmaps = Y_test_ofs)
    
    # concatenated original, fliped, scaled and rotated (ofsr)----3
    X_train_ofsr = np.concatenate((X_train_rot,X_train_ofs), axis = 0)
    Y_train_ofsr = np.concatenate((Y_train_rot,Y_train_ofs), axis = 0)
    X_test_ofsr = np.concatenate((X_test_rot,X_test_ofs), axis = 0)
    Y_test_ofsr = np.concatenate((Y_test_rot,Y_test_ofs), axis = 0)
    
    return X_train_ofsr, Y_train_ofsr, X_test_ofsr, Y_test_ofsr 

def augment_shear (X_train_ofsr, Y_train_ofsr, X_test_ofsr, Y_test_ofsr):
    # concatenated original, fliped, scaled, rotated and shear(ofsrs)----4
    X_train_shr, Y_train_shr = iaa.Affine(shear=(-16, 16))(images = X_train_ofsr.astype('uint8'),heatmaps = Y_train_ofsr)
    X_test_shr, Y_test_shr = iaa.Affine(shear=(-16, 16))(images = X_test_ofsr.astype('uint8'),heatmaps = Y_test_ofsr)
    
    X_train_ofsrs = np.concatenate((X_train_shr,X_train_ofsr), axis = 0)
    Y_train_ofsrs = np.concatenate((Y_train_shr,Y_train_ofsr), axis = 0)
    X_test_ofsrs = np.concatenate((X_test_shr,X_test_ofsr), axis = 0)
    Y_test_ofsrs = np.concatenate((Y_test_shr,Y_test_ofsr), axis = 0)

    return X_train_ofsrs,  Y_train_ofsrs,  X_test_ofsrs, Y_test_ofsrs

def augment_crop_pad (X_train_ofsrs,  Y_train_ofsrs,  X_test_ofsrs, Y_test_ofsrs):
    # concatenated original, fliped, scaled, rotated, shear(ofsrs) and crop----4
    X_train_c, Y_train_c = iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255))(images = X_train_ofsrs.astype('uint8'),heatmaps = Y_train_ofsrs)

    X_test_c, Y_test_c = iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255))(images = X_test_ofsrs.astype('uint8'),heatmaps = Y_test_ofsrs)
    
    X_train_ofsrsc = np.concatenate((X_train_c,X_train_ofsrs), axis = 0)
    Y_train_ofsrsc = np.concatenate((Y_train_c,Y_train_ofsrs), axis = 0)
    X_test_ofsrsc = np.concatenate((X_test_c,X_test_ofsrs), axis = 0)
    Y_test_ofsrsc = np.concatenate((Y_test_c,Y_test_ofsrs), axis = 0)

    return  X_train_ofsrsc, Y_train_ofsrsc, X_test_ofsrsc, Y_test_ofsrsc

def augment_blur (X_train_ofsrsc, Y_train_ofsrsc, X_test_ofsrsc, Y_test_ofsrsc, concat = False):
    
    X_train_b, Y_train_b = iaa.GaussianBlur(sigma=(0, 2.0))(images = X_train_ofsrsc.astype('uint8'),heatmaps = Y_train_ofsrsc)

    X_test_b, Y_test_b = iaa.GaussianBlur(sigma=(0, 2.0))(images = X_test_ofsrsc.astype('uint8'),heatmaps = Y_test_ofsrsc)

    if concat == True:
        
        X_train_b = np.concatenate((X_train_b,X_train_ofsrsc), axis = 0)
        Y_train_b = np.concatenate((Y_train_b,Y_train_ofsrsc), axis = 0)
        X_test_b = np.concatenate((X_test_b,X_test_ofsrsc), axis = 0)
        Y_test_b = np.concatenate((Y_test_b,Y_test_ofsrsc), axis = 0)
        
    return X_train_b, Y_train_b, X_test_b, Y_test_b

def augment_multi ( X_train_ofsrsc, Y_train_ofsrsc, X_test_ofsrsc, Y_test_ofsrsc, concat = False ):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.
    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            iaa.Flipud(0.2), # vertically flip 20% of all images
            # crop images by -5% to 10% of their height/width
            sometimes(iaa.CropAndPad(
                percent=(-0.05, 0.1),
                pad_mode=ia.ALL,
                pad_cval=(0, 255)
            )),
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45), # rotate by -45 to +45 degrees
                shear=(-16, 16), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                [
                    sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                        #iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                    # search either for all edges or for directed edges,
                    # blend the result with the original image using a blobby mask
                    iaa.SimplexNoiseAlpha(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0.5, 1.0)),
                        iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                    ])),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                        iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                    ]),
                    iaa.Invert(0.05, per_channel=True), # invert color channels
                    iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                    iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                    # either change the brightness of the whole image (sometimes
                    # per channel) or change the brightness of subareas
                    iaa.OneOf([
                        iaa.Multiply((0.5, 1.5), per_channel=0.5),
                        iaa.FrequencyNoiseAlpha(
                            exponent=(-4, 0),
                            first=iaa.Multiply((0.5, 1.5), per_channel=True),
                            second=iaa.ContrastNormalization((0.5, 2.0))
                        )
                    ]),
                    iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                    sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                ],
                random_order=True
            )
        ],
        random_order=True
    )
    
    
    X_train_multi, Y_train_multi = seq ( images = X_train_ofsrsc , heatmaps = Y_train_ofsrsc )
    X_test_multi, Y_test_multi = seq ( images = X_test_ofsrsc , heatmaps = Y_test_ofsrsc )
    
    if concat == True:
        
        X_train_multi = np.concatenate((X_train_multi,X_train_ofsrsc), axis = 0)
        Y_train_multi = np.concatenate((Y_train_multi,Y_train_ofsrsc), axis = 0)
        X_test_multi = np.concatenate((X_test_multi,X_test_ofsrsc), axis = 0)
        Y_test_multi = np.concatenate((Y_test_multi,Y_test_ofsrsc), axis = 0)
    
    return X_train_multi, Y_train_multi, X_test_multi, Y_test_multi

def augment_high_order (X_train, Y_train, X_test, Y_test,mode):

    if mode==0:
        X_train, Y_train, X_test, Y_test = augment_flip (X_train, Y_train, X_test, Y_test)
        print("Step 1 completed")
        X_train, Y_train, X_test, Y_test = augment_scale (X_train, Y_train, X_test, Y_test)
        print("Step 2 completed")
        X_train, Y_train, X_test, Y_test= augment_rotate (X_train, Y_train, X_test, Y_test)
        print("Step 3 completed")
        return X_train, Y_train, X_test, Y_test	
    if mode == 1:
        X_train, Y_train, X_test, Y_test = augment_shear (X_train, Y_train, X_test, Y_test)
        print("Step 4 completed")
        X_train, Y_train, X_test, Y_test = augment_crop_pad (X_train, Y_train, X_test, Y_test)
        print("Step 5 completed")
        X_train, Y_train, X_test, Y_test = augment_blur (X_train, Y_train, X_test, Y_test)
        print("Step 6 completed")
        print("All Done !!")
    
    return X_train, Y_train, X_test, Y_test
    #X_train_multi, Y_train_multi, X_test_multi, Y_test_multi = augment_multi ( X_train, Y_train, X_test, Y_test, concat = False )
    
    
    
    #print("Step 7 completed and concatenating")
    
    #X_train_aug = np.concatenate((X_train_b,X_train), axis = 0)
    #Y_train_aug = np.concatenate((Y_train_b,Y_train), axis = 0)
    #X_test_aug = np.concatenate((X_test_b,X_test), axis = 0)
    #Y_test_aug = np.concatenate((Y_test_b,Y_test), axis = 0)
    
    
    
    
