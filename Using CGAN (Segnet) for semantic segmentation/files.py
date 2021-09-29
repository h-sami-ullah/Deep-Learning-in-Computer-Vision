#!/usr/bin/env python
# coding: utf-8

# In[27]:


import os
def filepaths():
    path_data='//home//sami//Downloads//ijrr_sugarbeets_2016_annotations//'
    subfolderall = [f.path for f in os.scandir(path_data) if f.is_dir() ]
    subfolderall=sorted(subfolderall)
    number_of_folders=len(subfolderall)
    minisub=l = [[]] * 20
    for i in range(number_of_folders):
        minisub[i]=[f.path for f in os.scandir(subfolderall[i]) if f.is_dir() ] 
    minisub=sorted(minisub)
    imgs_folder_for_train=[[]]*20
    for i in range(len(minisub)):
        imgs_folder_for_train[i]=[f.path for f in os.scandir(minisub[i][1]) if f.is_dir() ]
    imgs_folder_for_train=sorted(imgs_folder_for_train)  
    imgs_folder_for_labeled=[[]]*20
    for i in range(len(minisub)):
        imgs_folder_for_labeled[i]=[f.path for f in os.scandir(minisub[i][0]) if f.is_dir() ]
    imgs_folder_for_labeled_next=[[]]*20
    for i in range(len(imgs_folder_for_labeled)):
        imgs_folder_for_labeled_next[i]=[f.path for f in os.scandir(imgs_folder_for_labeled[i][0]) if f.is_dir() ]
    imgs_folder_for_labeled=sorted(imgs_folder_for_labeled)
    imgs_folder_for_labeled_next=sorted(imgs_folder_for_labeled_next)
    filesnir = []
    filesrgb = []
# r=root, d=directories, f = files
    for i in range(len(imgs_folder_for_train)):
        for rnir, dnir, fnir in os.walk(imgs_folder_for_train[i][1]):
            for filenir in fnir:
                if '.png' in filenir:
                    filesnir.append(os.path.join(rnir, filenir))
        for rrgb, drgb, frgb in os.walk(imgs_folder_for_train[i][0]):
            for filergb in frgb:
                if '.png' in filergb:
                    filesrgb.append(os.path.join(rrgb, filergb))
    filesrgb=sorted(filesrgb)
    filesnir=sorted(filesnir)
    files_coloured_label = []
# r=root, d=directories, f = files
    for i in range(len(imgs_folder_for_labeled_next)):
        for rlabel, dlabel, flabel in os.walk(imgs_folder_for_labeled_next[i][1]):
            for filelabel in flabel:
                if '.png' in filelabel:
                    files_coloured_label.append(os.path.join(rlabel, filelabel))
    files_coloured_label=sorted(files_coloured_label)
    y_address=files_coloured_label[0:11552]
    x_rgb_address=filesrgb[0:11552]
    x_nir_address=filesnir[0:11552]
    #for i in range(len(x_nir_address)):
    #    x_rgb_address.append(x_nir_address[i])
    #x_rgb_and_nir_address=x_rgb_address
    return x_rgb_address,x_nir_address, y_address

