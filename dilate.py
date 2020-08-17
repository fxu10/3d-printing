# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 11:10:15 2019

@author: Frederick Xu (fxu10)
"""

import numpy as np
import nibabel as nib

def main():
    print("This script will dilate a given .img image by a kernel size \
          provided by the user.")
    
    anatomy = "b4_ca1_top"
    fname = anatomy+".img"
    img = nib.load(fname)   
    hdr = img.header
    afn = img.affine
    
    data = img.get_data()
    print("Image size: " + str(data.shape))

    voxels = np.array(data.nonzero())
    vox_xyz = np.transpose(voxels[0:3])
    print("Number of non-zero voxels: " + str(vox_xyz.shape[0]))
    #Pulls the non-zero binary data from one of the voxels
    binary1 = data[vox_xyz[0][0]][vox_xyz[0][1]][vox_xyz[0][2]]

    k_size = 3
    k_size_half = float(k_size/2)
    k_u = int(k_size_half)
    k_l = -k_u
    kernel = []
    
    '''
    #builds a list of vertices that define a hollow offset box using the kernel size

    for xk in range(k_l, k_u+1):
        for yk in range(k_l, k_u+1):
            for zk in range(k_l, k_u+1):
                if(abs(xk) == k_size_half or abs(yk) == k_size_half or abs(zk) == k_size_half):
                    kernel.append([xk,yk,zk])
    print("Kernel box built: " + str(k_size) + " x " + str(k_size) + "\t Elements: " + str(len(kernel)))

    '''
    
    #builds a list of vertices that roughly define an offset sphere using the kernel size
    for xk in range(k_l, k_u+1):
        for yk in range(k_l, k_u+1):
            for zk in range(k_l, k_u+1):
                if(xk**2 + yk**2 + zk**2 < k_size_half**2):
                    kernel.append([xk,yk,zk])
    print("Kernel sphere built: Diameter " + str(k_size) + "\t Voxels: " + str(len(kernel)))

    dilate_index = []
    #dilate_index = np.array([0,0,0])
    for v in vox_xyz:
        dilate_index.extend(v+kernel)
        #dilate_index = np.concatenate((dilate_index,v+kernel),axis=None)
    dilate_index= np.array(dilate_index)
    #dilate_index = np.transpose(dilate_index)
    print("Dilation complete: " + str(dilate_index.shape))
    
    print("Writing out...")
    data_dil = data.copy()
    for d in dilate_index:
        data_dil[d[0]][d[1]][d[2]] = binary1
    
    dil_fname = anatomy+"_dilat_sph_"+str(k_size)+".img"
    dil_img = nib.AnalyzeImage(data_dil,afn)
    dil_img.to_filename(dil_fname)
    print("Write-out done")
main()