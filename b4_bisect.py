# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

"""
Data is structured as [][][][]:
    
X, Y, Z, I

All are 3D arrays of the same size.  I is a binary image.  X,Y,Z tell 
you the X and Y and Z component of each pixel.

Find the 1st and 2nd central moments of this data.

1st moment is a 3x1 vector, 2nd moment is a 3x3 matrix.

These moments define an ellipsoid in 3D.

*Probably* what you want to do, is cut the ellipsoid in half.

Find the shortest major axis, and cut the data about this plane.

You can construct a plane based on one point (1st moment) and normal 
vector (smallest axis).

Some notes:
    https://en.m.wikipedia.org/wiki/Moment_(mathematics)
    
    https://en.m.wikipedia.org/wiki/Center_of_mass
    
    http://adaptivemap.ma.psu.edu/websites/moment_intergrals/centroids_3D/centroids3D.html
    
    https://nipy.org/nibabel/reference/nibabel.analyze.html#module-nibabel.analyze

"""

import numpy as np
import nibabel as nib
import math as m

def main():
    print("This script will bisect a given analyze image (.img) using a centroid and eigenvectors of an ellipse.\n")
    
    # PRELIMINARY LOADING
    fname = 'srlm.img'
    img = nib.load(fname)   
    hdr = img.header
    afn = img.affine
    nxI = hdr['dim'][1:4] 
    data = img.get_data()
    
    # FIRST MOMENT CALCULATION
    print("-----FIRST MOMENT CALCULATION-----\n")
    
    voxels = np.array(data.nonzero())
    vox_xyz = np.transpose(voxels[0:3])
    img_size = np.shape(vox_xyz)[0]
    print("Non-zero voxels: " + str(img_size))
    
    X0 = np.sum(vox_xyz,axis=0)/img_size
    print("Centroid: " + str(X0) + "\n")
    
    # SECOND MOMENT
    print("-----SECOND MOMENT CALCULATION-----\n")
    x_c, y_c, z_c = X0[0], X0[1], X0[2] 
    
    xx_c, xy_c, xz_c, yy_c, yz_c, zz_c = 0, 0, 0, 0, 0, 0

    for v in vox_xyz:
        xx_c = xx_c + (v[0] - x_c)*(v[0] - x_c);
        xy_c = xy_c + (v[0] - x_c)*(v[1] - y_c);
        xz_c = xz_c + (v[0] - x_c)*(v[2] - z_c);
        yy_c = yy_c + (v[1] - y_c)*(v[1] - y_c);
        yz_c = yz_c + (v[1] - y_c)*(v[2] - z_c);
        zz_c = zz_c + (v[2] - z_c)*(v[2] - z_c);
    
    T = [[xx_c,xy_c,xz_c],
        [xy_c,yy_c,yz_c],
        [xz_c,yz_c,zz_c]]
    
    T = np.array(T)
    e = np.linalg.eig(T)
    
    print("Eigenvalues: " + str(e[0]))
    print("Eigenvectors: \n" + str(e[1]) + "\n")
    
    # The eigenvector corresponding to the smallest eigenvalue is the smallest axis
    # This will be normal to the bisecting plane to cut the data along the largest face
    min_ind = np.argmin(e[0])
    norm = e[1][min_ind]
    print("Smallest eigenvalue:  " + str(e[0][min_ind]))
    print("Smallest eigenvector: " + str(norm) + "\n")
    
    norm = e[1][2]
    print("Chosen bisection plane normal: " + str(norm) + "\n")
    
    print("Note: the chosen bisection normal is selected as the eigenvector corresponding to the smallest eigenvector.\nIf you want to use another eigenvector, you can manually select it using indexing.\n")

    # VECTOR ROTATION (OPTIONAL)
    # NOTE: These must be ordered in the way you want them to work, rotations are NOT commutative
    # All theta values are in radians.
    
    '''
    # X-AXIS ROTATION
    theta_x = 0 
    
    xrot_mat = [[1, 0,                0             ], 
                [0, m.cos(theta_x),  -m.sin(theta_x)],
                [0, m.sin(theta_x),   m.cos(theta_x)]]
    norm = np.transpose(np.matmul(xrot_mat, np.transpose(norm)))
    '''
    
    # Y-Axis Rotation
    #theta_y = m.pi/6 #Radians
    #theta_y = m.pi/8 #Radians

    theta_y = m.pi/7 #Radians
    yrot_mat = [[  m.cos(theta_y),  0, m.sin(theta_y)], 
                [  0,               1, 0             ],
                [-(m.sin(theta_y)), 0, m.cos(theta_y)]]
    norm = np.transpose(np.matmul(yrot_mat, np.transpose(norm)))
    
    
    # Z-Axis Rotation
    #theta_z = m.pi/5
    theta_z = m.pi/4 #Radians
    #theta_z = m.pi/3 #Radians    
    #theta_z = m.pi/2 #Radians
    zrot_mat = [[m.cos(theta_z), -(m.sin(theta_z)), 0], 
                [m.sin(theta_z),   m.cos(theta_z),  0],
                [0,                0,               1]]
    norm = np.transpose(np.matmul(zrot_mat, np.transpose(norm)))
    
    
    '''
    # Y-Axis Rotation
    theta_y = m.pi/4 #Radians
    yrot_mat = [[  m.cos(theta_y),  0, m.sin(theta_y)], 
                [  0,               1, 0             ],
                [-(m.sin(theta_y)), 0, m.cos(theta_y)]]
    norm = np.transpose(np.matmul(yrot_mat, np.transpose(norm)))
   '''
    print("Rotated Norm: " + str(norm) + "\n")
    
    #Values that work:  theta_z = m.pi/3   theta_y = m.pi/8
    
    
    X0[2] = X0[2] + 25
    
    print("Shifted Centroid: " + str(X0) + "\n")
        
    # WRITING OUT
    print("-----SAVING ANALYZE IMAGE-----\n")
    print("Writing out file...")
    bot = data.copy();
    top = data.copy();
    # For all of the non-zero data points, check if they lie above or below the plane
    # if they lie below, remove them from the top dataset
    # if they lie above, remove them from the bottom dataset
    for v in vox_xyz:
        if(norm[0]*(v[0]-X0[0]) + norm[1]*(v[1]-X0[1]) + norm[2]*(v[2]-X0[2]) <= 0):
            top[v[0]][v[1]][v[2]] = 0;
        if(norm[0]*(v[0]-X0[0]) + norm[1]*(v[1]-X0[1]) + norm[2]*(v[2]-X0[2]) > 0):
            bot[v[0]][v[1]][v[2]] = 0;
    
    bot_img = nib.AnalyzeImage(bot,afn)
    top_img = nib.AnalyzeImage(top,afn)
    
    bot_img.to_filename("b4_srlm_bot_TEST.img")
    top_img.to_filename("b4_srlm_top_TEST.img")
    
    file = open("bisect_param.txt","w")
    file.write(str(norm[0]) + "\n")
    file.write(str(norm[1]) + "\n")
    file.write(str(norm[2]) + "\n")
    file.write(str(X0[0]) + "\n")
    file.write(str(X0[1]) + "\n")
    file.write(str(X0[2]) + "\n")
    file.close()
    
    print("Write-out done.")
main()