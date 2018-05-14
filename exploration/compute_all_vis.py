import os
import sys
import code

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skfmm import distance
import argparse


def vis2d(phi, O):
# phi is the sdf of the scene
# O is the grid index of the vantage point

    Ny, Nx = phi.shape[:2]
    psi = 1e12 * np.ones((Ny, Nx))
    psi[O[0], O[1]] = phi[O[0], O[1]]

    # TODO: indexing is weird
    # NE sweep
    for i in range(O[0],Ny):
        for j in range(O[1],-1,-1):
            r1 = i - O[0]
            r2 = j - O[1]        
            if (r1 != 0) or (r2 != 0):
                A = 1.0/(r1-r2)
                psi[i,j] = A*(r1*psi[i-1,j]-r2*psi[i,j+1])
                psi[i,j] = min(phi[i,j],psi[i,j])

    # NW sweep
    for i in range(O[0],-1,-1):
        for j in range(O[1],-1,-1):
            r1 = i - O[0]
            r2 = j - O[1]        
            if (r1 != 0) or (r2 != 0):
                A = 1.0/(-r1-r2)
                psi[i,j] = A*(-r1*psi[i+1,j]-r2*psi[i,j+1])
                psi[i,j] = min(phi[i,j],psi[i,j])

    # SW sweep
    for i in range(O[0],-1,-1):
        for j in range(O[1],Nx):
            r1 = i - O[0]
            r2 = j - O[1]        
            if (r1 != 0) or (r2 != 0):
                A = 1.0/(-r1+r2)
                psi[i,j] = A*(-r1*psi[i+1,j]+r2*psi[i,j-1])
                psi[i,j] = min(phi[i,j],psi[i,j])

    # SE sweep
    for i in range(O[0],Ny):
        for j in range(O[1],Nx):
            r1 = i - O[0]
            r2 = j - O[1]        
            if (r1 != 0) or (r2 != 0):
                A = 1.0/(r1+r2)
                psi[i,j] = A*(r1*psi[i-1,j]+r2*psi[i,j-1])
                psi[i,j] = min(phi[i,j],psi[i,j])

    return psi


def compute_visibility(phi, psi, x0, dx):
    # assumes x0 is [n,2] matrix, each row is an observing location
    psi_current = vis2d(phi, np.round(x0[-1,:]/dx).astype(int))
    psi_current = distance(psi_current,dx)
    psi = np.maximum(psi, psi_current)
    return psi, psi_current


def img2obj(image):
    # from image, compute obstacles using row major indexing
    h,w = image.shape[:2]
    obj = []

    # assuming
    for i in range(h):
        for j in range(w):
            if image[i,j] == 0:
                obj.append(i*w + j)  # row major?
    return obj 


def obj2img(obj, h, w):

    # convert to image for computation
    image = 255 * np.ones((h,w))
    
    numObj = len(obj)
    
    for ind in obj:
        i = ind // w
        j = ind % w
        image[i,j] = 0

    return image


def createCircle(center, radius, x, y):
    dx = x[0,1] - x[0,0]
    mask = (x-center[1])**2 + (y-center[0])**2 <= radius**2
    mask = 1*mask # convert to number
    return mask

def compute_visibility_for_all(obj, h, w, radius = np.inf):
    image = obj2img(obj, h, w)
    image = image * 1.0/image.max() # rescale to [0,1]
    dx = 1.0/h
    phi = distance( (image-.5)*2*dx, dx ) # do we need to convert to signed distance function? yes

    notVis = {}  # dictionary containing {position: [non visible positions] }

    # create a grid for circular mask 
    x = np.linspace(0,w-1,w)
    y = np.linspace(0,h-1,h)
    x,y = np.meshgrid(x,y)

    for i in range(2,h-2):  # for now we have to skip borders until i have time to modify the edge cases for vis2d
        for j in range(2,w-2): 
            if phi[i,j] == 0:
                continue  # ignore positions inside obstacles
            mask = createCircle([i,j], radius, x, y)
            psi = vis2d(phi, [i,j])
            psi = 1*(psi>0) * mask
            I,J = np.where(psi==0)  # subscripts of obstacles
            occluded = (I*w + J).tolist()
            notVis[i*w + j] = occluded
    return notVis



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path')
    args = parser.parse_args()

    np.random.seed(0) 
    
    # simulate suda's gridworld input
    image = cv2.imread(args.image_path,cv2.IMREAD_GRAYSCALE) # 0 if obstacle, 255 if free space
    h,w = image.shape[:2]
    obj = img2obj(image)   # list containing row major indices of objects
   
    # compute visibility for each state
    notVis = compute_visibility_for_all(obj, h, w, radius = 10)

