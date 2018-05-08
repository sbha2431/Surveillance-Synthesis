import os
import sys
import code
import pickle

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skfmm import distance
import scipy.ndimage

# for cnn
import argparse
import tensorflow as tf
import numpy as np
from tensorpack import PredictConfig
from tensorpack import OfflinePredictor
from tensorpack import SaverRestore
from tensorpack import InputDesc
import model


INPUT_SIZE = 128
plt.rcParams['image.cmap'] = 'gray' #global colormap


def imfill(img):
    return 255-scipy.ndimage.morphology.binary_fill_holes(255-img)


def im2double(im):
    min_val = np.min(1*im.ravel())
    max_val = np.max(1*im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out

def preprocess(image_path, size):
    im = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE) 
    im = cv2.resize(im,(size, size))
    # hacky, model works on a fixed size
    im = (255 * (im>128) ).astype(np.uint8)

    small_im = 1*im

    im = cv2.resize(im,(INPUT_SIZE, INPUT_SIZE),cv2.INTER_NEAREST)
    im = (255 * (im>128) ).astype(np.uint8)
    im = imfill(im)
    im = im2double(im)

    return im, small_im


def delta(x, epsilon):
    chi = (x>(-epsilon/2)) * (x<(epsilon/2))
    y = 2/epsilon * chi * np.cos(np.pi * x / epsilon) **2
    return y


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


def plot_path(psi, x0):
    [m,n] = psi.shape
    x0[:,0] = 1+m-x0[:,0] - 0.5 # flipud for plotting purposes plt.contour(flipud(psi),0)
    x0[:,1] = x0[:,1] - 0.5
    plt.contour(np.flipud(psi),0) 
    plt.plot(x0[:,1], x0[:,0],'ko',mfc='none')
    #plt.plot(x0[0,1],x0[0,0],'r*')
    plt.plot(x0[-1,1],x0[-1,0],'r.')


def predict(vis, hor, predict_func):
    h,w = vis.shape
    vis = np.expand_dims(vis, -1)
    hor = np.expand_dims(hor, -1)
    image = np.concatenate((vis,hor),axis=-1)
    prediction = predict_func(np.expand_dims(image, 0))
    prediction = prediction * vis
    prediction = prediction[0].squeeze()
    prediction = cv2.resize(prediction, (h, w))
    location = np.unravel_index(prediction.argmax(),prediction.shape)
    return location[0], location[1], prediction


def moveOn(nextStep, x0, gain, tol):
    if (gain>.6).sum() == 0:
        print('No more gains')
        return True
    
    dist = np.sqrt( ((x0-nextStep)**2).sum(axis=1) )
    if np.any(dist < tol):
        return True


def show(psi, x0, dx, phi, hor, gain, psi_current, name):
    #plt.figure(num=None, figsize=(18, 4), dpi=80, facecolor='w', edgecolor='k')
    x0 = x0/dx-0.5
    temp = 1*(psi>0)
    temp[(psi_current>0)] = temp[psi_current>0] + 1
    plt.imshow(temp, vmin=-1,vmax=2)

    #plt.imshow(psi>0,vmin=-1,vmax=1)
    #plt.imshow(psi_current>0,vmin=-1,vmax=1)

    temp = psi*1.0
    temp[phi>2*dx] = None
    plt.contour(temp,0,colors='k')
    plt.plot(x0[:,1], x0[:,0],'b.', mfc='none')
    plt.plot(x0[-1,1],x0[-1,0],'r.')
    plt.plot(x0[-1,1],x0[-1,0],'ko', mfc='none')
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')

    plt.savefig(name,bbox_inches='tight', pad_inches=0, dpi=600)
    plt.close() 


def get_random_x0(im):
    # given image of free space, pick one
    m = im.shape[0] 
    dx = 1.0/m
    pad = 3
    temp = 1*im;
    temp[:pad,:] = 0
    temp[:,:pad] = 0
    temp[-pad:,:] = 0
    temp[:,-pad:] = 0
    idx = np.where(temp>0) 
    numPositions = (temp>0).sum()

    j = np.random.randint(numPositions)
    x0 = np.array([[idx[0][j], idx[1][j]]]) * dx
    return x0


def run_sequence(im, predict_func, x0, output_path = None):
# demonstration of how to setup the scene and observing locations
# to use the visibility algorithm to find the path
# psi is the phi
# psi is visibility

    # setup the grid
    h,w = im.shape
    m = h
    dx = 1.0/m 
    machine_eps = 1e-12
    eps = 2*dx
    grid_space = np.linspace(0,m-1,m)*dx 
    [x,y] = np.meshgrid(grid_space, grid_space)
  
    # compute the signed distance function for the map 
    phi = distance((2*im-1)*dx, dx)
 
    residual = np.zeros(0)
    psi = -1e12
    stepNum = 0
    index = 0
    finished = False

    numBlocks = m /INPUT_SIZE

    # determine next step in sequential fashion
    while not finished:
        print(stepNum)
        psi, psi_current = compute_visibility(phi, psi, x0, dx)
        predicted_gain = 0*psi
        vis = 1*(psi>0)
        hor = delta(psi,2*dx)*dx*(delta(phi,2*dx)==0)
        _, _, predicted_gain = predict(vis, hor, predict_func)

        # process the output
        tooClose = True            
        while tooClose: 
            new_y, new_x = np.unravel_index(predicted_gain.argmax(),predicted_gain.shape) 
            nextStep = np.array([new_y,new_x])*dx
            dist = np.sqrt( ((x0-nextStep)**2).sum(axis=1) )
            if np.any(dist < 20*dx):
                predicted_gain[new_y,new_x] = 0
                continue
            else:
                tooClose = False

        if moveOn(np.array([new_y,new_x])*dx, x0, predicted_gain, 20*dx): 
            print('WE ARE DONE')
            break

        if output_path is not None:
            show(psi, x0, dx, phi, hor, predicted_gain,psi_current, output_path+'_%.2d.png' % stepNum)

        x0 = np.append(x0,np.array([[new_y,new_x]])*dx,axis=0)
        residual = np.append(residual, (1*(phi>0)-1*(psi>0)).sum())
        stepNum += 1
   
    if output_path is not None: 
        show(psi, x0, dx, phi, hor, predicted_gain,psi, output_path+'_%.2d.png' % stepNum)
      
    residual = residual * 1.0/ (phi>0).sum()
    return x0, residual



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path')
    parser.add_argument('--model_path',default= 'log/checkpoint')
    parser.add_argument('--output_path', default='figures/')
    parser.add_argument('--size', type=int, default=32)
    args = parser.parse_args()

    np.random.seed(0) 
    # initialize the model
    predict_func = OfflinePredictor(
            PredictConfig(
                inputs_desc=[InputDesc(tf.float32, [None, INPUT_SIZE, INPUT_SIZE, 2], 'input_image')],
                tower_func=model.feedforward,
                session_init=SaverRestore(args.model_path),
                input_names=['input_image'],
                output_names=['prob']))
   
    # load the underlying map and resize to desired
    im, small_im = preprocess(args.image_path, args.size)

    # create frames for animation 
    prefix = os.path.join(args.output_path, os.path.basename(args.image_path)[:-4])
    x0 = get_random_x0(im)
    x0, residual = run_sequence(im, predict_func, x0)

    phi =  distance(im2double(small_im)-.5, 1.0/args.size)
    show(phi, x0, 1.0/args.size, phi, 0*phi, 0*phi, phi, prefix +'_path.png') 
    cv2.imwrite(prefix + '_map.png', small_im)

    # vantage points are in range [0,1] so we scale back to pixels [0,args.size]
    x0 = x0 * args.size
    print('The patrol stations are at:')
    print(x0)    

