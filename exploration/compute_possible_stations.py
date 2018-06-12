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


from skimage.feature import peak_local_max

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


def createCircle(center, radius, x, y):
    dx = x[0,1] - x[0,0]
    mask = (x-center[1])**2 + (y-center[0])**2 <= (radius*dx)**2
    mask = 1*mask # convert to number
    return mask


def compute_visibility_for_sequence(phi, psi, x0, dx):
    # assumes x0 is [n,2] matrix, each row is an observing location
    n = x0.shape[0]
    for i in range(n):
        psi_current = vis2d(phi, np.round(x0[i,:]/dx).astype(int))
        psi_current = distance(psi_current,dx)
        psi = np.maximum(psi, psi_current)

    return psi



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


def s2x(s, h, w, dx):
    # convert from index to x,y subscript
    s = np.array(s)
    i = np.expand_dims(s // w, axis=-1)
    j = np.expand_dims(s % w, axis=-1)

    x = np.concatenate((i, j), axis=1) * dx

    return x

def x2s(x, h, w, dx):
    # convert from x,y subscript to index   
    temp = np.round(x/dx)
    s = temp[:,0] * w + temp[:,1]
    return s




def get_stations(obj, h, w, predict_func, s, radius = 10, min_distance = 1, usePeaks=False):
# demonstration of how to setup the scene and observing locations
# to use the visibility algorithm to find the path
# psi is the phi
# psi is visibility

    # convert obj to image
    orig_im = obj2img(obj, h, w)

    # convert state to x,y 
    x0 = s2x(s, h, w, 1.0/h) 

    im = cv2.resize(orig_im, (INPUT_SIZE,INPUT_SIZE), cv2.INTER_NEAREST)  # resize to match model requirements

    # setup the grid
    m = INPUT_SIZE
    dx = 1.0/INPUT_SIZE
    scale = INPUT_SIZE * 1.0/h
    machine_eps = 1e-12
    eps = 2*dx
    grid_space = np.linspace(0,m-1,m)*dx 
    [x,y] = np.meshgrid(grid_space, grid_space)

     
    # compute the signed distance function for the map 
    phi = im2double(im)>.5
    phi = distance((2*phi-1)*dx, dx)
 
    residual = np.zeros(0)
    psi = -1e12
    index = 0
    finished = False

    # determine next step in sequential fashion
    psi = compute_visibility_for_sequence(phi, psi, x0, dx)
    predicted_gain = 0*psi
    vis = 1*(psi>0)
    hor = delta(psi,2*dx)*dx*(delta(phi,2*dx)==0)
    _, _, predicted_gain = predict(vis, hor, predict_func)
    
    # mask limited range
    mask = createCircle(x0[-1,:], radius*scale, x,y)
    predicted_gain = predicted_gain*mask*im

    # smooth to help with peak detection
    predicted_gain = cv2.GaussianBlur(predicted_gain,(5,5),0)

    threshold = 0.4 * 255  # only keep positions above this threshold
    
    if usePeaks:
        coordinates = peak_local_max(predicted_gain, min_distance= int(min_distance*scale))  # returns row, col
    else:
        coordinates = np.where(predicted_gain> threshold)
        coordinates = np.array([coordinates[0], coordinates[1]]).T


    # debug
    #plt.imshow(phi>0);plt.show()
    #plt.imshow(psi>0);plt.show()
    #plt.imshow(predicted_gain);plt.plot(coordinates[:,1],coordinates[:,0],'x');plt.show()

    # give pixel location in original image size

    coordinates = (coordinates/scale).astype(int)

    stations = []
    # remove stations if they are already visited or land on obstacle
    for i in range(coordinates.shape[0]):
        isNotObstacle = orig_im[coordinates[i,0], coordinates[i,1]] != 0
        newStation = coordinates[i,0] * w + coordinates[i,1]
        isNotVisited = newStation not in s
        if isNotObstacle and isNotVisited:
            stations.append(newStation)

    return stations



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
   

    # simulate suda's gridworld input
    image = cv2.imread(args.image_path,cv2.IMREAD_GRAYSCALE) # 0 if obstacle, 255 if free space
    h,w = image.shape[:2]
    obj = img2obj(image)   # list containing row major indices of objects

    # specify position is recent memory
    radius = 6
    #s = [340/2, 110/2]  # needs to be a list
    s = [131,147, 162]
    min_distance = 1  # minimum distance between each station (smaller means more locations are returned)

    usePeaks = True
    stations = get_stations(obj, h, w, predict_func, s, radius, min_distance, usePeaks)


    # plot the results
    x = s2x(s, h, w, 1)
    x_stations = s2x(stations, h, w, 1)
    print(stations)

    plt.imshow(image)
    plt.plot(x_stations[:,1], x_stations[:,0],'x')
    plt.plot(x[:,1], x[:,0],'ro')
    plt.show()


    

