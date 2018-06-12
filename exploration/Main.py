__author__ = 'sudab'

from gridworld import *
import os
import subprocess
import Salty_input
import simulateController
import compute_all_vis
import pickle
from tqdm import *
import cv2
import model
import copy
from tensorpack import PredictConfig
from tensorpack import OfflinePredictor
from tensorpack import SaverRestore
from tensorpack import InputDesc
import tensorflow as tf
import compute_possible_stations

model_path = 'log/checkpoint'
mapname = 'chicago4_45_2454_5673_map_16'
# mapname = 'chicago4_45_2454_5673_map'
filename = 'figures/'+mapname+'.png'

image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)  # 0 if obstacle, 255 if free space
h, w = image.shape[:2]
obj = compute_all_vis.img2obj(image)
visdist = [10] # Visibility range for agent
np.random.seed(0)
initial = [131] # Initial agent position
moveobstacles = [132]
visited = []
currentvisit = initial[0]
while True:
    stationnum = 0
    visited.append(currentvisit)
    print 'Visited stations ', visited
    # initialize the model
    predict_func = OfflinePredictor(
            PredictConfig(
                inputs_desc=[InputDesc(tf.float32, [None, 128, 128, 2], 'input_image')],
                tower_func=model.feedforward,
                session_init=SaverRestore(model_path),
                input_names=['input_image'],
                output_names=['prob']))
    min_distance = 1  # minimum distance between each station (smaller means more locations are returned)
    stations = compute_possible_stations.get_stations(obj, h, w, predict_func, visited, visdist[0], min_distance)
    print stations

     #Initial state of moving obstacle(s)
    gwg = Gridworld(filename,targets=[stations],initial=[currentvisit],moveobstacles=moveobstacles) #Generate gridworld class
    gwg.render()
    gwg.save('Example1.png')
    gwg.draw_state_labels()
    slugs = '/home/sudab/Applications/slugs/src/slugs' # Path to slugs executable



    ##Defining allowed states for agents to move and abstract belief states
    allowed_states = [None]
    allowed_states[0] = list(set(range(gwg.nstates)) - set(gwg.walls)) # This removes all the bordering states.
    fullvis_states = [[]] #States where we have full visibility at all times
    partitionGrid = dict()
    partitionGrid[(0,0)] = copy.deepcopy(allowed_states[0]) #Number of belief states - more states -> more refined but slower synthesis
    pg = [dict.fromkeys((0,0),partitionGrid[(0,0)])]


    vel = [3] #No of states the agent can move in one timestep
    print 'Writing input file...'
    invisibilityset = []
    outfile = 'Example1_'+str(stationnum)+'.json'
    infile = 'Example1'
    print 'output file: ', outfile
    print 'input file name:', infile

    h, w = image.shape[:2]
    for n in range(gwg.nagents):
        obj = compute_all_vis.img2obj(image)
        # compute visibility for each state
        iset = compute_all_vis.compute_visibility_for_all(obj, h, w,radius=visdist[n])
        Salty_input.write_to_slugs_part_dist(infile,gwg,currentvisit,moveobstacles[n],iset,stations,vel[n],visdist[n],
                                                    allowed_states[n],fullvis_states[n], pg[n], belief_safety = 1, belief_liveness =0, target_reachability = True) #Write input file
        invisibilityset.append(iset)
        print ('Converting input file...')
        os.system('python compiler.py ' + infile + '.structuredslugs > ' + infile + '.slugsin')
        print('Computing controller...')
        sp = subprocess.Popen(slugs + ' --explicitStrategy --jsonOutput ' + infile + '.slugsin > '+ outfile,shell=True, stdout=subprocess.PIPE)
        # sp = subprocess.Popen(slugs + ' ' + infile + '.slugsin simple.cost --twoDimensionalCost --explicitStrategy --jsonOutput > '+ outfile,shell=True, stdout=subprocess.PIPE)
        sp.wait()

    #

    currentvisit, moveobstacles = simulateController.userControlled_partition_dist([outfile],gwg,pg,moveobstacles,allowed_states,invisibilityset,stations)
    print 'Reached station {}. Computing strategy for new stations'.format(currentvisit)
    stationnum+=1