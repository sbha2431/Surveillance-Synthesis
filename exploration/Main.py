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
# from tensorpack import PredictConfig
# from tensorpack import OfflinePredictor
# from tensorpack import SaverRestore
# from tensorpack import InputDesc
# import tensorflow as tf
# import compute_possible_stations
import Control_Parser

model_path = 'log/checkpoint'
mapname = 'unnamed'
# mapname = 'chicago4_45_2454_5673_map'
filename = 'figures/'+mapname+'.png'

image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)  # 0 if obstacle, 255 if free space
image = cv2.resize(image,dsize=(8,8),interpolation=cv2.INTER_AREA)
h, w = image.shape[:2]
obj = compute_all_vis.img2obj(image)
visdist = [1000] # Visibility range for agent
np.random.seed(0)
initial = [0] # Initial agent position
moveobstacles = [6]
visited = []
currentvisit = initial[0]
while True:
    # stationnum = 0
    # visited.append(currentvisit)
    # print 'Visited stations ', visited
    # # initialize the model
    # predict_func = OfflinePredictor(
    #         PredictConfig(
    #             inputs_desc=[InputDesc(tf.float32, [None, 128, 128, 2], 'input_image')],
    #             tower_func=model.feedforward,
    #             session_init=SaverRestore(model_path),
    #             input_names=['input_image'],
    #             output_names=['prob']))
    # min_distance = 1  # minimum distance between each station (smaller means more locations are returned)
    # stations = compute_possible_stations.get_stations(obj, h, w, predict_func, visited, visdist[0], min_distance)
    # print stations

     #Initial state of moving obstacle(s)
    gwg = Gridworld(filename,targets=[[]],initial=initial,moveobstacles=moveobstacles) #Generate gridworld class
    gwg.render()
    gwg.save('Example1.png')
    # gwg.draw_state_labels()
    slugs = '/home/sudab/Applications/slugs/src/slugs' # Path to slugs executable



    ##Defining allowed states for agents to move and abstract belief states
    allowed_states = [None]
    allowed_states[0] = list(set(range(gwg.nstates)) - set(gwg.obstacles))
    fullvis_states = [[]] #States where we have full visibility at all times
    partitionGrid = dict()
    partitionGrid[(0,0)] = set(range(h*w)) #Number of belief states - more states -> more refined but slower synthesis
    pg = [dict.fromkeys((0,0),partitionGrid[(0,0)])]


    vel = [1] #No of states the agent can move in one timestep
    print 'Writing input file...'
    invisibilityset = []
    outfile = 'Example1_'+'perm'+'.json'
    infile = 'Example1_perm'
    print 'output file: ', outfile
    print 'input file name:', infile

    h, w = image.shape[:2]
    for n in range(gwg.nagents):
        obj = compute_all_vis.img2obj(image)
        # compute visibility for each state
        iset = compute_all_vis.compute_visibility_for_all(obj, h, w,radius=visdist[n])
        # Salty_input.write_to_slugs_part_dist(infile,gwg,currentvisit,moveobstacles[n],iset,[],vel[n],visdist[n],
        #                                             allowed_states[n],fullvis_states[n], pg[n], belief_safety = 1, belief_liveness =0, target_reachability = False) #Write input file
        # Salty_input.write_to_slugs_imperfect_sensor(infile,gwg,currentvisit,moveobstacles[n],iset,[],vel=vel[n],visdist = 5,allowed_states = allowed_states[n],
        #                                     fullvis_states = [],partitionGrid =pg[n], belief_safety = 1, belief_liveness = 0, target_reachability = False,sensor_uncertainty=1,sensor_uncertain_dict = dict())
        Salty_input.write_to_slugs_part_dist_J(infile,gwg,currentvisit,moveobstacles[n],iset,[],[],vel=1,visdist = 5,allowed_states = allowed_states[n],
                                fullvis_states = [],partitionGrid =dict(), belief_safety = 1, belief_liveness = 0, target_reachability = False,
                                target_has_vision = False, target_vision_dist = 1.1, filename_target_vis = None, compute_vis_flag = False)
        invisibilityset.append(iset)
        print ('Converting input file...')
        os.system('python compiler.py ' + infile + '.structuredslugs > ' + infile + '.slugsin')
        print('Computing controller...')
        sp = subprocess.Popen(slugs + ' --explicitStrategy --jsonOutput ' + infile + '.slugsin > '+ outfile,shell=True, stdout=subprocess.PIPE)
        sp.wait()
    break
    #
Control_Parser.parseJson(outfile,outfilename='Example1_perm_readable')
    # currentvisit, moveobstacles = simulateController.userControlled_partition_dist([outfile],gwg,pg,moveobstacles,allowed_states,invisibilityset,[])
    # print 'Reached station {}. Computing strategy for new stations'.format(currentvisit)
    # stationnum+=1