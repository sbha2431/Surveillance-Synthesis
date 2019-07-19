from gridworld import *
import argparse
# import write_structured_slugs_copy_2
import write_structured_slugs
import compute_all_vis
import cv2
import visibility
import os
import subprocess
import time
import copy
import cPickle as pickle
from tqdm import *
import simulateController as Simulator
import itertools
import Control_Parser
import json
def parseArguments():
    #### From --> https://stackoverflow.com/questions/28479543/run-python-script-with-some-of-the-argument-that-are-optional
    #### EVEN BETTER --> https://pymotw.com/2/argparse/
    # # Create argument parser
    parser = argparse.ArgumentParser()

    # # Positional mandatory arguments
    # parser.add_argument("SynthesisFlag", help="Include this boolean to run the synthesis", type=bool)

    #  # Optional arguments
    parser.add_argument("-synF", action='store_true', default=False, dest='synFlag',
                        help="Include this boolean to run the synthesis")
    parser.add_argument("-cvF", action='store_true', default=False, dest='visFlag',
                        help="Include this boolean to compute belief visibility of the target")
    parser.add_argument("-noVisF", action='store_false', default=True, dest='noVisFlag',
                        help="Include this boolean to run synthesis without any target vision")

    # # Parse arguments
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parseArguments()
    print 'Synthesis Flag is: ', args.synFlag
    if not args.noVisFlag:
        print '--> Target has No vision'
    if args.noVisFlag and args.synFlag:
        print 'Compute Vision Flag is: ', args.visFlag

    then = time.time()
    # Make sure to state the agent and the target far enough from each other such that the games initial conditions do not violate safety.
    mapname = '3ne'
    # mapname = 'chicago4_45_2454_5673_map'
    scale = (int(80*2.6),80)
    filename = ['figures/' + mapname + '.pgm',scale,cv2.INTER_LINEAR_EXACT]

    image = cv2.imread(filename[0], cv2.IMREAD_GRAYSCALE)  # 0 if obstacle, 255 if free space
    image = cv2.resize(image, dsize=scale, interpolation=filename[2])
    # image[image<220] = 0
    # image[image >= 254] = 255
    h, w = image.shape[:2]

    ######################################
    folder_locn = 'Examples/'
    example_name = 'Example_Dist_Gazebo_3_liveness'
    trial_name = folder_locn + example_name
    version = '01'
    slugs = '/home/sudab/Applications/slugs/src/slugs'
    save_to_Gazebo = True
    outfile = trial_name + '.json'
    infile = copy.deepcopy(trial_name)
    gwfile = folder_locn + '/figs/gridworldfig_' + example_name + '.png'
    target_vis_file = trial_name + '.txt'
    nagents = 2
    targets = [[857],[634],[],[],[]]
    initial = [1103,797]
    moveobstacles = [1140]
    gwg = Gridworld(filename,nagents=nagents, targets=targets, initial=initial, moveobstacles=moveobstacles)
    gwg.colorstates = [set(), set()]
    gwg.render()
    # gwg.draw_state_labels()
    gwg.save(gwfile)
    partition = dict()
    allowed_states = [[None]] * nagents
    pg = [[None]]*nagents
    #################### Agent allowed states ####################
    allowed_states[0] = set.union(*[set(range(586 + x * scale[0], 586 + x * scale[0] + 14)) for x in range(25)]).union(*[set(range(961 + x * scale[0], 961 + x * scale[0] + 25)) for x in range(15)])  - set(gwg.obstacles)
    allowed_states[1] = set(gwg.states) - allowed_states[0] - set(gwg.obstacles)
    ################### Single Partition ###################
    pg[0] = {0: set.union(*[set(range(586 + x * scale[0], 586 + x * scale[0] + 14)) for x in range(11)])  - set(gwg.obstacles), 1: set.union(*[set(range(1026 + x * scale[0], 1026 + x * scale[0] + 14)) for x in range(14)])  - set(gwg.obstacles),
             2: set.union(*[set(range(961 + x * scale[0], 961 + x * scale[0] + 12)) for x in range(15)])  - set(gwg.obstacles), 3: set.union(*[set(range(973 + x * scale[0], 973 + x * scale[0] + 13)) for x in range(15)])  - set(gwg.obstacles)}
    pg[1] = {0: set.union(*[set(range(361 + x * scale[0], 361 + x * scale[0] + 14)) for x in range(14)])  - set(gwg.obstacles), 1: set.union(*[set(range(335 + x * scale[0], 335 + x * scale[0] + 11)) for x in range(16)])  - set(gwg.obstacles),
             2: set.union(*[set(range(66 + x * scale[0], 66 + x * scale[0] + 13)) for x in range(12)])  - set(gwg.obstacles), 3: set.union(*[set(range(52 + x * scale[0], 52 + x * scale[0] + 14)) for x in range(7)])  - set(gwg.obstacles),
             4: set.union(*[set(range(41 + x * scale[0], 41 + x * scale[0] + 11)) for x in range(7)]) - set(gwg.obstacles)}
    # pg[2] = {0: allowed_states[2]}
    # pg[3] = {0:allowed_states[3]}
    #########################################################

    # partition[0] = set.union(*[set(range(162 + x * scale[0], 162 + x * scale[0] + 8)) for x in range(10)]).union({325,356,387,418})
    # partition[1] = set.union(*[set(range(625 + x * scale[0], 625 + x * scale[0] + 11)) for x in range(9)]).union({595,596,597,598})
    # partition[2] = set.union(*[set(range(32 + x * scale[0], 32 + x * scale[0] + 2)) for x in range(32)])
    # partition[3] = set.union(*[set(range(34 + x * scale[0], 34 + x * scale[0] + 27)) for x in range(2)])
    # partition[4] = set.union(*[set(range(964 + x * scale[0], 964 + x * scale[0] + 27)) for x in range(2)])
    # partition[5] = set.union(*[set(range(109 + x * scale[0], 109 + x * scale[0] + 14)) for x in range(13)])
    # partition[6] = set.union(*[set(range(607 + x * scale[0], 607 + x * scale[0] + 12)) for x in range(12)])
    # partition[7] = set(range(h*w)) - partition[0] - partition[1] - partition[2] - partition[3]- partition[4] - partition[5] - partition[6]
    # partition[0] = set(range(nrows*ncols))

    visdist = [10,20,3500,3500]
    target_vis_dist = 2
    vel = [2,2,2,2]
    invisibilityset = []
    sensor_uncertainty = 1
    filename = []
     ######################### Create sensor uncertainty dictionary #############################
    belief_ncols = gwg.ncols - sensor_uncertainty + 1
    belief_nrows = gwg.nrows - sensor_uncertainty + 1
    sensor_uncertain_dict = dict.fromkeys(range(belief_ncols * belief_nrows))
    for i in range(belief_nrows):
        for j in range(belief_ncols):
            belief_gridstate = i * belief_ncols + j
            sensor_uncertain_dict[belief_gridstate] = set()
            for srow in range(i, i + sensor_uncertainty):
                for scol in range(j, j + sensor_uncertainty):
                    gridstate = gwg.rcoords((srow, scol))
                    uset = list(itertools.product(['E', 'S', 'R'], repeat=sensor_uncertainty - 1))
                    for u in uset:
                        snext = copy.deepcopy(i * gwg.ncols + j)
                        for v in range(sensor_uncertainty - 1):
                            act = u[v]
                            snext = np.nonzero(gwg.prob[act][snext])[0][0]
                        # if gridstate not in iset[belief_gridstate]:
                        sensor_uncertain_dict[belief_gridstate].add(snext)
                    # sensor_uncertain_dict[belief_gridstate].add(gridstate)
    ##############################################################################################

    for n in [0,1]:
        obj = compute_all_vis.img2obj(image)
        # compute visibility for each state
        iset = compute_all_vis.compute_visibility_for_all(obj, h, w, radius=visdist[n])

        # iset = dict.fromkeys(set(gwg.states), frozenset({gwg.nrows * gwg.ncols + 1}))
        # for s in tqdm(set(gwg.states)):
        #     iset[s] = visibility.invis(gwg, s, visdist[n])
        #     if s in gwg.obstacles:
        #         iset[s] = {-1}
        # pickle_out = open("Examples/iset_gazeboexample_2.pickle", "wb")
        # pickle.dump(iset, pickle_out)
        # pickle_in = open("Examples/iset_gazeboexample.pickle", "rb")
        # iset = pickle.load(pickle_in)
        invisibilityset.append(iset)
        outfile = trial_name+'agent'+str(n) +'.json'
        filename.append(outfile)
        print 'output file: ', outfile
        print 'input file name:', infile
        # Do the synthesis if asked
        if args.synFlag:
            write_structured_slugs.write_to_slugs_imperfect_sensor(infile, gwg, initial[n], moveobstacles[0], iset,
                                                                   targets[n], vel[n], visdist[n], allowed_states[n],
                                                                   [],
                                                                   pg[n], belief_safety=0, belief_liveness=10,
                                                                   target_reachability=True,
                                                                   sensor_uncertainty=sensor_uncertainty,
                                                                   sensor_uncertain_dict=sensor_uncertain_dict)

            # write_structured_slugs.write_to_slugs_part_dist(infile, gwg, initial[n], moveobstacles[0], iset, [], targets[n], vel[n],
            #                          visdist[n], allowed_states[n],[], pg[n], belief_safety=1, belief_liveness=0,
            #                          target_reachability=False,
            #                          target_has_vision=False, target_vision_dist=1.1, filename_target_vis=None,
            #                          compute_vis_flag=False)

            print ('Converting input file...')
            os.system('python compiler.py ' + infile + '.structuredslugs > ' + infile + '.slugsin')
            print('Computing controller...')
            sp = subprocess.Popen(slugs + ' --explicitStrategy --jsonOutput ' + infile + '.slugsin > ' + outfile,
                                  shell=True, stdout=subprocess.PIPE)
            # sp = subprocess.Popen(slugs + ' --extractExplicitPermissiveStrategy ' + infile + '.slugsin > ' + outfile,
            #                       shell=True, stdout=subprocess.PIPE)
            sp.wait()

    # automaton = write_structured_slugs.parseJson(outfile)
    # print(automaton)

    now = time.time()
    print('Synthesis took ', now - then, ' seconds')

    #### Save for gazebo
    if save_to_Gazebo:
        isetlist = dict()
        for s1 in iset.keys():
            isetlist[s1] = copy.deepcopy(list(iset[s1]))
        # j = json.dumps(isetlist, indent=1)
        # f = open(folder_locn + 'GazeboFiles/' + 'iset_' + version +'.json', "wb")
        # print >> f, j
        # f.close()
        with open(folder_locn + 'GazeboFiles/' + 'iset_' + version +'.json', "wb") as fp:
            json.dump(isetlist, fp)

        allowedstates = dict()
        for n in range(gwg.nagents):
            keyname = 'uav'+str(n+1)
            allowedstates[keyname] = copy.deepcopy(list(allowed_states[n]))

        with open(folder_locn + 'GazeboFiles/' + 'allowedstates_' + version +'.json', "wb") as fp:
            json.dump(allowedstates, fp)
        gwg.save(folder_locn+'GazeboFiles/gridfig'+version)

    #### Run Simulation
    Simulator.userControlled_imperfect_sensor(filename, gwg, pg, moveobstacles, allowed_states, invisibilityset,
                                              sensor_uncertain_dict, sensor_uncertainty,saveImage=folder_locn+'GazeboFiles/livenessExperiment_1agent/trajectory_figs/t-')



