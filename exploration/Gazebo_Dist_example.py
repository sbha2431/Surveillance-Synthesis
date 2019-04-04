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
import pickle
from tqdm import *
import simulateController as Simulator
import itertools
import Control_Parser

def parseArguments():
    #### From --> https://stackoverflow.com/questions/28479543/run-python-script-with-some-of-the-argument-that-are-optional
    #### EVEN BETTER --> https://pymotw.com/2/argparse/
    # # Create argument parser
    parser = argparse.ArgumentParser()

    # # Positional mandatory arguments
    # parser.add_argument("SynthesisFlag", help="Include this boolean to run the synthesis", type=bool)

    #  # Optional arguments
    parser.add_argument("-synF", action='store_true', default=True, dest='synFlag',
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
    mapname = 'sanVisit_02'
    # mapname = 'chicago4_45_2454_5673_map'
    scale = (31,34)
    filename = ['figures/' + mapname + '.png',scale,cv2.INTER_AREA]

    image = cv2.imread(filename[0], cv2.IMREAD_GRAYSCALE)  # 0 if obstacle, 255 if free space
    image = cv2.resize(image, dsize=scale, interpolation=filename[2])
    # image[image<220] = 0
    # image[image >= 254] = 255
    h, w = image.shape[:2]

    ######################################
    folder_locn = 'Examples/'
    example_name = 'Example_Dist_Gazebo'
    trial_name = folder_locn + example_name
    slugs = '/home/sudab/Applications/slugs/src/slugs'
    outfile = trial_name + '.json'
    infile = copy.deepcopy(trial_name)
    gwfile = folder_locn + '/figs/gridworldfig_' + example_name + '.png'
    target_vis_file = trial_name + '.txt'
    nagents = 4
    targets = [[],[],[],[],[]]
    initial = [169,533,342,986]
    moveobstacles = [321]
    gwg = Gridworld(filename,nagents=nagents, targets=targets, initial=initial, moveobstacles=moveobstacles)
    gwg.colorstates = [set(), set()]
    gwg.render()
    gwg.draw_state_labels()
    gwg.save(gwfile)
    partition = dict()
    allowed_states = [[None]] * nagents
    pg = [[None]]*nagents
    #################### Agent allowed states ####################
    allowed_states[0] = set.union(*[set(range(162 + x * scale[0], 162 + x * scale[0] + 8)) for x in range(10)]).union({418}).union(*[set(range(109 + x * scale[0], 109 + x * scale[0] + 14)) for x in range(13)]).union(*[set(range(34 + x * scale[0], 34 + x * scale[0] + 27)) for x in range(2)])
    allowed_states[1] = set.union(*[set(range(625 + x * scale[0], 625 + x * scale[0] + 11)) for x in range(9)]).union({595}).union(*[set(range(499 + x * scale[0], 499 + x * scale[0] + 15)) for x in range(3)])
    allowed_states[2] = set.union(*[set(range(32 + x * scale[0], 32 + x * scale[0] + 2)) for x in range(32)])
    allowed_states[3] = set(range(h*w)) - allowed_states[0] - allowed_states[1]- allowed_states[2] - set(gwg.obstacles)

    ################### Single Partition ###################
    # partition[0] = range(h*w)
    pg[0] = {0:allowed_states[0]}
    pg[1] = {0:allowed_states[1]}
    pg[2] = {0: allowed_states[2]}
    pg[3] = {0:allowed_states[3]}
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

    visdist = [20,20,20,20,20]
    target_vis_dist = 2
    vel = [4,2,2,2]
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

    for n in [0,1,2,3,4]:
        obj = compute_all_vis.img2obj(image)
        # compute visibility for each state
        iset = compute_all_vis.compute_visibility_for_all(obj, h, w, radius=visdist[n])

        # iset = dict.fromkeys(set(gwg.states), frozenset({gwg.nrows * gwg.ncols + 1}))
        # for s in tqdm(set(gwg.states)):
        #     iset[s] = visibility.invis(gwg, s, visdist[n])
        #     if s in gwg.obstacles:
        #         iset[s] = {-1}
        # pickle_out = open("Examples/iset_gazeboexample.pickle", "wb")
        # pickle.dump(iset, pickle_out)
        # pickle_in = open("Examples/iset_gazeboexample.pickle", "rb")
        # iset = pickle.load(pickle_in)
        invisibilityset.append(iset)
        outfile = trial_name+'agent'+str(n)+'.json'
        filename.append(outfile)
        print 'output file: ', outfile
        print 'input file name:', infile
        # Do the synthesis if asked
        if args.synFlag:
            write_structured_slugs.write_to_slugs_imperfect_sensor(infile, gwg, initial[n], moveobstacles[0], iset,
                                                                   targets[n], vel[n], visdist[n], allowed_states[n],
                                                                   [],
                                                                   pg[n], belief_safety=0, belief_liveness=1,
                                                                   target_reachability=False,
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
    # simulator_with_orientation.userControlled_partition_dist(filename,gwg,pg,moveobstacles,allowed_states,invisibilityset,visset_target)
    # Control_Parser.parseJson(outfile, outfilename='Example2_perm_readable')
    # Control_Parser.parsePermissiveStrategy(outfile, outfilename='Example1_Perm_readable.json')
    Simulator.userControlled_imperfect_sensor(filename, gwg, pg, moveobstacles, allowed_states, invisibilityset,
                                              sensor_uncertain_dict, sensor_uncertainty)

