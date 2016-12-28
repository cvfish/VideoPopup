import os
import numpy as np
import cPickle as pickle

import logging

#from video_popup.utils import util
from depth_reconstruction import DepthReconstruction

expr = 'kitti_rigid'
#expr = 'kitti_dense'

if(expr == 'kitti_rigid'):

    print expr

    seg_file = '../../data/Kitti/05_rigid2/broxmalik_Size2/broxmalikResults/f1t2/v1/' \
               'vw10_nn10_k1_thresh100_max_occ2_op0_cw2.5/init200/mdl20000_pw3000_oc10_engine0_it5/results.pkl'

    bin_gt_file = '../../data/Kitti/05_rigid2/broxmalik_Size2/002648.bin'

    K = np.array([[707.0912, 0,  601.8873],
                  [0,  707.0912, 183.1104],
                  [0,        0,       1]])

    with open(seg_file, 'r') as f:
        seg = pickle.load(f)

    Z = seg['Z']
    mask = np.logical_and(Z[0], Z[1])

    W = seg['W'][0:4,mask]
    Z = seg['Z'][0:2,mask]
    # labels = seg['labels_objects'][mask]
    # labels = seg['labels_parts'][mask]
    labels = seg['labels'][mask]
    images = seg['images'][0:2]

    # plot the segmentation result
    # util.plot_nbor(seg['W'], seg['Z'], seg['s'], seg['images'], seg['labels_objects'])
    # util.plot_nbor(seg['W'], seg['Z'], seg['s'], seg['images'], seg['labels_parts'])

    data = (W, Z, labels, K, images)

    # parameters
    # num_segments
    # lambda_reg, kappa, gamma

    para = {}
    para['seg_folder'] = os.path.dirname(seg_file)
    para['num_segments'] = 5000
    para['lambda_reg_list'] = [100]
    para['kappa_list'] = [1]
    para['gamma_list'] = [1]

    para['has_gt'] = 1
    para['expr'] = 'kitti'

    Tr = np.array([[-0.001857739385241,  -0.999965951351000,  -0.008039975204516,  -0.004784029760483],
                   [-0.006481465826011,   0.008051860151134,  -0.999946608177400,  -0.073374294642310],
                   [0.999977309828700,  -0.001805528627661,  -0.006496203536139,  -0.333996806443300]])

    para['Tr'] = Tr
    para['bin_gt_file'] = bin_gt_file

    logging.basicConfig(filename=para['seg_folder'] + '/record.log', level=logging.DEBUG,
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    depth_map_recons = DepthReconstruction(data, para)

    depth_map_recons.run()

elif(expr == 'kitti_dense'):

    seg_file = '../../data/Kitti/05_2f/dense_flow/epicflowResults/f1t2/v1_d4/' \
               'vw10_nn10_k1_thresh100_max_occ2_op0_cw2.5/init200/mdl1000_pw10000_oc10_engine0_it5/results.pkl'

    K = np.array([[707.0912, 0,  601.8873],
                  [0,  707.0912, 183.1104],
                  [0,        0,       1]])

    with open(seg_file, 'r') as f:
        seg = pickle.load(f)

    Z = seg['Z']
    mask = np.logical_and(Z[0], Z[1])

    W = seg['W'][0:4,mask]
    Z = seg['Z'][0:2,mask]
    labels = seg['labels'][mask]
    images = seg['images'][0:2]

    # plot the segmentation result
    # util.plot_nbor(seg['W'], seg['Z'], seg['s'], seg['images'], seg['labels_objects'])
    # util.plot_nbor(seg['W'], seg['Z'], seg['s'], seg['images'], seg['labels_parts'])

    data = (W, Z, labels, K, images)

    # parameters
    # num_segments
    # lambda_reg, kappa, gamma

    para = {}
    para['seg_folder'] = os.path.dirname(seg_file)
    para['num_segments'] = 5000
    para['lambda_reg_list'] = [0]
    para['kappa_list'] = [1]
    para['gamma_list'] = [1]
    para['lambda_reg2_list'] = [5]
    para['lambda_constr_list'] = [10000]

    para['has_gt'] = 1
    para['expr'] = 'kitti'

    Tr = np.array([[-0.001857739385241,  -0.999965951351000,  -0.008039975204516,  -0.004784029760483],
                   [-0.006481465826011,   0.008051860151134,  -0.999946608177400,  -0.073374294642310],
                   [0.999977309828700,  -0.001805528627661,  -0.006496203536139,  -0.333996806443300]])

    para['Tr'] = Tr
    para['bin_gt_file'] = '../../data/Kitti/05_2f/002491.bin'

    logging.basicConfig(filename=para['seg_folder'] + '/record.log', level=logging.DEBUG,
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    depth_map_recons = DepthReconstruction(data, para)

    depth_map_recons.run()

