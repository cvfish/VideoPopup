#import json
import cPickle as pickle
import os
import sys

# from time import sleep
# from scipy.sparse import csgraph
# from scipy.sparse import csr_matrix

import scipy.io
import numpy as np
import logging

# import matplotlib
# matplotlib.use('Agg')

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import video_popup_pb2
sys.path.append('../../libs/segmentation_rui_new')

from video_popup.utils import util
from model_fitting import ModelFitting

segmentaton_para = video_popup_pb2.SegmentationPara()
neighborhood_para = video_popup_pb2.NeighborhoodPara()

def persp_segmentation(neighborhood_para, segmentaton_para, model_fitting_para, images, start_frame,
                       end_frame, saving_seg_result = 0, downsampling = 1, min_tracks = 5000):

    #loading trajectories

    if(downsampling > 1):
        tracks_file = os.path.dirname( segmentaton_para.tracks_path ) + \
            '/f{:d}t{:d}/v{:d}_d{:d}/data.pkl'.format(start_frame, end_frame,
                                                      segmentaton_para.min_vis_frames,
                                                      downsampling)
    else:
        tracks_file = os.path.dirname( segmentaton_para.tracks_path ) + \
                '/f{:d}t{:d}/v{:d}/data.pkl'.format(start_frame, end_frame,
                                                    segmentaton_para.min_vis_frames)

    try:
        util.ensure_dir(tracks_file)
        with open(tracks_file, 'r') as f:
            W,labels,Z = pickle.load(f)
    except:
        W, labels, Z = util.load_trajectory(segmentaton_para.tracks_path,
                                       start_frame, end_frame,
                                       segmentaton_para.min_vis_frames)

        mask = (W[0,:] % downsampling == 0) * (W[1,:] % downsampling == 0)

        W = W[:,mask]
        labels = labels[mask]
        Z = Z[:,mask]

        W = np.ascontiguousarray(W, dtype=np.float64)
        labels = np.ascontiguousarray(labels, dtype=np.int32)
        Z = np.ascontiguousarray(Z, dtype=np.bool)

        with open(tracks_file, 'w') as f:
            pickle.dump((W,labels,Z), f, True)
            # data = {'W':W, 'labels':labels, 'Z':Z}
            # pickle.dump(data, f, True)

    #get neighbors

    logging.info("number of point tracks {:d}".format(W.shape[1]))

    neighbors_file = os.path.dirname(tracks_file) + '/vw{:g}_nn{:g}_k{:g}_thresh{:g}_max_occ{:g}_op{:g}_cw{:g}/nbor.pkl'.format(
        neighborhood_para.velocity_weight,
        neighborhood_para.neighbor_num,
        neighborhood_para.top_frames_num,
        neighborhood_para.dist_threshold,
        neighborhood_para.max_occlusion_frames,
        neighborhood_para.occlusion_penalty,
        neighborhood_para.color_weight
    )

    # save data to speed up
    try:
        util.ensure_dir(neighbors_file)
        with open(neighbors_file, 'r') as f:
            M, s = pickle.load(f)
    except:
        M, s = util.get_nbor(W, Z, neighborhood_para)
        with open(neighbors_file, 'w') as f:
            pickle.dump((M,s), f, True)

    results_folder = os.path.dirname(neighbors_file) + '/init{:g}/mdl{:g}_pw{:g}_oc{:g}_engine{:g}_it{:g}/'.format(
        model_fitting_para.init_proposal_num,
        model_fitting_para.mdl,
        model_fitting_para.graph_cut_para.pairwise_weight,
        model_fitting_para.graph_cut_para.overlap_cost,
        model_fitting_para.graph_cut_para.engine,
        model_fitting_para.iters_num
    )

    try:

        with open('{:s}/results.pkl'.format(results_folder), "r") as f:

            seg = pickle.load(f)

            W = seg['W']
            Z = seg['Z']
            labels = seg['labels_objects']
            images = seg['images']

            # plot the segmentation result
            # util.plot_nbor(seg['W'], seg['Z'], seg['s'], seg['images'], labels=seg['labels_objects'], frame_time = 0.1,
            #                show_overlap=1, show_broken=1, plot_text=1, assignment=seg['assignment'])

            return results_folder

    except:

        # plot_nbor(W, Z, M, s, images, labels = labels)

        # simplify stuff and just use the neighborhood distance to set up parameters for graphcut

        if(W.shape[1] < min_tracks):
            return results_folder

        data = (W, Z, M, s, images, labels)

        # model_fitting_para = segmentaton_para.model_fitting_para
        # model_fitting_para.mdl = 100
        # model_fitting_para.graph_cut_para.pairwise_weight = 10000
        # model_fitting_para.graph_cut_para.overlap_cost = 100
        # model_fitting_para.graph_cut_para.engine = 0
        # model_fitting_para.iters_num = 3

        options = {}
        options['show_iteration'] = 1
        options['save_path']= os.path.dirname(neighbors_file)
        model_fitting = ModelFitting(data, model_fitting_para, options)

        model_fitting.init_proposals()
        assignment, labels, outliers, inliers = model_fitting.run()

        # plot result
        # util.plot_nbor(W, Z, s, images, labels=labels, show_edge=0, show_overlap=1, show_broken=1, assignment=assignment)

        # delete outliers or not
        # labels_objects, W_new, Z_new, assignment_new, s_new, labels_parts = util.merge_parts_to_objects(W, Z, s, labels, assignment, thresh=50)
        labels_objects, W_new, Z_new, assignment_new, s_new, labels_parts = util.merge_parts_to_objects(W, Z, s, labels, assignment,
                                                                                                        thresh=50, outliers=outliers)

        labels = np.copy(labels_parts)

        # util.plot_nbor(W_new, Z_new, s_new, images, labels_parts)
        # util.plot_nbor(W_new, Z_new, s_new, images, labels_objects)

        # start doing reconstruction
        # according to labels_objects, break labels_new into different connected object components
        labels_parts, assignment_new = util.break_parts(labels_parts, labels_objects, s_new,
                                                   lambda_weight = model_fitting_para.graph_cut_para.lambda_weight)

        # data = (W_new, Z_new, s_new, assignment_new, images, labels_objects, labels_parts)
        data = {'W':W_new, 'Z':Z_new, 's':s_new, 'assignment': assignment_new, 'labels': labels,
                'images': images, 'labels_objects': labels_objects, 'labels_parts': labels_parts}

        if(saving_seg_result):

            results_folder = os.path.dirname(neighbors_file) + '/init{:g}/mdl{:g}_pw{:g}_oc{:g}_engine{:g}_it{:g}/'.format(
                model_fitting_para.init_proposal_num,
                model_fitting_para.mdl,
                model_fitting_para.graph_cut_para.pairwise_weight,
                model_fitting_para.graph_cut_para.overlap_cost,
                model_fitting_para.graph_cut_para.engine,
                model_fitting_para.iters_num
            )

            try:
                util.ensure_dir(results_folder)
                with open('{:s}/neighbor.pro'.format(results_folder), "wb") as f:
                    f.write(neighborhood_para.SerializeToString())
                with open('{:s}/seg.pro'.format(results_folder), "wb") as f:
                    f.write(segmentaton_para.SerializeToString())
                with open('{:s}/results.pkl'.format(results_folder),"wb") as f:
                    pickle.dump(data, f, True)
                with open('{:s}/results.mat'.format(results_folder),"wb") as f:
                    scipy.io.savemat(f, mdict=data)
            except:
                print "saving error"

            # util.plot_traj2(W_new, Z_new, images, labels=labels, save_fig=1, frame_time = 0.1,
            #                 frame_step=2, save_folder=results_folder, save_name='labels.png')
            # util.plot_traj2(W_new, Z_new, images, labels=labels_parts, save_fig=1, frame_time = 0.1,
            #                 frame_step=2, save_folder=results_folder, save_name='parts.png')
            # util.plot_traj2(W_new, Z_new, images, labels=labels_objects, save_fig=1, frame_time = 0.1,
            #                 frame_step=2, save_folder=results_folder, save_name='objects.png')

            return results_folder