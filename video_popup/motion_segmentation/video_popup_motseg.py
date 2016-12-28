""" video popup code

Code for the paper "Video Pop-up: Monocular 3D Reconstruction of
Dynamic Scenes"

"""

import numpy as np
import glob as glob

import video_popup_pb2
import persp_segmentation as ps

segmentaton_para = video_popup_pb2.SegmentationPara()
neighborhood_para = video_popup_pb2.NeighborhoodPara()

expr = 'kitti_rigid'
expr = 'kitti_dense_test'
expr = 'bird7'
expr = 'two_men'

downsampling = 1
saving_seg_result = 0

if(expr == 'two_men'):

    start_frame = 1
    end_frame = 30

    neighborhood_para.dist_threshold = 10000
    neighborhood_para.top_frames_num = 5
    neighborhood_para.max_occlusion_frames = 10
    neighborhood_para.occlusion_penalty = 0
    neighborhood_para.velocity_weight = 10

    segmentaton_para.images_path = '../../data/Two-men/*.png'
    images = sorted(glob.glob(segmentaton_para.images_path))[0: end_frame - start_frame + 1]
    segmentaton_para.min_vis_frames = 5

    model_fitting_para = segmentaton_para.model_fitting_para

    segmentaton_para.tracks_path = '../../data/Two-men/images/broxmalik_size4/broxmalikResults/broxmalikTracks30.dat'
    model_fitting_para.mdl = 2000
    model_fitting_para.graph_cut_para.pairwise_weight = 10
    model_fitting_para.graph_cut_para.pairwise_sigma = 0.5
    model_fitting_para.graph_cut_para.overlap_cost = 10
    model_fitting_para.graph_cut_para.engine = 0
    model_fitting_para.iters_num = 5

    # segmentaton_para.tracks_path = '../../data/Kitti/05/broxmalik_Size4/broxmalikResults/broxmalikTracks15.dat'
    # model_fitting_para.mdl = 2000
    # model_fitting_para.graph_cut_para.pairwise_weight = 10000
    # model_fitting_para.graph_cut_para.overlap_cost = 10
    # model_fitting_para.graph_cut_para.engine = 0
    # model_fitting_para.iters_num = 5

    # segmentaton_para.tracks_path = '../../data/Kitti/05/broxmalik_Size8/broxmalikResults/broxmalikTracks15.dat'
    # model_fitting_para.mdl = 2000
    # model_fitting_para.graph_cut_para.pairwise_weight = 10000
    # model_fitting_para.graph_cut_para.overlap_cost = 10
    # model_fitting_para.graph_cut_para.engine = 0
    # model_fitting_para.iters_num = 5

    downsampling = 4
    saving_seg_result = 1

elif(expr == 'kitti_dense_test'):

    start_frame = 1
    end_frame = 2

    # neighborhood_para.dist_threshold = np.inf
    neighborhood_para.dist_threshold = 100
    neighborhood_para.top_frames_num = 1
    neighborhood_para.max_occlusion_frames = 2
    neighborhood_para.occlusion_penalty = 0
    neighborhood_para.velocity_weight = 10

    segmentaton_para.images_path = '../../data/Kitti/05_2f/*.png'
    images = sorted(glob.glob(segmentaton_para.images_path))[0: end_frame- start_frame + 1]
    segmentaton_para.min_vis_frames = 1

    model_fitting_para = segmentaton_para.model_fitting_para

    segmentaton_para.tracks_path = '../../data/Kitti/05_2f/dense_flow/epicflowResults/epicflowTracks2.dat'

    model_fitting_para.mdl = 1000
    model_fitting_para.graph_cut_para.pairwise_weight = 10000
    model_fitting_para.graph_cut_para.overlap_cost = 10
    model_fitting_para.graph_cut_para.engine = 0
    model_fitting_para.iters_num = 5

    downsampling = 4
    saving_seg_result = 1

elif(expr == 'kitti_rigid'):

    start_frame = 1
    end_frame = 2

    # neighborhood_para.dist_threshold = np.inf
    neighborhood_para.dist_threshold = 100
    neighborhood_para.top_frames_num = 1
    neighborhood_para.max_occlusion_frames = 2
    neighborhood_para.occlusion_penalty = 0
    neighborhood_para.velocity_weight = 10

    segmentaton_para.images_path = '../../data/Kitti/05_rigid2/*.png'
    images = sorted(glob.glob(segmentaton_para.images_path))[0: end_frame- start_frame + 1]
    segmentaton_para.min_vis_frames = 1

    model_fitting_para = segmentaton_para.model_fitting_para

    segmentaton_para.tracks_path = '../../data/Kitti/05_rigid2/broxmalik_Size2/broxmalikResults/broxmalikTracks2.dat'
    model_fitting_para.mdl = 20000
    model_fitting_para.graph_cut_para.pairwise_weight = 3000
    model_fitting_para.graph_cut_para.overlap_cost = 10
    model_fitting_para.graph_cut_para.engine = 0
    model_fitting_para.iters_num = 5

    # segmentaton_para.tracks_path = '../../data/Kitti/05/broxmalik_Size4/broxmalikResults/broxmalikTracks15.dat'
    # model_fitting_para.mdl = 2000
    # model_fitting_para.graph_cut_para.pairwise_weight = 10000
    # model_fitting_para.graph_cut_para.overlap_cost = 10
    # model_fitting_para.graph_cut_para.engine = 0
    # model_fitting_para.iters_num = 5

    # segmentaton_para.tracks_path = '../../data/Kitti/05/broxmalik_Size8/broxmalikResults/broxmalikTracks15.dat'
    # model_fitting_para.mdl = 2000
    # model_fitting_para.graph_cut_para.pairwise_weight = 10000
    # model_fitting_para.graph_cut_para.overlap_cost = 10
    # model_fitting_para.graph_cut_para.engine = 0
    # model_fitting_para.iters_num = 5

    downsampling = 1
    saving_seg_result = 1

elif(expr == 'bird7'):

    start_frame = 1
    end_frame = 35

    neighborhood_para.dist_threshold = np.inf

    neighborhood_para.top_frames_num = 5
    neighborhood_para.max_occlusion_frames = 30
    neighborhood_para.occlusion_penalty = 0

    segmentaton_para.tracks_path = '../../data/Youtube_birds/bird_shot07/Tracks201.dat'
    segmentaton_para.images_path = '../../data/Youtube_birds/bird_shot07/images/*.jpg'

    images = sorted(glob.glob(segmentaton_para.images_path))[0: end_frame - start_frame + 1]

    segmentaton_para.min_vis_frames = 5

    model_fitting_para = segmentaton_para.model_fitting_para

    model_fitting_para.mdl = 1000
    model_fitting_para.graph_cut_para.pairwise_weight = 100
    model_fitting_para.graph_cut_para.overlap_cost = 0
    model_fitting_para.graph_cut_para.engine = 2

    model_fitting_para.graph_cut_para.pointwise_breaking_lambda = 1
    model_fitting_para.graph_cut_para.pointwise_outlier_lambda = 1

    model_fitting_para.iters_num = 5
    model_fitting_para.init_proposal_num = 300

    saving_seg_result = 1

print start_frame

ps.persp_segmentation(neighborhood_para, segmentaton_para, model_fitting_para, images, start_frame, end_frame,
                      saving_seg_result = saving_seg_result, downsampling = downsampling)