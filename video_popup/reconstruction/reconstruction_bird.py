### test reconstruction part

import numpy as np
import glob as glob

import scipy.io
import cPickle as pickle
import matplotlib.image as mpimg

from video_popup.utils import util

from video_popup.motion_segmentation import video_popup_pb2
import reconstruction

expr = 'bird7'

reconstruction_para = video_popup_pb2.ReconstructionPara()

if(expr == 'bird7'):

    """
    try new stuff on bird sequence shot 07
    """

    seg_file = '../../data/Youtube_birds/bird_shot07/f1t35/v5/' \
               'vw2000_nn10_k5_threshinf_max_occ30_op0_cw2.5/init300/mdl1000_pw100_oc0_engine2_it5/results.pkl'

    with open(seg_file, 'r') as f:
        seg = pickle.load(f)

    W_new = seg['W']
    Z_new = seg['Z']
    s_new = seg['s']
    assignment_new = seg['assignment']
    image_files = seg['images']
    labels = seg['labels']
    labels_parts = seg['labels_parts']
    labels_objects = seg['labels_objects']

    try_list = [0]

    data_path = '../../data/Youtube_birds/bird_shot07/f1t35/v5/' \
                'vw2000_nn10_k5_threshinf_max_occ30_op0_cw2.5/init300/mdl1000_pw100_oc0_engine2_it5'

# util.plot_nbor(W_new, Z_new, s_new, image_files, labels_parts)
# util.plot_nbor(W_new, Z_new, s_new, image_files, labels_objects)

# labels_parts, assignment_new = util.break_parts(labels_parts, labels_objects, s_new, lambda_weight = 1)
# util.plot_nbor(W_new, Z_new, s_new, image_files, labels=labels,
#                assignment=assignment_new, show_overlap=1, show_broken=1)

# util.plot_nbor(W_new, Z_new, s_new, image_files, labels_objects)
# util.plot_nbor(W_new, Z_new, s_new, image_files, labels_parts)

# start doing reconstruction
# according to labels_objects, break labels_new into different connected object components
data = {'W':W_new, 'Z':Z_new, 's':s_new, 'assignment': assignment_new, 'labels': labels,
        'images': image_files, 'labels_objects': labels_objects, 'labels_parts': labels_parts}
data['colors'] = util.retrieve_colors(data['W'], data['Z'], data['images'])

options = {}
options['save_path'] = data_path
options['try_list'] = try_list

reconstruction = reconstruction.Reconstruction(data, reconstruction_para, options)

reconstruction.run()

