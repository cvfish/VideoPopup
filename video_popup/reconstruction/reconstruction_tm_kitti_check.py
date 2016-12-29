"""
Here we have three different visualizations for checking the result

show = 0: showing point cloud
show = 1: showing depth maps created by interpolating sparse point clouds
show = 2: use piecewise perspective reconstruction visualization app

In piecewise perspective reconstruction visualization app,  we could
where we could manually tune the scale/depth for each piece.

"""

import os
import numpy as np
import cPickle as pickle
import matplotlib.image as mpimg

from video_popup.utils import util
from video_popup.depth_reconstruction import depth_util

from skimage.segmentation import slic

expr = 'Kitti'
expr = 'two-men'

show = 2

def get_depth_map(W, Z, labels, images, vertices, points, K_persp, showFrames = 1, sequence='bird', save_seg = 0):

    W = W[:, points.astype(np.int32)]
    Z = Z[:, points.astype(np.int32)]

    labels_parts = labels[points.astype(np.int32)]
    image_files = images

    if(save_seg == 1):
        util.plot_traj2(W, Z, image_files, frame_step=1, labels=labels_parts, frame_time=0.1, plot_text=0,
                        save_folder='/results/'+sequence, save_fig=1)

    depth_maps = {}
    ref_images = {}

    numFrames = W.shape[0] / 2
    if(showFrames == 0):
        showFrames = numFrames

    vertices_new = vertices.reshape((numFrames, -1, 3))
    _, indices = np.unique(points, return_index=True)

    for f in range(showFrames):

        mask = Z[f,:]
        xyz = vertices_new[f, indices, :]
        xyz = xyz[mask]
        depths = np.dot(K_persp, xyz.T)
        uv = depths / depths[2]
        uv = uv[0:2,:]
        values = depths[2]
        if(np.mean(values) < 0):
            values = -values

        img = mpimg.imread(image_files[f])
        if(img.dtype.type is np.uint8 or img.dtype.type is np.uint16):
            img = img.astype(np.float32) / 255.0

        bg_mask = 0
        depth_map = util.dummy_image_interp(img, uv, values, bg_mask, method='nearest')

        ref_images[f] = img
        depth_maps[f] = depth_map

    return  depth_maps, ref_images

if expr == 'Kitti':

    nW = 1225
    nH = 369

    K_persp = np.array([[707.0912, 0,  601.8873],
                        [0,  707.0912, 183.1104],
                        [0,        0,       1]])

    seg_file = '../../data/Kitti/05/broxmalik_Size4/broxmalikResults/' \
               'f1t15/v5/vw10_nn10_k5_thresh10000_max_occ12_op0_cw2.5/init200/mdl2000_pw10000_oc10_engine0_it5/results.pkl'

    seg_path, file_name = os.path.split(seg_file)
    vispy_file = seg_path + '/OpenSfM/' + 'vispy_output.pkl'

    with open(vispy_file, 'r') as f:
        scene_reconstructions = pickle.load(f)

    scales = np.array([1, 3.5])

elif expr == 'two-men':

    nW = 960
    nH = 540

    K_persp = np.array([[1200, 0,  479.5],
                        [0, 1200,  269.5],
                        [0,    0,      1]])

    seg_file = '../../data/Two-men/images/broxmalik_size4/broxmalikResults/' \
               'f1t30/v5_d4/vw10_nn10_k5_thresh10000_max_occ10_op0_cw2.5/init200/mdl20_pw10_oc10_engine0_it5/results.pkl'

    seg_path, file_name = os.path.split(seg_file)
    vispy_file = seg_path + '/OpenSfM/' + 'vispy_output.pkl'

    with open(vispy_file, 'r') as f:
        scene_reconstructions = pickle.load(f)

    scales = np.array([2, 1, 0.75])


reconstruct = {}
reconstruct['rotations3d'] = scene_reconstructions['rotations']
reconstruct['translations3d'] = scene_reconstructions['translations']
reconstruct['shapes'] = scene_reconstructions['shapes']
reconstruct['colors'] = scene_reconstructions['colors']
reconstruct['points'] = scene_reconstructions['points']

reconstruct['W'] = scene_reconstructions['W']
reconstruct['Z'] = scene_reconstructions['Z']
reconstruct['labels'] = scene_reconstructions['labels']
reconstruct['images'] = scene_reconstructions['images']

# K = np.array([[800, 0, 640],[0, 800, 360],[0,0,1]])
K = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])

## test
F = reconstruct['rotations3d'][0].shape[0] / 3
vertices = np.array([]).reshape(0,3*F)
colors = np.array([]).reshape(0,3)
labels = np.array([]).reshape(0,1)
points = np.array([])

for i in range(len(reconstruct['rotations3d'])):

    vertices_i = np.dot(reconstruct['rotations3d'][i], reconstruct['shapes'][i]) + \
                 reconstruct['translations3d'][i].reshape((-1,1))

    colors_i = reconstruct['colors'][i]
    points_i = reconstruct['points'][i][0]

    vertices_i = vertices_i.T

    vertices = np.concatenate((vertices, scales[i] * vertices_i))

    num_pnts = vertices_i.shape[0]

    colors = np.concatenate((colors, colors_i.T))
    points = np.concatenate((points, points_i))

    labels = np.concatenate((labels, np.ones((num_pnts, 1))*i ))

num_pnts = vertices.shape[0]
vertices = np.reshape(vertices, (num_pnts, -1, 3))
vertices = np.swapaxes(vertices, 0, 1)
vertices = vertices.reshape((-1, 3))

if(show == 0):
    depth_util.point_cloud_plot(vertices, colors, K_persp, nH, nW, labels=labels, nframes = F)
elif(show == 1):
    """
    visualization of depth maps, press q to save rendering results
    """
    depth_maps, ref_images = get_depth_map(reconstruct['W'],
                                           reconstruct['Z'],
                                           reconstruct['labels'],
                                           reconstruct['images'],
                                           vertices, points, K_persp,
                                           showFrames=1, sequence=expr)

    depth_util.depth_maps_plot(depth_maps, ref_images, K_persp)
elif(show == 2):
    """ perspective visualization """
    import sys
    from video_popup.visualization import  qt_app_persp
    from PyQt4 import QtGui

    appQt = QtGui.QApplication(sys.argv)
    win = qt_app_persp.MainWindow(scene_reconstructions, reconstruct['images'], K_persp, False)
    win.show()
    appQt.exec_()