"""
create visualization app for piecewise orthographic reconstruction results,
where we could manually tune the flipping and translation along depth direction

Basically, this is very similar to the perspective reconstruction visualization app.
Maybe it is better to write a single app which could handle both orthographic and perspective
reconstruction results, but I decide to write for each separately

To handle the ambiguity we have in orthographic cases,
we have per object flipping ,per frame flipping and also per frame depth ambiguity

we fix the camera, and pick fixed intrinsic camera parameters just for visualization

"""

import os
import numpy as np
import cPickle as pickle
import matplotlib.image as mpimg

from video_popup.utils import util
from video_popup.depth_reconstruction import depth_util

from skimage.segmentation import slic

expr = 'bird7'

flip = 0
show = 0
perspective_result = False

def get_depth_map(seg_file, vertices, points, showFrames = 1, sequence='bird', save_seg = 0):

    with open(seg_file, 'r') as f:
        seg = pickle.load(f)

    W = seg['W']
    Z = seg['Z']
    labels_parts = seg['labels_parts']
    labels_objects = seg['labels_objects']

    image_files = seg['images']

    if(save_seg == 1):
        util.plot_traj(W, Z, image_files, frame_step=1, labels=labels_parts, frame_time=0.1, plot_text=0,
                       save_folder='/results/'+sequence, save_fig=1)
        util.plot_traj(W, Z, image_files, frame_step=1, labels=labels_objects, frame_time=0.1, plot_text=0,
                       save_folder='/results/'+sequence+'_object', save_fig=1)

    depth_maps = {}
    ref_images = {}

    numFrames = W.shape[0] / 2
    if(showFrames == 0):
        showFrames = numFrames

    vertices_new = vertices.reshape((numFrames, -1, 3))
    _, indices = np.unique(points, return_index=True)

    for f in range(showFrames):

        mask = Z[f,:]
        Wf = W[2*f:2*f+2, mask]
        uv = Wf.reshape((2, -1))
        values = vertices_new[f, indices, 2]
        values = values[mask]

        img = mpimg.imread(image_files[f])
        if(img.dtype.type is np.uint8 or img.dtype.type is np.uint16):
            img = img.astype(np.float32) / 255.0

        superpixels = slic(img, n_segments=5000, sigma=5)
        dense_labels = util.seg_dense_interp(uv, labels_parts, superpixels).astype(np.int8)

        bg_mask = 0
        depth_map = util.dummy_image_interp(img, uv, values, bg_mask, method='nearest')

        depth_map = depth_map - np.min(vertices_new[:, :, 2]) + 500
        depth_map[dense_labels == -1] = np.max(vertices_new[:, :, 2]) - np.min(vertices_new[:, :, 2]) + 500

        ref_images[f] = img
        depth_maps[f] = depth_map

    return  depth_maps, ref_images

if expr == 'bird7':
    flip = 1
    reconstruct_file = '../../data/Youtube_birds/bird_shot07/f1t35/v5/' \
                       'vw2000_nn10_k5_threshinf_max_occ30_op0_cw2.5/init300/mdl1000_pw100_oc0_engine2_it5/object_00/stitched_rot_based.pkl'
    seg_file = '../../data/Youtube_birds/bird_shot07/f1t35/v5/' \
               'vw2000_nn10_k5_threshinf_max_occ30_op0_cw2.5/init300/mdl1000_pw100_oc0_engine2_it5/results.pkl'

with open(reconstruct_file, 'r') as f:
    reconstruct = pickle.load(f)

scene_reconstructions = {}
scene_reconstructions['rotations'] = reconstruct['rotations3d']
scene_reconstructions['translations'] = reconstruct['translations3d']
scene_reconstructions['shapes'] = reconstruct['shapes']
scene_reconstructions['colors'] = reconstruct['colors']
scene_reconstructions['points'] = reconstruct['points']

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
    vertices = np.concatenate((vertices, vertices_i))

    num_pnts = vertices_i.shape[0]

    colors = np.concatenate((colors, colors_i.T))
    points = np.concatenate((points, points_i))

    labels = np.concatenate((labels, np.ones((num_pnts, 1))*i ))

num_pnts = vertices.shape[0]
vertices = np.reshape(vertices, (num_pnts, -1, 3))
vertices = np.swapaxes(vertices, 0, 1)
vertices = vertices.reshape((-1, 3))

if(flip == 1):
    vertices[:, 2] = -vertices[:,2]

vertices[:,2] = vertices[:,2] - np.min(vertices[:,2]) + 100

if(show == 0):
    """
    show point cloud
    """
    depth_util.point_cloud_plot(vertices, colors, 0, 720, 1280, labels=labels, nframes = F)
elif(show == 1):
    """
    visualization of depth maps, press q to save rendering results
    """
    K_persp = np.array([[800, 0, 640],
                        [0, 800, 360],
                        [0, 0, 1]])
    depth_maps, ref_images = get_depth_map(seg_file, vertices, points, showFrames=1, sequence=expr)
    depth_util.depth_maps_plot(depth_maps, ref_images, K_persp)
elif(show == 2):
    """
    orthographic visualization tool showing different patches,
    the depth and flipping ambiguity of which could be fixed manually
    """
    import sys
    from video_popup.visualization import qt_app_ortho
    from PyQt4 import QtGui

    show = False
    appQt = QtGui.QApplication(sys.argv)
    win = qt_app_ortho.MainWindow(scene_reconstructions, reconstruct['images'], K, False)
    win.show()
    sys.exit(appQt.exec_())
