
import os
import numpy as np
import networkx as nx
import matplotlib.image as mpimg

import itertools
import collections

import cv2

from opensfm import opensfm_io
from opensfm import dataset

def ensure_dir(f):

    d = os.path.dirname(f)

    if not os.path.exists(d):
        os.makedirs(d)

def create_dummy_camera_model(data_path, width, height):

    with open(data_path + '/camera_models.json', 'w') as fout:
        obj = {}
        obj['dummy'] = {
            'projection_type': 'perspective',
            'width': width,
            'height': height,
            'focal': 1.0,
            'k1': 0.0,
            'k2': 0.0,
            'focal_prior': 1.0,
            'k1_prior': 0.0,
            'k2_prior': 0.0
        }
        # io.json_dump(obj, fout)
        opensfm_io.json_dump(obj, fout)

def tracks_to_opensfm_input(input_data):

    W, Z, labels, K, images = input_data

    num_frames = W.shape[0] / 2
    num_tracks = W.shape[1]

    """normalize the input tracks """
    inv_K = np.linalg.inv(K)

    W_normalized = np.dot(
        np.hstack((
            W.T.reshape((-1, 2)),
            np.ones((num_frames * num_tracks, 1))
        )), inv_K.T)

    W_normalized = W_normalized[:, 0:2].reshape((num_tracks, num_frames * 2)).T
    # np.dot( inv_K, np.vstack( (W[0:2,:], np.ones((1, num_tracks)) ) ) )

    """retrieve the colors for tracks. pick the color from the first frame it appears in"""
    track_colors = np.zeros((3, num_tracks)).astype(np.float)
    mask = np.zeros((1, num_tracks)).astype(np.bool)

    for frame in range(num_frames):
        img = mpimg.imread(images[frame])
        if(img.dtype.type is np.float32):
            img = img * 255
        mask2 = np.logical_and(np.logical_not(mask), Z[frame,:])
        track_colors[:, mask2.reshape(-1)] = img[ W[2*frame+1, mask2.reshape(-1)].astype(np.int32),
                                             W[2*frame, mask2.reshape(-1)].astype(np.int32), : ].T
        mask[:, mask2.reshape(-1)] = True

    # img = mpimg.imread(image_files[f])
    # if (img.dtype.type is np.uint8 or img.dtype.type is np.uint16):
    #     img = img.astype(np.float32) / 255.0

    """create dataset"""
    data_path, image_name = os.path.split(images[0])

    data = dataset.DataSet(data_path)
    data.config['use_dummy_camera'] = True
    data.config['align_method'] = 'no_alignment'

    if(data.config['use_dummy_camera']):
        create_dummy_camera_model(data_path, img.shape[1], img.shape[0])

    data.image_files = {}
    data.image_list = []

    for frame in range(num_frames):
        data_path, image_name = os.path.split(images[frame])
        data.image_files[image_name] = images[frame]
        data.image_list.append(image_name)

    # """
    # create common tracks
    # """
    # common_tracks_all = {}
    # for label in np.unique(labels):
    #     common_tracks = {}
    #     for pair in itertools.combinations(range(num_frames), 2):
    #         frame1 = pair[0]; frame2 = pair[1]
    #         mask = np.logical_and(Z[frame1], Z[frame2])
    #         v = list(np.where(mask)[0])
    #         p1 = W_normalized[2*frame1:2*frame1+2, v].T
    #         p2 = W_normalized[2*frame2:2*frame2+2, v].T
    #         common_tracks[(data.image_list[frame1], data.image_list[frame2])] = (v, p1, p2)
    #     common_tracks_all[label] = common_tracks

    """create graphs for each label"""
    graphs = {}
    for label in np.unique(labels):
        g = nx.Graph()
        for frame in range(num_frames):
            image = data.image_list[frame]
            g.add_node( image, bipartite=0 )
            tracks = np.where(np.logical_and(Z[frame, :], labels == label))[0]
            for track in tracks:
                g.add_node( str(track), bipartite=1 )
                g.add_edge(
                    image,
                    str(track),
                    feature = (W_normalized[2*frame, track], W_normalized[2*frame+1, track]),
                    feature_id = track+frame*num_tracks,
                    feature_color = tuple(track_colors[:,track].reshape(-1))
                )
        graphs[label] = g

    # return data, graphs, common_tracks_all
    return data, graphs

def opensfm_output_to_vispy(opensfm_reconstructions, my_data):

    scene_reconstructions = {}

    scene_reconstructions['shapes'] = {}
    scene_reconstructions['colors'] = {}
    scene_reconstructions['points'] = {}
    scene_reconstructions['rotations'] = {}
    scene_reconstructions['translations'] = {}

    W, Z, labels, K, images = my_data

    scene_reconstructions['W'] = W
    scene_reconstructions['Z'] = Z
    scene_reconstructions['labels'] = labels
    scene_reconstructions['images'] = images

    for label in opensfm_reconstructions:

        shots = collections.OrderedDict(sorted(opensfm_reconstructions[label][0].shots.items())).values()

        reconstruction = opensfm_reconstructions[label][0]
        points = reconstruction.points
        num_points = len(points)
        vertices = np.ones((num_points, 3)).astype(np.float32)
        colors = np.ones((num_points, 3)).astype(np.float32)
        vertex_ids = np.ones((num_points, 1))
        index = 0
        for id in points:
            vertices[index] = points[id].coordinates
            colors[index] = points[id].color
            vertex_ids[index] = id
            index += 1

        scene_reconstructions['shapes'][label] = vertices.T
        scene_reconstructions['colors'][label] = colors.T / 255.0
        scene_points = vertex_ids.reshape(-1)
        scene_reconstructions['points'][label] = (scene_points, np.ones_like(scene_points))

        numFrames = len(shots)
        rotations = np.zeros((3*numFrames, 3))
        translations = np.zeros((3 * numFrames, 1))
        for f in range(numFrames):

            pose = shots[f].pose
            rot = pose.rotation
            trans = pose.translation

            rotations[3*f : 3*f+3, :] = cv2.Rodrigues(rot)[0]
            translations[3*f : 3*f+3, :] = trans.reshape((3,1))

        scene_reconstructions['rotations'][label] = rotations
        scene_reconstructions['translations'][label] = translations.reshape(-1)

    return scene_reconstructions
