import os
import sys

import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D

from scipy.sparse import csgraph
from scipy.sparse import csr_matrix

from skimage.segmentation import mark_boundaries
from scipy.interpolate import griddata

import numpy as np
import cvxpy as cp

import vispy.scene
from vispy.scene import visuals

sys.path.append('../../libs/read_write_tracks/python_tracks')
sys.path.append('../../libs/nhood/python_neighbors')
sys.path.append('../../libs/permutohedral_python')

import myfilter
import tracks as pytracks
import neighbors as nbor

def ensure_dir(f):

    d = os.path.dirname(f)

    if not os.path.exists(d):
        os.makedirs(d)

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

def get_colors(N):
    colors = cmx.rainbow(np.linspace(0, 1, N))
    return colors

def permuteflat(args):
    outs = []
    olen = 1
    tlen = len(args)
    for seq in args:
        olen = olen * len(seq)
    for i in range(olen):
        outs.append([None] * tlen)
    plq = olen
    for i in range(len(args)):
        seq = args[i]
        plq = plq / len(seq)
        for j in range(olen):
            si = (j / plq) % len(seq)
            outs[j][i] = seq[si]
    for i in range(olen):
        outs[i] = tuple(outs[i])
    return outs

def rotation2dTo3d(R):
    if(R.shape[0] == 2):
        R3 = np.cross(R[0], R[1])
        R_new = np.vstack((R, R3))
    else:
        frames = R.shape[0] / 2
        R_new = np.zeros((3*frames, 3))
        for f in range(frames):
            R_new[3*f:3*f+2, :] = R[2*f:2*f+2,:]
            R3 = np.cross(R[2*f], R[2*f+1])
            R_new[3*f+2] = R3
    return R_new

def load_trajectory(tracks_path, start_frame, end_frame, min_vis_frames = 3):
    with open(tracks_path) as f:
        frames = int(next(f))
        tracks = int(next(f))
    W = np.zeros((2*frames, tracks))
    # labels = np.zeros(tracks).astype(np.int32)
    labels = np.zeros(tracks).astype(np.int32)
    W, labels = pytracks.read_tracks(tracks_path.encode(), W, labels)
    W = W[2*start_frame-2:2*end_frame,:]
    Z = (W[0::2,:] != 0) | (W[1::2,:] != 0 )
    select = sum(Z) > min_vis_frames
    W = W[:, select]
    Z = Z[:, select]
    labels = labels[select]
    return np.ascontiguousarray(W), np.ascontiguousarray(labels), np.ascontiguousarray(Z)

def get_nbor(W, Z, nhood_para):
    frames, points = Z.shape
    vw, nn, k, thresh, max_occ, occ_penalty = nhood_para.velocity_weight, nhood_para.neighbor_num, nhood_para.top_frames_num, \
                                              nhood_para.dist_threshold, nhood_para.max_occlusion_frames, nhood_para.occlusion_penalty
    bottom = np.argmax(Z, axis=0)
    top = np.argmax(Z[::-1,:], axis=0)
    top = frames - 1 - top
    M = np.zeros((nn, points))
    s = np.zeros((nn, points))
    # M, s = nbor.nhood_old(W, Z, vw, nn, k, thresh, max_occ, occ_penalty, bottom, top, M, s)
    color_matrix = np.zeros((3, points))
    color_weight = nhood_para.color_weight
    M, s = nbor.nhood(W, Z, vw, nn, k, thresh, max_occ, occ_penalty, bottom, top, M, s, color_matrix, color_weight)
    return M, s

def save_depth_map(idepth_map, file_name, vmin=0, vmax=1, cmap='gray'):

    DPI = 96

    height, width = idepth_map.shape

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize= (width / float(DPI), height / float(DPI) ))

    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(idepth_map, vmin=vmin, vmax=vmax, cmap=cmap)

    ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)

    ensure_dir(file_name)
    fig.savefig(file_name, dpi=DPI, transparent=True)

    plt.close()

def generate_para_lists(loop_dict):
    """
    :param loop_list: dictionaries of looping parameters
    :return:  list of all possible combination of parameters
    """
    expr_dict_list = []

    loop_names_list = loop_dict.keys()
    parameters_list_list = []

    for loop_name in loop_names_list:
        parameters_list_list.append(loop_dict[loop_name])

    for para_config in permuteflat(parameters_list_list):

        expr_dict = {}
        for para, loop_name in zip(para_config, loop_names_list):
            expr_dict[loop_name] = para

        expr_dict_list.append(expr_dict)

    return  expr_dict_list

def plot_proposals(W, Z, model_points, images, frame_step = 1, frame_time = 0, plot_text = 1):

    frames, points = Z.shape
    num_models = len(model_points)

    models_mask = {}
    for i in range(num_models):
        mask = np.in1d(range(points), np.array(model_points[i]))
        models_mask[i] = mask

    for frame in range(0, frames, frame_step):
        img = mpimg.imread(images[frame])
        plt.imshow(img)
        cmap = get_colors(num_models)

        for i in range(num_models):

            mask = np.logical_and( Z[frame], models_mask[i] )

            if(not any(mask)):
                continue

            plt.scatter(x=W[2*frame,  mask],
                        y=W[2*frame+1, mask],
                        color=cmap[i],
                        s=10,
                        marker='*')
            if (plot_text):
                xm = np.mean(W[2 * frame, mask])
                ym = np.mean(W[2 * frame + 1, mask])
                plt.text(xm, ym, str(i), bbox=dict(facecolor=cmap[i], alpha=0.5))

        plt.axis('off')
        # mng = plt.get_current_fig_manager()
        # mng.frame.Maximize(True)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

        plt.ion()

        if(frame_time == 0):
            plt.waitforbuttonpress()
        else:
            plt.pause(frame_time)

        plt.clf()
        #plt.draw()

    plt.close()

def plot_traj(W, Z, images, frame_step = 1, labels = 0, frame_time = 0, plot_text = 0):

    frames, points = Z.shape

    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)

    for frame in range(0,frames,frame_step):

        img = mpimg.imread(images[frame])
        plt.imshow(img)

        if(not isinstance(labels, np.ndarray)):
            plt.scatter(x=W[2*frame, Z[frame]],
                        y=W[2*frame+1, Z[frame]],
                        color='g', s=10, marker='*')

        else:
            labels = labels.reshape(-1)
            unique_labels = np.unique(labels)
            cmap = get_colors(len(unique_labels))
            for i in range(len(unique_labels)):
                mask = labels == unique_labels[i]
                mask = np.logical_and( Z[frame], mask)
                plt.scatter(x=W[2*frame, mask ],
                            y=W[2*frame+1, mask ],
                            color=cmap[i],
                            s=10,
                            marker='*')
                if(plot_text):
                    xm = np.mean(W[2 * frame, mask])
                    ym = np.mean(W[2 * frame + 1, mask])
                    plt.text(xm, ym, str(i), bbox=dict(facecolor=cmap[i], alpha=0.5))

        plt.axis('off')
        # mng = plt.get_current_fig_manager()
        # mng.frame.Maximize(True)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

        plt.ion()

        if(frame_time == 0):
            plt.waitforbuttonpress()
        else:
            plt.pause(frame_time)

        plt.clf()
        plt.draw()

    plt.close()

def plot_traj2(W, Z, images, frame_step = 1, labels = 0, frame_time = 0,
               plot_text = 0, save_fig = 0, save_folder = '', save_name=''):

    frames, points = Z.shape

    # DPI = fig.get_dpi()
    DPI = 96

    img = mpimg.imread(images[0])
    height, width, _ = img.shape

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize= (width / float(DPI), height / float(DPI) ))

    for frame in range(0,frames,frame_step):

        img = mpimg.imread(images[frame])

        # Create a figure of the right size with one axes that takes up the full figure
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.imshow(img)

        if(not isinstance(labels, np.ndarray)):
            ax.scatter(x=W[2*frame, Z[frame]],
                       y=W[2*frame+1, Z[frame]],
                       color='r', s=50, marker='.')
        else:
            labels = labels.reshape(-1)
            unique_labels = np.unique(labels)
            # cmap = get_colors(len(unique_labels))
            cmap = get_colors(np.maximum(np.max(labels) + 1, len(unique_labels)))

            for i in range(len(unique_labels)):
                mask = labels == unique_labels[i]
                mask = np.logical_and( Z[frame], mask)

                if(not any(mask)):
                    continue

                ax.scatter(x=W[2*frame, mask ],
                           y=W[2*frame+1, mask ],
                           color=cmap[i],
                           s=50,
                           marker='.')

                if(plot_text):
                    xm = np.mean(W[2*frame, mask ])
                    ym = np.mean(W[2*frame+1, mask ])
                    ax.text(xm, ym, str(i), bbox=dict(facecolor=cmap[i], alpha=0.5))

        ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)

        # mng = plt.get_current_fig_manager()
        # mng.frame.Maximize(True)

        # figManager = plt.get_current_fig_manager()
        # figManager.window.showMaximized()

        if(save_fig):
            folder, filename = os.path.split( images[frame] )
            if(save_name == ''):
                save_name = filename
            save_file = save_folder + '/' + save_name
            ensure_dir( save_file )
            fig.savefig( save_file, dpi=DPI, transparent=True)

        if(frame_time == 0):
            plt.waitforbuttonpress()
        else:
            plt.pause(frame_time)

        plt.clf()
        plt.draw()

    plt.close()

def plot_traj_test(W, W_proj, Z, images, frame_step = 1, labels = 0, frame_time = 0):

    frames, points = Z.shape

    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)

    for frame in range(0,frames,frame_step):

        img = mpimg.imread(images[frame])
        plt.imshow(img)

        if(not isinstance(labels, np.ndarray)):
            plt.scatter(x=W[2*frame, Z[frame]],
                        y=W[2*frame+1, Z[frame]],
                        color='g', s=10, marker='*')
            plt.scatter(x=W_proj[2 * frame, Z[frame]],
                        y=W_proj[2 * frame + 1, Z[frame]],
                        color='r', s=10, marker='o')
        else:
            labels = labels.reshape(-1)
            unique_labels = np.unique(labels)
            cmap = get_colors(len(unique_labels))
            for i in range(len(unique_labels)):
                mask = labels == unique_labels[i]
                mask = np.logical_and( Z[frame], mask)
                plt.scatter(x=W[2*frame, mask ],
                            y=W[2*frame+1, mask ],
                            color=cmap[i],
                            s=10,
                            marker='*')
                plt.scatter(x=W_proj[2 * frame, mask],
                            y=W_proj[2 * frame + 1, mask],
                            color=cmap[i],
                            s=10,
                            marker='o')

        plt.quiver(W[2 * frame, Z[frame]], W[2 * frame + 1, Z[frame]], \
                   W_proj[2 * frame, Z[frame]] - W[2 * frame, Z[frame]], \
                   W_proj[2 * frame + 1, Z[frame]] - W[2 * frame + 1, Z[frame]], color='r', \
                   angles='xy', scale_units='xy', scale=1, width=0.001)

        plt.axis('off')
        # mng = plt.get_current_fig_manager()
        # mng.frame.Maximize(True)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

        plt.ion()

        if(frame_time == 0):
            plt.waitforbuttonpress()
        else:
            plt.pause(frame_time)

        plt.clf()
        plt.draw()

    plt.close()

def scatter3d(X):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[0,:], X[1,:], X[2,:], color='r', marker='*')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

def plot_nbor(W, Z, s, images, labels = 0, frame_step = 1, show_edge = 0,
              show_overlap=0, show_broken=0, show_remain=0, assignment=0,
              frame_time = 0, plot_text=0):

    frames, points = Z.shape
    num_neighbors = s.shape[0]

    edge_first = np.tile(range(points), (1, num_neighbors))
    edge_second = np.reshape(s, (-1, num_neighbors*points))
    filter = edge_second != -1

    edge_first = edge_first[filter]
    edge_second = edge_second[filter]

    for frame in range(1,frames,frame_step):

        img = mpimg.imread(images[frame])
        plt.imshow(img)

        filter = Z[frame, edge_first] * Z[frame, edge_second]
        edge_first_id = edge_first[filter]
        edge_second_id = edge_second[filter]

        if(isinstance(labels, np.ndarray)):
            # cmap = get_cmap(max(labels)+1)
            cmap = get_colors(max(labels)+1)
            for k in np.unique(labels):
                # print 'labels '+ k + ": " + np.sum(np.logical_and(Z[frame], labels==k))
                # print "labels {:g} : {:g}".format(k, np.sum(np.logical_and(Z[frame], labels==k)))
                mask = np.logical_and(Z[frame], labels==k)
                plt.scatter(x=W[2*frame,   mask ],
                            y=W[2*frame+1, mask ],
                            color=cmap[k], s=2, marker='*')
                if(plot_text and np.any(mask)):
                    xm = np.mean(W[2*frame,   mask ])
                    ym = np.mean(W[2*frame+1, mask ])
                    plt.text(xm, ym, str(k), bbox=dict(facecolor=cmap[k], alpha=0.5))
        else:
            plt.scatter(x=W[2*frame, Z[frame]], y=W[2*frame+1, Z[frame]], color='r', s=10, marker='*')
        # plt.quiver(W[2*frame-1, edge_first_id], W[2*frame, edge_first_id], \
        #            W[2*frame-1, edge_second_id] - W[2*frame-1, edge_first_id], \
        #            W[2*frame, edge_second_id] - W[2*frame, edge_first_id], color='r')
        if(show_edge):
            plt.quiver(W[2*frame, edge_first_id], W[2*frame+1, edge_first_id], \
                       W[2*frame, edge_second_id] - W[2*frame, edge_first_id], \
                       W[2*frame+1, edge_second_id] - W[2*frame+1, edge_first_id], color='b', \
                       angles='xy', scale_units = 'xy', scale=1, width=0.001)

        if(show_overlap):

            if(not np.isscalar(assignment)):

                mask = np.logical_and( assignment[labels[edge_first_id], edge_second_id] == 1,
                                       labels[edge_first_id] != labels[edge_second_id] )

                first_id = edge_first_id[mask]
                second_id = edge_second_id[mask]

                plt.scatter(x=W[2*frame, first_id], y=W[2*frame+1, first_id], color='w', marker='o', s=50, facecolors='none')
                plt.scatter(x=W[2*frame, second_id], y=W[2*frame+1, second_id], color='w', marker='o', s=50, facecolors='none')

                plt.quiver(W[2*frame, first_id], W[2*frame+1, first_id], \
                           W[2*frame, second_id] - W[2*frame, first_id], \
                           W[2*frame+1, second_id] - W[2*frame+1, first_id], color='w', angles='xy', scale_units = 'xy', scale=1, width=0.001)

        if(show_broken):

            if(not np.isscalar(assignment)):

                mask = np.logical_and( assignment[labels[edge_first_id], edge_second_id] == 0,
                                       labels[edge_first_id] != labels[edge_second_id] )

                first_id = edge_first_id[mask]
                second_id = edge_second_id[mask]

                plt.quiver(W[2*frame, first_id], W[2*frame+1, first_id], \
                           W[2*frame, second_id] - W[2*frame, first_id], \
                           W[2*frame+1, second_id] - W[2*frame+1, first_id], color='r', \
                           angles='xy', scale_units = 'xy', scale=1, width=0.001)

        if(show_remain):

            if(not np.isscalar(assignment)):

                mask = np.logical_or( assignment[labels[edge_first_id], edge_second_id] == 1,
                                      labels[edge_first_id] == labels[edge_second_id] )

                first_id = edge_first_id[mask]
                second_id = edge_second_id[mask]

                plt.quiver(W[2*frame, first_id], W[2*frame+1, first_id], \
                           W[2*frame, second_id] - W[2*frame, first_id], \
                           W[2*frame+1, second_id] - W[2*frame+1, first_id], color='g', \
                           angles='xy', scale_units = 'xy', scale=1, width=0.001)

        # plt.plot([W[0, 0], W[0, 4]],[W[1, 0], W[1, 4]], color='r')
        # print W[0, 0], W[1, 0], W[0, 4] - W[0, 0], W[1, 4] - W[1, 0]
        # plt.plot(W[2*frame, edge_first_id], W[2*frame+1, edge_first_id], \
        #          W[2*frame, edge_second_id], W[2*frame+1, edge_second_id], \
        #          color='r')
        #plt.show(block=False)
        #plt.draw()

        plt.axis('off')
        # mng = plt.get_current_fig_manager()
        # mng.frame.Maximize(True)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

        plt.ion()

        if(frame_time == 0):
            plt.waitforbuttonpress()
        else:
            plt.pause(frame_time)

        plt.clf()
        plt.draw()

        # plt.show()

        # var = raw_input("Please enter something: ")
        # print "you entered", var

        #
        # if(frame != frames-1):
        #     plt.pause(0.5)
        #     plt.clf()
        #     sleep(1)

    plt.close()

def merge_parts_to_objects(W, Z, s, labels, assignment, thresh=0, del_outliers=1, outliers=0):

    """
    s: the neighborhood graph NxP matrix,
    labels: output from graphcut,
    thresh: threshold for minimum number of object points,
    del_outliers: remove outliers or not
    """
    num_neighbors, points = s.shape

    W_new = np.copy(W)
    Z_new = np.copy(Z)
    s_new = np.copy(s)
    labels_new = np.copy(labels)
    assignment_new = np.copy(assignment)


    if(del_outliers == 1 and isinstance(outliers, np.ndarray)):

        inliers = np.logical_not( outliers )
        W_new = W_new[:, inliers]
        Z_new = Z_new[:, inliers]
        s_new = s_new[:, inliers]
        labels_new = labels_new[inliers]
        assignment_new = assignment_new[:, inliers]

        # all the points, and add an extra element for -1
        mapping = np.zeros(points + 1).astype(np.int32)
        mapping[:] = -1

        # all the inlier points left
        points = np.sum(inliers)
        mapping[inliers] = range(points)

        s_new = mapping[s_new]


    edge_first_all = np.tile(range(points), (1, num_neighbors))
    edge_second_all = np.reshape(s_new, (-1, num_neighbors*points))
    filter = edge_second_all != -1
    edge_first = edge_first_all[filter]
    edge_second = edge_second_all[filter]

    # keep only unbroken edges, notice that in the case of outliers, assignment may be 0
    # but points are still connected
    mask = np.logical_or( assignment_new[labels_new[edge_first], edge_second] == 1,
                          labels_new[edge_first] == labels_new[edge_second] )
    edge_first = edge_first[mask]
    edge_second = edge_second[mask]
    data = np.ones(len(edge_first))

    edge_second_all[0][ np.setdiff1d( range(num_neighbors*points), np.where(filter.T)[0][ mask ]) ] = -1

    s_new = np.reshape(edge_second_all, (num_neighbors, points))

    # get connected components of the left neighborhood graph
    nbor_graph = csr_matrix((data, (edge_first, edge_second)), shape = (points, points))
    nums, labels_objects = csgraph.connected_components(nbor_graph, directed = False)

    # remove small labels
    hist = np.histogram(labels_objects, bins=range(nums+1))[0]
    proper_object_ind = np.where(hist > thresh)[0]
    small_object_ind = np.where(hist <= thresh)[0]

    if(len(small_object_ind) > 0):

        Zdata = np.reshape(np.hstack((Z_new, Z_new)), (-1, points))

        for i in small_object_ind:
            pmask = labels_objects == i
            fmask = np.any(Zdata[:, pmask], axis=1)
            mean_i = np.sum(W[np.ix_(fmask, pmask)] *
                            Zdata[np.ix_(fmask, pmask)], axis=1) / np.sum( Zdata[np.ix_(fmask, pmask)], axis=1 )

            dist2j = []

            for j in proper_object_ind:
                pmask2 = labels_objects == j
                fmask2 = Zdata[:, pmask2]
                Wj = W[:, pmask2]

                mask_ij = fmask2 * fmask.reshape((-1,1))
                pmask3 = np.any(mask_ij, axis=0)
                if(len(pmask3) == 0 or not any(pmask3)):
                    dist2j.append(np.inf)
                    continue

                dist_ij = np.sum((mean_i.reshape((-1,1)) - Wj[np.ix_(fmask, pmask3)]) ** 2 *
                                 mask_ij[np.ix_(fmask, pmask3)], axis=0) / np.sum(mask_ij[np.ix_(fmask, pmask3)], axis=0)
                dist2j.append(min(dist_ij))

            labels_objects[pmask] = proper_object_ind[ np.argmin(dist2j) ]

    # relabelling the left labels
    unique_labels = np.unique(labels_objects)
    mapping = np.zeros( max(unique_labels)+1, dtype=np.int )
    mapping[ unique_labels ] = range( len(unique_labels) )

    labels_objects = mapping[labels_objects ]

    return labels_objects, W_new, Z_new, assignment_new, s_new, labels_new

def break_parts(labels_parts, labels_objects, s, lambda_weight = 1):

    labels_parts_unique = np.unique( labels_parts )
    max_label = max(labels_parts_unique)

    for i in labels_parts_unique:
        mask = labels_parts == i
        obj_unique = np.unique(labels_objects[ mask ])
        if(len(obj_unique) == 1):
            continue
        else:
            obj_unique_left =  np.setdiff1d(obj_unique, min(obj_unique))
            for j in obj_unique_left:
                mask_obj = labels_objects[ mask ] == j
                labels_parts[ np.where(mask)[0][mask_obj] ] = max_label + 1
                max_label = max_label + 1

    # generate a new assignment_new based on labels_parts
    points = len(labels_parts)
    models = len( np.unique(labels_parts) )

    assignment_new = np.zeros((models, points))
    num_neighbors = s.shape[0]

    edge_first_all = np.tile(range(points), (1, num_neighbors))
    edge_second_all = np.reshape(s, (-1, num_neighbors*points))
    filter = edge_second_all != -1
    edge_first = edge_first_all[filter]
    edge_second = edge_second_all[filter]

    assignment_new[ labels_parts[edge_first_all],  edge_first_all ] = 1
    assignment_new[ labels_parts[edge_first], edge_second ] = lambda_weight

    return labels_parts, assignment_new

def plot_superpixels(img, superpixels = 0, sp_edges = 0, sp_centers = 0, color = np.array([0, 1, 0])):

    plt.imshow( img )

    if(isinstance(superpixels, np.ndarray)):
        plt.imshow(mark_boundaries(img, superpixels, mode='inner'))
        # mode: string in {'thick', 'inner', 'outer', 'subpixel'}

    if(not np.isscalar( sp_edges )):
        # plot the edges
        plt.quiver(sp_centers[0][sp_edges[0]], sp_centers[1][sp_edges[0]],
                   sp_centers[0][sp_edges[1]] - sp_centers[0][sp_edges[0]],
                   sp_centers[1][sp_edges[1]] - sp_centers[1][sp_edges[0]],
                   color=color, angles='xy', scale_units = 'xy', scale=1, width=0.001)

def plot_dense_segmentation(seg, superpixels, sp_edges = 0, sp_centers = 0):

    H, W = seg.shape
    seg_img = np.zeros([H, W, 3])
    cmap = get_colors(np.max(seg)+1)

    for i in np.unique(seg):
        if(i != -1):
            mask = seg == i
            seg_img[mask,:] = cmap[i,0:3]

    plt.imshow( mark_boundaries(seg_img, superpixels) )

    if(not np.isscalar( sp_edges )):
        # plot the edges
        plt.quiver(sp_centers[0][sp_edges[0]],  sp_centers[1][sp_edges[0]],
                   sp_centers[0][sp_edges[1]] - sp_centers[0][sp_edges[0]],
                   sp_centers[1][sp_edges[1]] - sp_centers[1][sp_edges[0]],
                   color='g', angles='xy', scale_units = 'xy', scale=1, width=0.001)

        # plt.waitforbuttonpress()
        #
        # plt.clf()
        # plt.draw()

def save_dense_segmentation(dense_labels, save_file):

    # DPI = fig.get_dpi()
    DPI = 96

    height, width = dense_labels.shape

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize= (width / float(DPI), height / float(DPI) ))


    seg_img = np.zeros([height, width, 3])
    cmap = get_colors(np.max(dense_labels)+1)

    for i in np.unique(dense_labels):
        if(i != -1):
            mask = dense_labels== i
            seg_img[mask,:] = cmap[i,0:3]

    # Create a figure of the right size with one axes that takes up the full figure
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(seg_img)

    ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)

    ensure_dir( save_file )
    fig.savefig( save_file, dpi=DPI, transparent=True)

    plt.close()

### finding the scales
def get_init_scales(labels, bg_label, inv_depths, superpixels_W, object_edges,
                    dense_labels = 0, superpixels = 0, sp_centers = 0):

    # scales = np.ones_like( np.unique(labels) ).astype(np.float64) * -1
    scales = np.ones( len(np.unique(labels)) ) * -1

    for fg_label in np.unique(labels):

        if(fg_label != bg_label):

            # W_bg = W[0:2, labels == bg_label]
            # W_fg = W[0:2, labels == fg_label]

            if(object_edges[0].size > 0):

                mask = np.logical_and(
                    object_edges[1,:] == bg_label,
                    object_edges[3,:] == fg_label
                )

                bg_superpixels1 = object_edges[0, mask]
                fg_superpixels1 = object_edges[2, mask]

                mask2 = np.logical_and(
                    object_edges[3,:] == bg_label,
                    object_edges[1,:] == fg_label
                )

                bg_superpixels2 = object_edges[2, mask2]
                fg_superpixels2 = object_edges[0, mask2]

                bg_superpixels = np.hstack( (bg_superpixels1, bg_superpixels2 ) )
                fg_superpixels = np.hstack( (fg_superpixels1, fg_superpixels2 ) )

                # if(fg_superpixels.size != 0):
                #     scales[fg_label] = np.inf

                # """plot the connections from foreground to background"""
                # edges = np.vstack((fg_superpixels, bg_superpixels))
                # plt.clf(); plot_dense_segmentation(dense_labels, superpixels, edges, sp_centers)

                for bg_seg, fg_seg in zip(bg_superpixels, fg_superpixels):

                    # plt.clf()
                    # plot_dense_segmentation(dense_labels, superpixels,np.array([[fg_seg],[bg_seg]]), sp_centers)

                    bg_invd = inv_depths[superpixels_W == bg_seg]
                    fg_invd = inv_depths[superpixels_W == fg_seg]

                    bg_label_mask = labels[superpixels_W == bg_seg] == bg_label
                    fg_label_mask = labels[superpixels_W == fg_seg] == fg_label

                    bg_invd = bg_invd[bg_label_mask]
                    fg_invd = fg_invd[fg_label_mask]

                    """print inverse depth values"""
                    print "____________________________"
                    print bg_seg, fg_seg
                    print "bg inverse depth"
                    print bg_invd
                    print "fg inverse depth"
                    print fg_invd
                    print "____________________________"

                    # filter negative values
                    fg_invd = fg_invd[ fg_invd > 0 ]
                    bg_invd = bg_invd[ bg_invd > 0 ]

                    if( len(fg_invd) > 0 and len(bg_invd) > 0 ):
                        if(scales[fg_label] == -1):
                            scales[fg_label] = np.min(fg_invd) / np.max(bg_invd)
                        else:
                            scales[fg_label] = np.minimum(scales[fg_label], np.min(fg_invd) / np.max(bg_invd))

            if(scales[fg_label] != -1):
                inv_depths[labels == fg_label] /= scales[fg_label]

    depths = 1.0 / inv_depths

    return scales, depths, inv_depths

### convert sparse segmentation results into dense, based on superpixel segmentation
def seg_dense_interp(Ws, labels, superpixel_seg):

    H, W = superpixel_seg.shape
    labels_num = len(np.unique(labels))
    voting = np.zeros([H, W, labels_num])
    # if no points are assigned, set to -1
    result = np.ones([H,W]) * -1

    for p in xrange(Ws.shape[1]):
        voting[ min(max(round(Ws[1, p]),0), H -1),
                min(max(round(Ws[0, p]),0), W -1),
                labels[p] ] += 1

    # sweep over each superpixel, and check the voting
    for sup in np.unique(superpixel_seg):
        mask = superpixel_seg == sup
        count = np.sum(voting[ mask, : ], axis=0)
        if(max(count) > 0):
            result[mask] = np.argmax(count, axis=0)

    return result.astype(int)

### recover image from sparse point values using TV prior
def image_inpainting_sparse(vu, depths, nH, nW):

    depth_map_sparse = np.zeros((nH, nW))

    vu_int = (vu + 0.5).astype(np.int32)
    vu_int[0] = np.maximum( np.minimum(vu_int[0], nH - 1), 0 )
    vu_int[1] = np.maximum( np.minimum(vu_int[1], nW - 1), 0 )

    depth_map_sparse[vu_int[0], vu_int[1]] = depths

    Known = np.zeros((nH, nW))
    Known[vu_int[0], vu_int[1]] = 1

    U = cp.Variable(nH, nW)
    obj = cp.Minimize(cp.tv(U))
    constraints = [cp.mul_elemwise(Known, U) == cp.mul_elemwise(Known, depth_map_sparse)]
    prob = cp.Problem(obj, constraints)
    # Use SCS to solve the problem.
    prob.solve(verbose=True, solver=cp.SCS)

    depth_map = U.value

    return depth_map

### denoising image using TV prior
def image_inpainting_dense(depth_map, lambda_coef = 1):

    nH, nW = depth_map.shape
    U = cp.Variable(nH, nW)
    obj = cp.Minimize(cp.tv(U) + lambda_coef * cp.sum_squares( depth_map - U ))
    constraints = []
    prob = cp.Problem(obj, constraints)

    prob.solve(verbose=True, solver=cp.SCS)

    depth_map_tv = U.value

    return  depth_map_tv

### bilateral image interpolation based on fast permutohedral filtering
def bilateral_filter_image_interp(image, uv, values, iter = 5, cf = 0.1, sf = 5,
                                  bg_mask = '', cf_last = 0.1, sf_last = 0.5,
                                  flip_fg = 0):
    """
    :param image: input rgb image
    :param uv: sparse points where we have values
    :param values: values on sparse points
    :param iter: number of iterations
    :param cf: color sigma
    :param sf: spatial sigma
    :return: interpolated image
    """
    nH, nW, nC = image.shape

    uv_int = (uv + 0.5).astype(np.int32)
    uv_int[0] = np.maximum( np.minimum(uv_int[0], nW - 1), 0 )
    uv_int[1] = np.maximum( np.minimum(uv_int[1], nH - 1), 0 )

    perturb_image = np.zeros((nH, nW, 2))
    perturb_image[uv_int[1], uv_int[0], 0] = values
    perturb_image[uv_int[1], uv_int[0], 1] = 1
    # perturb_image[ perturb_image[:,:,0] == 0, 1 ] = 0

    perturb_image2 = perturb_image

    for i in xrange(iter):

        out = myfilter.filter(image, perturb_image2, sf, cf)

        # mask of points have been influenced
        mask = out[:,:,1] > 0
        out[ mask, 0 ] /= out[mask, 1]
        out[ mask, 1 ] = 1

        # reset the values of input points
        out[ perturb_image != 0 ] = perturb_image[ perturb_image != 0 ]
        perturb_image2 = out

    if(isinstance(bg_mask, np.ndarray)):
        fg_mask = np.logical_not(bg_mask)
        if(flip_fg == 1):
            out[fg_mask, 0] = -out[fg_mask, 0]
        out[bg_mask, 0] = np.max(out[fg_mask, 0])

    # out = myfilter.filter(image, perturb_image2, sf_last, cf_last)
    #
    # mask = out[:,:,1] > 0
    # out[ mask, 0 ] /= out[mask, 1]
    # out[ mask, 1] = 1
    #
    # # reset the values of input points
    # out[ perturb_image != 0 ] = perturb_image[ perturb_image != 0 ]

    return out[:,:,0]

### dummy image interpolation
def dummy_image_interp(image, uv, values, bg_mask = '', flip_fg = 0, method = 'nearest'):

    nH, nW, nC = image.shape
    grid_x, grid_y = np.mgrid[0:nH, 0:nW]
    depth_map = griddata(uv[1::-1,:].T, values, (grid_x, grid_y), method = method)

    ## computer the valid mask and do a nearest neighbor interpolation
    # mask = np.logical_not( np.isnan(depth_map) )
    # vut = np.vstack( (np.where(mask)[0], np.where(mask)[1]) ).T
    # depth_map = griddata(vut, depth_map[mask], (grid_x, grid_y), method='nearest')

    if(isinstance(bg_mask, np.ndarray)):
        fg_mask = np.logical_not(bg_mask)
        if(flip_fg == 1):
            depth_map[fg_mask] = -depth_map[fg_mask]
        depth_map[bg_mask] = np.max(depth_map[fg_mask])

    return  depth_map

### plot 3d point clouds elegantly
def plot_3d_point_cloud_vispy(points, colors, size=5):

    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor='w')
    view = canvas.central_widget.add_view()

    scatter = visuals.Markers()
    scatter.set_data(points / (np.max(points) - np.min(points)),
                     edge_color = colors, size = 1)

    # view.camera = 'turntable'
    view.camera = 'arcball'
    view.add(scatter)
    visuals.XYZAxis(parent=view.scene)

    vispy.app.run()

"""retrieve the colors for tracks. pick the color from the first frame it appears in"""
def retrieve_colors(W, Z, image_files, color_range=0):

    num_tracks = W.shape[1]
    num_frames = W.shape[0] / 2

    track_colors = np.zeros((3, num_tracks)).astype(np.float)
    mask = np.zeros((1, num_tracks)).astype(np.bool)

    for frame in range(num_frames):
        img = mpimg.imread(image_files[frame])
        if(color_range == 0):
            if (img.dtype.type is np.uint8 or img.dtype.type is np.uint16):
                img = img.astype(np.float32) / 255.0
        if(color_range == 1):
            if (img.dtype.type is np.float32):
                img = img * 255
        mask2 = np.logical_and(np.logical_not(mask), Z[frame, :])
        track_colors[:, mask2.reshape(-1)] = img[W[2 * frame + 1, mask2.reshape(-1)].astype(np.int32),
                                             W[2 * frame, mask2.reshape(-1)].astype(np.int32), :].T
        mask[:, mask2.reshape(-1)] = True

    return track_colors