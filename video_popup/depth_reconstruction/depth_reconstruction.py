# external libraries
import cPickle as pickle
import os.path

import cv2
import cvxpy as cp

import matplotlib
matplotlib.use('Agg')

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from scipy.interpolate import griddata
from skimage.segmentation import find_boundaries
from skimage.segmentation import mark_boundaries
from skimage.segmentation import slic, quickshift

import logging

# my own stuff

import depth_util

from video_popup.utils import util
from video_popup.utils import sintel_io as sio
from video_popup.reconstruction import sfm2

class DepthReconstruction(object):

    def __init__(self, data, para):
        """
        Initialize data
        """
        W, Z, labels, K, images = data

        self.W = W
        self.Z = Z
        self.K = K
        self.images = images
        self.labels = labels

        self.depths = np.zeros_like(labels).astype(np.float64)
        self.inv_depths = np.zeros_like(labels).astype(np.float64)
        self.pnts_num = np.zeros_like(np.unique(labels))
        self.fitting_cost = np.zeros_like(labels).astype(np.float64)

        self.ref_image = mpimg.imread(images[0])

        ## get colors for the tracks
        self.track_colors = self.ref_image[W[1].astype(np.int32), W[0].astype(np.int32), :]

        self.para = para

    def run_check(self, check_num = 0):

        """  check if experiment has been done  """
        """
         create loop names list and corresponding looping parameters
         """
        loop_dict = {
            'kappa': self.para.get('kappa_list', [1]),
            'gamma': self.para.get('gamma_list', [1]),
            'fg_coef': self.para.get('fg_coef_list', [1]),
            'lambda_reg': self.para.get('lambda_reg_list', [0.5]),
            'lambda_reg2': self.para.get('lambda_reg2_list', [0]),
            'lambda_depth': self.para.get('lambda_depth_list', [0]),
            'lambda_constr': self.para.get('lambda_constr_list', [0])
        }

        expr_dict_list = util.generate_para_lists(loop_dict)
        energy_para = expr_dict_list[check_num]

        kappa = energy_para.get('kappa', 1)
        gamma = energy_para.get('gamma', 1)
        fg_coef = energy_para.get('fg_coef', 1)

        lambda_reg = energy_para.get('lambda_reg', 1)
        lambda_reg2 = energy_para.get('lambda_reg2', 0)
        lambda_depth = energy_para.get('lambda_depth', 0)
        lambda_constr = energy_para.get('lambda_constr', 0)

        num_segments = self.para['num_segments']

        results_folder = 'segs{:d}_lreg{:g}_kpa{:g}_gamma{:g}_lregtheta{:g}_ldepth{:g}_lconstr{:g}_fg_coef{:g}'.format(
            num_segments, lambda_reg, kappa, gamma, lambda_reg2, lambda_depth, lambda_constr, fg_coef)

        return results_folder

    def run(self):

        # results_folder = self.para['seg_folder'] + '/VladlenDense/' + self.run_check( check_num=0 )
        # if(os.path.isfile(results_folder + '/results.pkl')):
        #     return

        # plot segmentation results
        #util.plot_traj2(self.W, self.Z, self.images, labels=self.labels, save_fig=1)
        # util.plot_traj2(self.W, self.Z, self.images, labels=self.labels)

        self.sparse_reconstruction()

        try:
            results_file = self.para['seg_folder'] + '/SuperPixels/' + str(self.para['num_segments']) + '/sp_info.pkl'
            util.ensure_dir(results_file)
            with open(results_file, 'r') as f:
                superpixels, sp_centers, sp_colors, superpixel_edges, object_edges, dense_labels = pickle.load(f)
        except:
            superpixels, sp_centers, sp_colors = self.superpixel_seg(num_segments=self.para['num_segments'])
            superpixel_edges, object_edges, dense_labels = self.create_edges(superpixels, sp_centers, sp_colors)
            data = (superpixels, sp_centers, sp_colors, superpixel_edges, object_edges, dense_labels)
            with open(results_file, 'w') as f:
                    pickle.dump(data, f)

        # plt.imshow(dense_labels)
        nH, nW, nC = self.ref_image.shape
        sp_mcolor_image = sp_colors[:, superpixels.reshape(-1,1)]
        sp_mcolor_image = sp_mcolor_image.T.reshape(nH, nW, nC)
        # sp_mgray_image = np.dot(sp_mcolor_image[...,:3], [0.299, 0.587, 0.114])
        # plt.imshow(sp_mgray_image, cmap = 'gray')

        self.sp_mcolor_image = sp_mcolor_image
        self.superpixels = superpixels; self.sp_centers = sp_centers; self.sp_colors = sp_colors
        self.superpixel_edges = superpixel_edges; self.object_edges = object_edges; self.dense_labels = dense_labels

        # util.plot_superpixels(sp_mcolor_image, superpixels)
        # util.plot_superpixels(sp_mcolor_image, superpixels=superpixels, sp_edges=superpixel_edges[[2, 5]],
        #                       sp_centers=sp_centers, color=np.random.rand(sp_centers.shape[1], 3))

        # compute the relative scales between different objects, with the scale of background fixed to 1
        scales = self.get_init_scales( superpixels, object_edges)
        objects_num = len(scales)

        # since objects have been registered to the background, we could now make the sparse reconstructions dense
        nH, nW, nC = self.ref_image.shape
        grid_x, grid_y = np.mgrid[0:nH, 0:nW]

        # #plot sparse 3D reconstruction results elegantly
        # vertices = np.dot(np.linalg.inv(self.K),
        #                   self.depths * ( np.vstack((self.W[0:2,:],
        #                                              np.ones( (1,self.W.shape[1]) ) ) ) ) ).T
        #
        # # util.plot_3d_point_cloud_vispy( vertices, self.track_colors )
        #
        # vertices[:,1] = -vertices[:,1]
        # vertices[:,2] = -vertices[:,2]
        #
        # from video_popup.visualization import vispy_viewer
        # vispy_viewer.app_call(vertices, self.track_colors, self.K, nH, nW)

        # we already create edges between fg and bg, just set those unknown region as bg
        dense_labels[dense_labels == -1] = self.bg_label

        # plt.imshow(dense_labels)
        # util.plot_dense_segmentation(dense_labels, superpixels)

        """ check depth value range """
        # plt.plot(np.sort(self.depths))

        """ check the depth for each superpixel and the scales """
        # depth_map_sp = self.get_depth_map_sp(self.W[0:2,:], self.depths, self.labels, self.superpixels, self.dense_labels)
        # util.plot_superpixels(depth_map_sp.astype(np.float64), superpixels = superpixels,
        #                       sp_edges = object_edges[[0, 2]], sp_centers = sp_centers)

        depth_map_interp = griddata(self.W[1::-1,:].T, self.depths, (grid_x, grid_y), method='nearest')

        # try image inpainting
        # depth_map_inpainting = util.image_inpainting_sparse(self.W[1::-1,:], self.depths, nH, nW)
        # depth_util.depth_map_plot(depth_map_interp, self.ref_image, self.K)
        # depth_util.depth_map_plot(depth_map_interp, self.ref_image, self.K, labels = dense_labels)
        # depth_util.depth_map_plot(depth_map, sp_colors_t[superpixels.reshape(-1),:].reshape((nH,nW,3)), self.K, labels = superpixels)

        if (self.para['has_gt']):

            depth_gt = self.get_ground_truth_depth(self.para)

            doing_rigid = self.para.get('doing_rigid', 0)
            if(doing_rigid):
                global_scale = self.evaluation(depth_map_interp, depth_gt, self.para, method='SparseInterpRigid')
            else:
                global_scale = self.evaluation(depth_map_interp, depth_gt, self.para, method='SparseInterp')

            """ save sparse result using linear interpolation """
            depth_map_interp_linear = griddata(self.W[1::-1,:].T, self.depths, (grid_x, grid_y), method='linear')
            depth_map_interp_linear *= global_scale
            idepth = 1.0 / depth_map_interp_linear
            idepth[np.isnan(idepth)] = 1
            file_name = self.para['seg_folder'] + '/SparseInterp/' + 'idepth_map_linear_interp.png'
            util.save_depth_map(idepth, file_name, vmin= depth_gt['invd_min'], vmax= depth_gt['invd_max'], cmap='gray')
            idepth[np.isnan(idepth)] = 0
            file_name = self.para['seg_folder'] + '/SparseInterp/' + 'idepth_map_linear_interp_r.png'
            util.save_depth_map(idepth, file_name, vmin= depth_gt['invd_min'], vmax= depth_gt['invd_max'], cmap='gray_r')

        if(doing_rigid):
            return

        """
        create loop names list and corresponding looping parameters
        """
        loop_dict = {
            'kappa':        self.para.get('kappa_list', [1]),
            'gamma':        self.para.get('gamma_list', [1]),
            'fg_coef':      self.para.get('fg_coef_list', [1]),
            'lambda_reg':   self.para.get('lambda_reg_list',[0.5]),
            'lambda_reg2':  self.para.get('lambda_reg2_list', [0]),
            'lambda_depth': self.para.get('lambda_depth_list', [0]),
            'lambda_constr':self.para.get('lambda_constr_list', [0])
        }

        expr_dict_list = util.generate_para_lists(loop_dict)

        for energy_para in expr_dict_list:

            depth_map, fg_to_bg_edges, inv_scales = \
                self.global_optimization( superpixels, objects_num, superpixel_edges, object_edges,
                                          dense_labels, sp_colors, self.para, energy_para)

            if(self.para['has_gt']):
                depth_gt = self.get_ground_truth_depth(self.para)
                # # scale up depths
                # for obj in range(objects_num):
                #     depth_map_interp[dense_labels == obj] *= 1/inv_scales[obj,0]
                # self.evaluation(depth_map_interp, depth_gt, self.para, method='SparseInterp', scale_optim = 1)
                self.evaluation( depth_map, depth_gt, self.para, epara = energy_para)

    def sparse_reconstruction(self, plot_seg = 0, plot_recons = 0):
        """
        Do sparse reconstruction on each segmented rigid objects independently
        To make things robust, we just keep depth values within [5, 95] range
        """

        try:

            results_file = self.para['seg_folder'] + '/SparseResults/' + 'results.pkl'
            util.ensure_dir(results_file)

            with open(results_file, 'r') as f:
                data = pickle.load(f)

            self.W = data['W']; self.Z = data['Z']
            self.depths = data['depths']
            self.inv_depths = data['inv_depths']
            self.labels = data['labels']
            self.track_colors = data['track_colors']
            self.pnts_num = data['pnts_num']
            self.fitting_cost = data['fitting_cost']

        except:

            for label in np.unique(self.labels):

                mask = self.labels == label
                Wi = self.W[:,mask]
                Zi = self.Z[:,mask]

                if(plot_seg):
                    util.plot_traj(Wi, Zi, self.images, 1)

                R, T, X, cost, inliers = sfm2.reconstruction(Wi, self.K)

                xmin = np.percentile(X[2,:], 5)
                xmax = np.percentile(X[2,:], 95)
                mask2 = np.logical_or( X[2,:] > xmax, X[2,:] < xmin )
                mask2 = np.logical_or( mask2, X[2,:] < 0)
                mask2[np.logical_not(inliers).reshape(-1)] = True

                rm = np.where(mask)[0][mask2]

                self.W = np.delete(self.W, rm, 1)
                self.Z = np.delete(self.Z, rm, 1)
                self.labels = np.delete(self.labels, rm, 0 )

                # if(plot_recons):
                #     util.scatter3d(X[:, mask2 == False])

                if(plot_recons):
                    util.plot_3d_point_cloud_vispy(X[0:3, mask2 == False].T,
                                                   self.track_colors[mask, :][mask2 == False])
                    # vertices = X[0:3, mask2 == False].T
                    # colors = self.track_colors[mask, :][mask2 == False]
                    # vertices[:,1] = -vertices[:,1]
                    # vertices[:,2] = -vertices[:,2]
                    #
                    # app_call(vertices, colors, self.K, 370, 1226)

                self.track_colors = np.delete(self.track_colors, rm, 0)

                self.depths[mask] = np.dot(self.K, X[0:3, :])[2,:]
                self.inv_depths[mask] = 1.0 / self.depths[mask]
                self.fitting_cost[mask] = cost

                self.depths = np.delete( self.depths, rm, 0 )
                self.inv_depths = np.delete( self.inv_depths, rm, 0 )
                self.fitting_cost = np.delete( self.fitting_cost, rm, 0 )

                self.pnts_num[label] = np.sum(mask)

            data = {'W': self.W, 'Z': self.Z, 'labels': self.labels,
                    'track_colors': self.track_colors, 'pnts_num':self.pnts_num,
                    'depths': self.depths, 'inv_depths': self.inv_depths,
                    'fitting_cost': self.fitting_cost}

            with open('{:s}'.format(results_file), "wb") as f:
                pickle.dump(data, f, True)

    def superpixel_seg(self, num_segments = 5000, method='slic', show_res = 0):
        """
        superpixel segmentation, we use slic method by default
        """
        if(method == 'slic'):
            superpixels = slic(self.ref_image, n_segments = num_segments, sigma = 5)
        elif(method == 'quickshift'):
            superpixels = quickshift(self.ref_image, ratio = 1.0, sigma = 5)

        nH, nW, nC = self.ref_image.shape

        # compute the center of all superpixels for ploting
        u = np.array(range(nW))
        uu = np.tile(u,(nH, 1) ).reshape(nH * nW, 1)
        v = np.array(range(nH)).reshape(nH,1)
        vv = np.tile(v,(1, nW) ).reshape(nH * nW, 1)
        ww = np.ones_like(uu)

        sp_centers = np.zeros((2, len(np.unique(superpixels))))
        sp_colors = np.zeros((3, len(np.unique(superpixels))))

        for i in np.unique(superpixels):

            mask = superpixels == i
            mask.reshape(nH * nW, 1)
            sp_centers[0,i] = np.mean( uu[mask.reshape((-1,1)) ] )
            sp_centers[1,i] = np.mean( vv[mask.reshape((-1,1)) ] )

            sp_colors[0,i] = np.mean(self.ref_image[:,:,0][mask])
            sp_colors[1,i] = np.mean(self.ref_image[:,:,1][mask])
            sp_colors[2,i] = np.mean(self.ref_image[:,:,2][mask])

        if(show_res):
            # show the output of SLIC
            fig = plt.figure("Superpixels -- %d superpixels" % (num_segments))
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(mark_boundaries(self.ref_image, superpixels))
            plt.axis("off")
            plt.show()

        return  superpixels, sp_centers, sp_colors
        # self.superpixels = superpixels
        # self.sp_centers = sp_centers

    def create_edges(self, superpixels, sp_centers, sp_colors, plot_sp_edges = 0,
                     plot_object_edges = 0, plot_sp_mcolor= 0):
        """
        :param superpixels: superpixel segmentation results
        :param sp_centers:  superpixel centers
        :param plot_sp_edges: whether plotting edges connecting superpixels or not
        :param plot_object_edges: whether ploting edges connecting superpixels belonging to different objects or not
        :return: superpixel edges and object edges
        """

        nH, nW, nC = self.ref_image.shape

        dense_labels = util.seg_dense_interp(self.W, self.labels, superpixels)

        sp_mcolor_image = sp_colors[:, superpixels.reshape(-1,1)]
        sp_mcolor_image = sp_mcolor_image.T.reshape(nH, nW, nC)

        boundary = find_boundaries(superpixels, mode='inner')

        pos1_xy = np.where(boundary)

        uv_shift = np.array([[1, 0],
                             [0, 1],
                             [-1,0],
                             [0,-1]])

        superpixel_edges = {}
        object_edges = {}

        # check pixels in superpixel boundaries, if itself and its neighbors belong to different superpixels
        # or if itself and its neighbors belong to different superpixels and different objects at the same time
        for i in range(4):

            if(i == 0):
                mask = np.logical_and(pos1_xy[0] + 1 <= nH - 1,
                    superpixels[ pos1_xy[0], pos1_xy[1] ] != superpixels[ np.minimum(pos1_xy[0] + 1, nH-1), pos1_xy[1] ])
            if(i == 1):
                mask = np.logical_and(pos1_xy[1] + 1 <= nW - 1,
                    superpixels[ pos1_xy[0], pos1_xy[1] ] != superpixels[ pos1_xy[0], np.minimum(pos1_xy[1] + 1, nW-1) ])
            if(i == 2):
                mask = np.logical_and(pos1_xy[0] - 1 >= 0,
                    superpixels[ pos1_xy[0], pos1_xy[1] ] != superpixels[ np.maximum(pos1_xy[0]-1, 0), pos1_xy[1] ])
            if(i == 3):
                mask = np.logical_and(pos1_xy[1] - 1 >= 0,
                    superpixels[ pos1_xy[0], pos1_xy[1] ] != superpixels[ pos1_xy[0], np.maximum(pos1_xy[1]-1, 0) ])

            superpixel_edges[i] = np.vstack((
                pos1_xy[0][mask],
                pos1_xy[1][mask],
                superpixels[ pos1_xy[0][mask], pos1_xy[1][mask] ],
                pos1_xy[0][mask] + uv_shift[i][0],
                pos1_xy[1][mask] + uv_shift[i][1],
                superpixels[ pos1_xy[0][mask] + uv_shift[i][0], pos1_xy[1][mask] + uv_shift[i][1] ]
            ))

            mask2 = np.logical_and.reduce((
                        dense_labels[ pos1_xy[0][mask], pos1_xy[1][mask] ] != -1,
                        dense_labels[ pos1_xy[0][mask] + uv_shift[i][0], pos1_xy[1][mask] + uv_shift[i][1] ] != -1,
                        dense_labels[ pos1_xy[0][mask], pos1_xy[1][mask] ] !=
                        dense_labels[ pos1_xy[0][mask] + uv_shift[i][0], pos1_xy[1][mask] + uv_shift[i][1] ]
                    ))

            object_edges[i] = np.vstack((
                superpixels[ pos1_xy[0][mask][mask2], pos1_xy[1][mask][mask2] ],
                dense_labels[ pos1_xy[0][mask][mask2], pos1_xy[1][mask][mask2] ],
                superpixels[ pos1_xy[0][mask][mask2] + uv_shift[i][0], pos1_xy[1][mask][mask2] + uv_shift[i][1] ],
                dense_labels[ pos1_xy[0][mask][mask2] + uv_shift[i][0], pos1_xy[1][mask][mask2] + uv_shift[i][1] ]
            ))

            # if there are any object edges, in kitti case, most scenes are static
            if(object_edges[0].size > 0):
                object_edges[i] = np.vstack({tuple(row) for row in object_edges[i].T}).T

        # put all four cases together
        superpixel_edges = np.hstack((
            superpixel_edges[0],
            superpixel_edges[1],
            superpixel_edges[2],
            superpixel_edges[3],
        ))

        # only keep one way connections
        se_mask = superpixel_edges[2] > superpixel_edges[5]
        superpixel_edges = superpixel_edges[:, se_mask]

        # remove redundant columns
        sps = superpixel_edges[[2,5],:].T
        pos = np.ascontiguousarray(sps).view(np.dtype((np.void, sps.dtype.itemsize * sps.shape[1])))
        _, idx = np.unique(pos, return_index=True)

        superpixel_edges = superpixel_edges[:, idx]

        if(plot_sp_edges):
            util.plot_dense_segmentation( dense_labels, superpixels, superpixel_edges[[2,5]], sp_centers )

        if(object_edges[0].size > 0):

            # put all four cases together
            object_edges = np.hstack((
                object_edges[0],
                object_edges[1],
                object_edges[2],
                object_edges[3]
            ))

            # only keep one way connections
            oe_mask = object_edges[0] > object_edges[2]
            object_edges = object_edges[:, oe_mask]

            object_edges = np.vstack({tuple(row) for row in object_edges.T}).T

            if(plot_object_edges):
                util.plot_dense_segmentation( dense_labels, superpixels, object_edges[[0,2]], sp_centers )

        if(plot_sp_mcolor):
            # plt.imshow(sp_mcolor_image)
            util.plot_dense_segmentation( sp_mcolor_image[:,:,0] * 0.299 +
                                          sp_mcolor_image[:,:,1] * 0.587 +
                                          sp_mcolor_image[:,:,2] * 0.114,
                                          superpixels, object_edges[[0,2]], sp_centers )

        return  superpixel_edges, object_edges, dense_labels
        # self.superpixel_edges = superpixel_edges
        # self.object_edges = object_edges

    def get_depth_map_sp(self, W, depths, labels, superpixels, dense_labels, method = 'average'):

        """
        compute the depth map for the superpixels based on sparse depth values of point tracks,
        we check this to see which superpixel depth computation method gives most robust scale estimates
        between neighboring superpixels around object and environment boundary.
        """

        depth_map_sp = np.zeros_like(dense_labels)
        tracks_label_map = np.ones_like(dense_labels) * -1

        uu = W[0].reshape(-1).astype(np.int32)
        vv = W[1].reshape(-1).astype(np.int32)

        depth_map_sp[vv, uu] = depths
        tracks_label_map[vv, uu] = labels

        sps = np.unique(superpixels)

        for sp in sps:

            sp_mask = superpixels == sp

            sp_tracks_mask = tracks_label_map[sp_mask] != -1

            if(any(sp_tracks_mask)):

                sp_track_label = np.argmax(np.bincount( tracks_label_map[sp_mask][sp_tracks_mask] ))
                sp_label_mask = tracks_label_map[sp_mask][sp_tracks_mask] == sp_track_label

                sp_depths = depth_map_sp[sp_mask][sp_tracks_mask][sp_label_mask]

                vmask = sp_depths != 0
                num_pnts_sp = np.sum(vmask)

                if(num_pnts_sp > 0):
                    depth_map_sp[sp_mask] = np.sum(sp_depths) / num_pnts_sp

        return depth_map_sp

    def get_init_scales(self, superpixels, object_edges):

        bg_label = np.argmax(self.pnts_num)
        self.bg_label = bg_label

        superpixels_W = superpixels[ self.W[1,:].astype(int), self.W[0,:].astype(int) ]

        scales, self.depths, self.inv_depths = \
            util.get_init_scales(self.labels, self.bg_label, self.inv_depths, superpixels_W, object_edges,
                                 superpixels=self.superpixels, dense_labels=self.dense_labels,
                                 sp_centers=self.sp_centers)
            # util.get_init_scales(self.labels, self.bg_label, self.inv_depths, superpixels_W, object_edges)
        return  scales

    def global_optimization(self, superpixels, objects_num, superpixel_edges,
                            object_edges, dense_labels, sp_colors, para, energy_para):

        """
        for better tuning parameters, probably we should normalize the parameters based on
        the number of superpixels, the number of point tracks and the depth range value.
        """

        # lambda_reg, kappa, gamma, fg_coef = 1, lamnda_depth = 0,
        # lambda_constr = 0, lambda_reg2 = 0

        print energy_para

        kappa = energy_para.get('kappa', 1)
        gamma = energy_para.get('gamma', 1)
        fg_coef = energy_para.get('fg_coef', 1)

        lambda_reg = energy_para.get('lambda_reg', 1)
        lambda_reg2 = energy_para.get('lambda_reg2', 0)
        lambda_depth = energy_para.get('lambda_depth', 0)
        lambda_constr = energy_para.get('lambda_constr', 0)

        seg_folder = para['seg_folder']
        num_segments = para['num_segments']

        depth_folder = '/VladlenDense/segs{:d}_lreg{:g}_kpa{:g}_gamma{:g}_lregtheta{:g}_ldepth{:g}_lconstr{:g}_fg_coef{:g}/'.\
            format(num_segments, lambda_reg, kappa, gamma, lambda_reg2, lambda_depth, lambda_constr, fg_coef)

        results_folder = seg_folder + depth_folder

        para['results_folder'] = results_folder

        optim_results = results_folder + '/optim_results.pkl'

        try:
            util.ensure_dir(optim_results)
            with open(optim_results, 'r') as f:
                thetas, inv_scales = pickle.load(f)
        except:
            thetas = cp.Variable(len(np.unique(superpixels)), 3)
            inv_scales = cp.Variable( objects_num )

            # thetas_test = np.ones((len(np.unique(superpixels)), 3))
            # inv_scales_test = 1.0 / scales

            # regularization term
            num_reg = superpixel_edges.shape[1]

            edge_weights = np.exp( - kappa * np.sum( np.square( sp_colors[:, superpixel_edges[2,:] ].T -
                                                                sp_colors[:, superpixel_edges[5,:] ].T ), axis=1) )
            if(fg_coef != 1):

                v1 = np.round(self.sp_centers[1, superpixel_edges[2,:]]).astype(np.int32)
                u1 = np.round(self.sp_centers[0, superpixel_edges[2,:]]).astype(np.int32)

                v2 = np.round(self.sp_centers[1, superpixel_edges[5, :]]).astype(np.int32)
                u2 = np.round(self.sp_centers[0, superpixel_edges[5, :]]).astype(np.int32)

                fg_mask = np.logical_and( dense_labels[ v1, u1 ] != self.bg_label,
                                          dense_labels[ v2, u2 ] != self.bg_label )

                edge_weights[fg_mask] = edge_weights[fg_mask] * fg_coef

                color = edge_weights.reshape((-1,1)) / np.max(edge_weights) * np.ones(( num_reg, 3 ))
                util.plot_superpixels(self.sp_mcolor_image, sp_edges= superpixel_edges[[2,5]],
                                      sp_centers=self.sp_centers, color= color  )

            """
            regularization term, an option is to enforce the constraint that boundary pixels of neighboring superpixels
            should be close
            """
            objective_reg = cp.Minimize(
                lambda_reg * cp.norm(
                    cp.mul_elemwise(edge_weights,
                                    (cp.sum_entries(
                                        cp.mul_elemwise(
                                            np.vstack( (superpixel_edges[1,:],
                                                        superpixel_edges[0,:],
                                                        np.ones((1,num_reg)) ) ).T,
                                            thetas[ superpixel_edges[2, :], : ]), axis=1)
                                     -
                                     cp.sum_entries(
                                         cp.mul_elemwise(
                                             np.vstack( (superpixel_edges[4,:],
                                                         superpixel_edges[3,:],
                                                         np.ones((1,num_reg)) ) ).T,
                                             thetas[ superpixel_edges[5, :], : ]), axis=1))
                                    )
                )
            )

            """ another option is to assume neighboring superpixels have similar plane parameters """
            objective_reg2 = cp.Minimize(
                lambda_reg2 * cp.norm(
                    thetas[ superpixel_edges[2, :], : ] - thetas[ superpixel_edges[5,:] ],
                    "fro"
                )
            )

            """
            a new regularization term, this is a constraint to stop superpixels to spread out in depth direction,
            just penalize the norm of the first two elements of thetas, the amount of stretching in depth
            """
            objective_reg_depth = cp.Minimize( lambda_depth * (cp.norm(thetas[:,0]) + cp.norm(thetas[:,1])) )

            # data term
            uv = self.W[0:2, :].astype(int)
            num_data = uv.shape[1]

            data_weights = np.exp(-gamma * self.fitting_cost)

            objective_data = cp.Minimize(
                cp.norm(
                    cp.mul_elemwise(data_weights,
                                    cp.sum_entries( cp.mul_elemwise(np.vstack( (uv[0], uv[1],np.ones((1, num_data)) ) ).T,
                                                                    thetas[ superpixels[ uv[1], uv[0] ], : ]), axis=1)
                                    -
                                    cp.mul_elemwise(self.inv_depths, inv_scales[ dense_labels[uv[1], uv[0]] ] )
                                    )
                )
            )

            """
            set up constraints for the optimization problem, we enforce that foreground superpixels should always
            be in front of background superpixels
            """

            constraints = {}
            constraints_objective = {}

            # for i in range(object_edges.shape[1]):
            for i in range(object_edges[0].size):

                if(np.logical_and( (object_edges[1, i] == self.bg_label), (object_edges[1, i] != object_edges[3,i]) )):

                    bg_sp = object_edges[0, i]
                    fg_sp = object_edges[2, i]

                elif(np.logical_and( (object_edges[3, i] == self.bg_label), (object_edges[1, i] != object_edges[3,i]) )):

                    bg_sp = object_edges[2, i]
                    fg_sp = object_edges[0, i]

                bg_uv = np.where(superpixels == bg_sp)
                fg_uv = np.where(superpixels == fg_sp)

                bg_idepths = thetas[bg_sp,:] * np.vstack( ( bg_uv[1], bg_uv[0], np.ones_like(bg_uv[0]) ) )
                fg_idepths = thetas[fg_sp,:] * np.vstack( ( fg_uv[1], fg_uv[0], np.ones_like(fg_uv[0]) ) )

                constraints[i] = cp.max_entries(bg_idepths) <= cp.min_entries(fg_idepths)

                constraints_objective[i] = cp.Minimize(
                    lambda_constr * cp.pos( cp.max_entries(bg_idepths) - cp.min_entries(fg_idepths) ) )

            """ start from data term and add up each regularization term gradually """
            objective = objective_data

            if(lambda_reg > 0):
                objective += objective_reg

            if(lambda_reg2 > 0):
                objective += objective_reg2

            if(lambda_depth > 0):
                objective += objective_reg_depth

            if(lambda_constr > 0):
                for key, obj in constraints_objective.iteritems():
                    objective += obj

            constraint = []

            for key, value in constraints.iteritems():
                constraint.append( value )

            for i in range(objects_num):
                constraint.append( inv_scales[i] > 0 )

            constraint.append( inv_scales[self.bg_label] == 1 )

            # """also add the constraint that the scales of other objects is also 1 """
            # for i in range(objects_num):
            #     constraint.append( inv_scales[i] == 1 )

            prob = cp.Problem(objective, constraint)

            # retrieve the results
            prob.solve(verbose=True, solver=cp.SCS)
            # prob.solve(verbose=True, solver=cp.SCS, scale=10)

            print "data cost : {:g}".format(objective_data.value)
            print "reg cost : {:g}".format(objective_reg.value)

            # check if constraint is satisfied
            constraint_values = {}
            for k,c in enumerate(constraint):
                # constraint_text = '%s %s %s' % (c.left.value, c.type, c.right)
                # print '%s becomes %s which is %s' % (c, constraint_text, eval(constraint_text))
                constraint_values[k] = c.value

            with open(optim_results, 'w') as f:
                pickle.dump((thetas, inv_scales), f, True)

            # depth_constraints2 = []
            # fg_idepths = []
            # bg_idepths = []
            # # for i in range(object_edges.shape[1]):
            # for i in range(object_edges[0].size):
            #
            #     if (np.logical_and((object_edges[1, i] == self.bg_label), (object_edges[1, i] != object_edges[3, i]))):
            #
            #         bg_sp = object_edges[0, i]
            #         fg_sp = object_edges[2, i]
            #
            #     elif (np.logical_and((object_edges[3, i] == self.bg_label), (object_edges[1, i] != object_edges[3, i]))):
            #
            #         bg_sp = object_edges[2, i]
            #         fg_sp = object_edges[0, i]
            #
            #     bg_uv = np.where(superpixels == bg_sp)
            #     fg_uv = np.where(superpixels == fg_sp)
            #
            #     bg_idepths2 = (bg_uv[1] * np.array(thetas.value[bg_sp, 0]) +
            #                    bg_uv[0] * np.array(thetas.value[bg_sp, 1]) +
            #                    np.array(thetas.value[bg_sp, 2]))
            #     fg_idepths2 = (fg_uv[1] * np.array(thetas.value[fg_sp, 0]) +
            #                    fg_uv[0] * np.array(thetas.value[fg_sp, 1]) +
            #                    np.array(thetas.value[fg_sp, 2]))
            #
            #     bg_idepths.append( np.max(bg_idepths2) )
            #     fg_idepths.append( np.min(fg_idepths2) )
            #
            #     depth_constraints2.append( bg_idepths[-1] <= fg_idepths[-1] )

        nH, nW, nC = self.ref_image.shape

        u = np.array(range(nW))
        uu = np.tile(u,(nH, 1) ).reshape(nH * nW, 1)
        v = np.array(range(nH)).reshape(nH,1)
        vv = np.tile(v,(1, nW) ).reshape(nH * nW, 1)
        ww = np.ones_like(uu)

        sp = superpixels[vv, uu]

        idepth = (uu * np.array(thetas.value[sp, 0]) +
                  vv * np.array(thetas.value[sp, 1]) +
                  ww * np.array(thetas.value[sp, 2]))

        depth_map = (1.0 / idepth).reshape(nH, nW)
        depth_constraints = []
        fg_depths = []
        bg_depths = []
        fg_sps = []
        bg_sps = []

        # check if the order constraint has been satisfied
        #for i in range(object_edges.shape[1]):
        for i in range(object_edges[0].size):

            if(np.logical_and( (object_edges[1, i] == self.bg_label), (object_edges[1, i] != object_edges[3,i]) )):

                bg_sp = object_edges[0, i]
                fg_sp = object_edges[2, i]
                fg_sps.append(fg_sp)
                bg_sps.append(bg_sp)

            elif(np.logical_and( (object_edges[3, i] == self.bg_label), (object_edges[1, i] != object_edges[3,i]) )):

                bg_sp = object_edges[2, i]
                fg_sp = object_edges[0, i]
                fg_sps.append(fg_sp)
                bg_sps.append(bg_sp)

            bg_uv = np.where(superpixels == bg_sp)
            fg_uv = np.where(superpixels == fg_sp)

            bg_depths.append( np.min(depth_map[bg_uv[0], bg_uv[1]]) )
            fg_depths.append( np.max(depth_map[fg_uv[0], fg_uv[1]]) )
            depth_constraints.append( bg_depths[-1] > fg_depths[-1] )

        dp_min = self.para.get('dp_min', 5)
        dp_max = self.para.get('dp_max', 95)

        depth_min = np.percentile(depth_map, dp_min)
        depth_max = np.percentile(depth_map, dp_max)
        # depth_min = np.percentile(depth_map, 20)
        # depth_max = np.percentile(depth_map, 90)

        depth_map[ depth_map <= depth_min ] = depth_min
        depth_map[ depth_map >= depth_max ] = depth_max

        return  depth_map, np.vstack((np.array(fg_sps), np.array(bg_sps))), np.array(inv_scales.value)

    def get_ground_truth_depth(self, para):

        depth_gt = {}

        expr = para['expr']

        if(expr == 'vkitti'):

            depth_gt_file = para['depth_gt_file']
            depth_map_gt = cv2.imread(depth_gt_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            # conversion from centimeter to meter
            depth_map_gt = depth_map_gt / 100.0

            mask = depth_map_gt < 20

            depth_gt['is_dense'] = 1
            depth_gt['mask'] = mask
            depth_gt['depth_map_gt'] = depth_map_gt
            depth_gt['plot_mask'] = depth_map_gt < 655.35

            depth_gt['invd_min'] = 0.01
            depth_gt['invd_max'] = 0.25
            depth_gt['used_fixed'] = 1

        elif(expr == 'sintel'):

            depth_gt_file = para['depth_gt_file']
            depth_map_gt = sio.depth_read(depth_gt_file)

            mask = depth_map_gt < 20

            depth_gt['is_dense'] = 1
            depth_gt['mask'] = mask
            depth_gt['depth_map_gt'] = depth_map_gt
            depth_gt['plot_mask'] = depth_map_gt < np.inf

            depth_gt['invd_min'] = 0.01
            depth_gt['invd_max'] = 2
            depth_gt['used_fixed'] = 0

        elif(expr == 'kitti'):

            bin_gt_file = para['bin_gt_file']
            Tr = para['Tr']

            nH, nW, nC = self.ref_image.shape

            depth_map_gt_interp, uvd = depth_util.get_kitti_depth_gt(self.K, Tr, nH, nW, bin_gt_file,
                                                                     depth_min = 10, depth_max=20)

            depth_gt['is_dense'] = 0
            depth_gt['depth_map_gt'] = depth_map_gt_interp
            depth_gt['uvd'] = uvd

            plot_mask = np.zeros_like(depth_map_gt_interp).astype(np.bool)
            plot_mask[nH/3+30:nH,:] = True
            depth_gt['plot_mask'] = plot_mask

            depth_gt['invd_min'] = 0.01
            depth_gt['invd_max'] = 0.25
            depth_gt['used_fixed'] = 1

        return depth_gt

    def evaluation(self, depth_map, depth_gt, para, method = 'Vladlen', epara = {}, scale_optim = 0):

        if(method == 'SparseInterp' or method == 'SparseInterpRigid'):
            results_folder = para['seg_folder'] + '/' + method + '/'
            if(scale_optim):
                results_folder = results_folder + '/ScaleOptim/'
        elif(method == 'Vladlen'):
            results_folder = para['results_folder']

        is_dense = depth_gt['is_dense']
        depth_map_gt = depth_gt['depth_map_gt']

        if(is_dense):
            mask = depth_gt['mask']
            error_mre, error_rmse, error_log10, depth_map_mre, outlier_mask, global_scale = \
                depth_util.evaluate_dense(depth_map, depth_map_gt, mask)
            # error_mre, error_rmse, error_log10, depth_map_mre, outlier_mask, global_scale = \
            #     depth_util.evaluate_dense(depth_map, depth_map_gt, mask,
            #                               K = self.K, check_3d = 1,
            #                               ref_img = self.ref_image)
        else:
            nH, nW = depth_map.shape
            u, v, d = depth_gt['uvd']
            mask = np.zeros_like(depth_map).astype(np.bool)
            mask[ np.maximum( 0, np.minimum(nH-1, v.astype(np.int32)) ),
                  np.maximum( 0, np.minimum(nW-1, u.astype(np.int32)) ) ] = True
            error_mre, error_rmse, error_log10, depth_map_mre, outlier_mask, global_scale = \
                depth_util.evaluate_sparse(depth_map, u ,v, d)
            # error_mre, error_rmse, error_log10, depth_map_mre, outlier_mask, global_scale = \
            #     depth_util.evaluate_sparse(depth_map, u ,v, d,
            #                                K = self.K, check_3d = 1,
            #                                ref_img = self.ref_image)

        depth_results = {'error_mre': error_mre, 'error_rmse': error_rmse, 'error_log10': error_log10,
                         'depth_map_mre': depth_map_mre}

        logging.info( method )

        if (method == 'SparseInterp' or method == 'SparseInterpRigid'):
            logging.info( 'global scaling optimization: {:g}'.format( scale_optim ) )

        if( method == 'Vladlen'):
            logging.info( epara )

        logging.info( 'error_mre: {:g}'.format(error_mre)  )
        logging.info( 'error_rmse: {:g}'.format(error_rmse) )
        logging.info( 'error_rmse: {:g}'.format(error_rmse) )

        try:
            util.ensure_dir(results_folder)
            with open('{:s}/results.pkl'.format(results_folder),"wb") as f:
                pickle.dump(depth_results, f, True)
            with open('{:s}/results.mat'.format(results_folder),"wb") as f:
                scipy.io.savemat(f, mdict=depth_results)
        except:
            print "saving error"

        plot_mask = depth_gt['plot_mask']

        rm_mask = np.logical_or( depth_map_mre == np.inf,
                                 depth_map_mre == -np.inf,
                                 np.isnan(depth_map_mre) )

        inv_depth_map_gt = 1.0 / depth_map_gt
        inv_depth_map = 1.0 / depth_map_mre

        inv_depth_map_gt[~plot_mask] = 1
        inv_depth_map[rm_mask] = 1

        # plt.imshow(inv_depth_map_gt, vmin = depth_gt['invd_min'], vmax = depth_gt['invd_max'], cmap = 'gray')
        # plt.imshow(inv_depth_map, vmin = depth_gt['invd_min'], vmax = depth_gt['invd_max'], cmap = 'gray')

        if(not depth_gt['used_fixed']):
            depth_gt['invd_min'] = np.min(inv_depth_map_gt)
            depth_gt['invd_max'] = np.max(inv_depth_map_gt)


        util.save_depth_map(inv_depth_map, results_folder + "idepth_map_mre.png",
                            vmin=depth_gt['invd_min'], vmax=depth_gt['invd_max'], cmap='gray')
        util.save_depth_map(inv_depth_map_gt, results_folder + "idepth_map_gt.png",
                            vmin=depth_gt['invd_min'], vmax=depth_gt['invd_max'], cmap='gray')

        inv_depth_map_gt[~plot_mask] = 0
        inv_depth_map[rm_mask] = 1

        # plt.imshow(inv_depth_map_gt, vmin = depth_gt['invd_min'], vmax = depth_gt['invd_max'], cmap = 'gray_r')
        # plt.imshow(inv_depth_map, vmin = depth_gt['invd_min'], vmax = depth_gt['invd_max'], cmap = 'gray_r')
        # plt.imshow(np.hstack((inv_depth_map, inv_depth_map_gt)), cmap=plt.get_cmap('gray'))

        util.save_depth_map(inv_depth_map, results_folder + "idepth_map_mre_r.png",
                            vmin=depth_gt['invd_min'], vmax=depth_gt['invd_max'], cmap='gray_r')
        util.save_depth_map(inv_depth_map_gt, results_folder + "idepth_map_gt_r.png",
                            vmin=depth_gt['invd_min'], vmax=depth_gt['invd_max'], cmap='gray_r')

        # cv2.imwrite(results_folder + "depth_map_gt.png", depth_map_gt)
        # cv2.imwrite(results_folder + "depth_map_mre.png", depth_map_mre)
        #
        # fig = plt.figure("depth evaluation")
        # ax1 = fig.add_subplot(2, 2, 1)
        # ax1.imshow(depth_map_mre)
        # plt.axis("off")
        # ax2 = fig.add_subplot(2, 2, 2)
        # ax2.imshow(depth_map_gt)
        # plt.axis("off")
        # ax3 = fig.add_subplot(2, 2, 3)
        # ax3.imshow(depth_map_gt - depth_map_mre)
        # plt.axis("off")
        # ax4 = fig.add_subplot(2, 2, 4)
        # ax4.imshow(mask)
        # # mng = plt.get_current_fig_manager()
        # # mng.frame.Maximize(True)
        #
        # manager = plt.get_current_fig_manager()
        # manager.window.showMaximized()
        #
        # plt.axis("off")
        # plt.show(block=False)
        # plt.tight_layout()
        #
        # #plt.waitforbuttonpress()
        #
        # fig.savefig(results_folder + 'depth_evaluation.png', bbox_inches='tight')
        #
        # plt.close('all')

        return global_scale

    def get_scores(self, method = 'Vladlen'):

        if(method == 'SparseInterp' or method == 'SparseInterpRigid'):

            results_folder = self.para['seg_folder'] + '/' + method + '/'

            try:
                with open(results_folder + '/results.pkl') as f:
                    results = pickle.load(f)
                scores = results['error_mre']
            except:
                scores = np.nan

        elif(method == 'Vladlen'):

            scores_vladlen = {}

            seg_folder = self.para['seg_folder']
            temp_folder = seg_folder + '/VladlenDense/'
            results_folder_list = next(os.walk(temp_folder))[1]

            for result in results_folder_list:

                results_folder = temp_folder + result

                try:
                    if (os.path.isfile(results_folder + '/results.pkl')):
                        with open(results_folder + '/results.pkl') as f:
                            results = pickle.load(f)
                            scores_vladlen[result] = results['error_mre']
                except:
                    scores_vladlen[result] = np.nan

            scores = scores_vladlen

        return scores

