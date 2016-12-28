import os.path

import numpy as np

import scipy.io

import cPickle as pickle

from video_popup.utils.transformations import quaternion_from_matrix
from video_popup.utils.transformations import quaternion_matrix

import cv2

from video_popup.utils.util import rotation2dTo3d

import sys
sys.path.append('../../libs/ceres-solver-mex_new/')
from cs_ba_python import cs_ba_python

from video_popup.utils import util
import networkx as nx

from video_popup.depth_reconstruction import depth_util

class OrthoReconstruction(object):

    def __init__(self, W, Z, s, labels, assign, images, colors, para, save_path = ''):

        self.data = {"W":W, "Z":Z, "s":s, "labels":labels, 'colors': colors,
                     "assign": assign, "images": images}

        self.data['Zdata'] = np.hstack((self.data['Z'],
                                        self.data['Z'])).reshape(self.data['W'].shape[0],
                                                                 self.data['W'].shape[1])

        self.para = para

        self.save_path = save_path


    def initialize(self, min_pnts = 4):

        # create the parts belong to this object
        points = {}
        rotations = {}
        translations = {}
        shapes = {}
        colors = {}

        W = self.data['W']
        Z = self.data['Zdata']
        assign = self.data['assign']
        para = self.para

        colors_obj = self.data['colors']

        self.num_frames = W.shape[0] / 2
        self.num_parts = assign.shape[0]

        init_file = self.save_path + '/init.pkl'

        try:
            util.ensure_dir(init_file)
            with open(init_file, 'r') as f:
                points, rotations, translations, shapes, colors = pickle.load(f)
        except:
            for i in range(self.num_parts):

                points_i = np.where(assign[i,:] > 0)[0]
                lambdas_i = assign[i,:][assign[i,:] > 0]

                points[i] = (points_i, lambdas_i)

                # Marques Costeria Factorization
                # only reconstruct those frames with enough points

                Wi = W[:, points_i]
                Zi = Z[:, points_i]

                filter = np.sum(Zi, axis=1) >= min_pnts
                # filter_wi = np.hstack((filter.reshape(-1,1), filter.reshape(-1,1))).reshape(-1)

                # # R, S, T, W_res = self.marques_costeria(Wi[filter_wi,:], Zi[filter,:], lambdas_i, para)
                R, S, T, W_res = self.marques_costeria(Wi[filter,:], Zi[filter,:], lambdas_i, para)
                #
                # Zt = Zi[filter, :]
                # Zt = Zt[0::2, :]
                #
                # RST = np.dot(R,S) + T.reshape((-1,1))
                # # util.plot_traj_test(Wi[filter, :], W_res, Zt, self.data['images'])
                # util.plot_traj_test(Wi[filter, :], RST, Zt, self.data['images'])

                rotations[i] = R
                translations[i] = T
                shapes[i] = S

                colors[i] = colors_obj[:, points_i]

            data = (points, rotations, translations, shapes, colors)
            with open(init_file, 'w') as f:
                pickle.dump(data, f, True)

        self.points = points
        self.rotations = rotations
        self.translations = translations
        self.shapes = shapes
        self.colors = colors

    def rigid_fact(self, Wc, epsilon, n_iter, R, scale_ortho = False):

        F, P = Wc.shape

        diff = np.inf; k = 0
        M = np.zeros((F, 3))

        while diff > epsilon and k <= n_iter:
            Rk = R
            for f in range(F/2):
                Rf = R[2*f:2*f+2,:]
                U, D, Vt = np.linalg.svd(Rf, full_matrices = False)
                if(scale_ortho):
                    M[2*f:2*f+2,:] = (D[0] + D[1])/2 * np.dot( U, Vt )
                else:
                    M[2*f:2*f+2,:] = np.dot( U, Vt )
            S = np.dot( np.linalg.pinv(M) , Wc )
            R = np.dot( Wc , np.linalg.pinv(S) )
            k = k + 1
            diff = np.sum(np.abs(R - Rk)) / (F/2)

        return M, S, R

    def marques_costeria(self, W, Z, lambdas, para):

        epsilon = para.MC1; n_iter = para.MC2; n_iter2 = para.MC3

        F, P = W.shape
        Z_hat = 1 - Z

        W[Z == 0] = 0
        Wk = np.reshape( np.sum(W, axis=1)/ np.sum(Z, axis=1), (F,1) ) * Z_hat + W
        ref_points = np.nonzero(np.sum(Z, axis=0) == F)[0]

        num_missing = np.count_nonzero(Z)

        if(num_missing == 0):   ## perfect, no missing point
            T = np.sum(W, axis=1)/ np.sum(Z, axis=1)
        elif( len(ref_points) > 0 ):  ## the first fully visible point as ref
            T = W[:, ref_points[0] ]
        else: ## take the average as ref
            T = np.mean(Wk, axis=1)

        Wc = Wk - np.reshape(T, (F,1))

        U, D, Ut = np.linalg.svd( np.dot(Wc, Wc.T), full_matrices=True)

        U = U[:, 0:3]
        D = np.sqrt(D[0:3])

        R = U*D

        if(num_missing == 0):

            M, S, R = self.rigid_fact(Wc, epsilon, n_iter2, R)

            W_res = Wc + np.reshape(T, (F,1))

            return M, S, T, W_res

        k = 1
        diff = np.inf

        while diff > epsilon and k <= n_iter:

            Wk = Wc
            M, S, R = self.rigid_fact(Wc, epsilon, n_iter2, R)
            Wk_new = np.dot(M, S) * Z_hat + Wc * Z

            diff =  np.sum(np.abs(Wk_new - Wk)) / num_missing

            delta_T = np.mean(Wk_new, axis = 1)
            Wc = Wk_new - np.reshape(delta_T, (F, 1))

            T = T + delta_T

            k += 1

        W_res = Wc + np.reshape(T, (F,1))

        return  M, S, T, W_res

    def bundle_adjustment(self):

        ba_file = self.save_path + '/ba.pkl'

        try:
            util.ensure_dir(ba_file)
            with open(ba_file, 'r') as f:
                self.rotations, self.translations, self.shapes = pickle.load(f)
        except:

            points = self.points
            translations = self.translations
            rotations = self.rotations
            shapes = self.shapes
            colors = self.colors

            for i in points.keys():

                print "doing bundle adjustment"
                print i

                colors_i = colors[i]

                pi = points[i][0]
                T = translations[i]
                R = rotations[i]
                S = shapes[i]

                #util.plot_3d_point_cloud_vispy(S.T, colors_i.T)

                W = self.data['W'][:, pi]
                Z = self.data['Z'][:, pi]
                Zdata = np.reshape(np.hstack((Z, Z)), (-1, len(pi)))

                F2, P = W.shape
                F = F2 / 2

                verbose = 1

                q = np.zeros((F, 4))
                for f in range(F):
                    q[f, :] = quaternion_from_matrix(
                        rotation2dTo3d(R[2*f:2*f+2, :]))

                q = np.hstack((q, np.reshape(T, (F, 2))))

                P0 = np.hstack((np.reshape(q, (-1)), np.reshape(S, (-1), order='F')))

                new_prior_terms = np.array([self.para.alpha_s,
                                            self.para.alpha_z,
                                            self.para.alpha_prior])

                sfmCase = 'motstr'
                cs_error_type = 'OrthoReprojectionErrorWithQuaternions'

                ba_model = self.para.ba_model
                # use motion prior for all consecutive frames
                mprior_mask = np.ones(F - 1)
                # optimize camera for all frames
                cconst_mask = np.zeros(F)

                knum = 0; kmask = np.array([0.0])

                # save data for testing in matlab
                # data = {'n': P, 'ncon': 0, 'm': F, 'mcon': 0, 'vmask': Z, 'p0': P0, 'cnp': 6,
                #         'pnp': 3, 'x':W.T[Zdata.T], 'errorType': cs_error_type.encode(),
                #         'sfmCase': sfmCase.encode(), 'verbose': 1, 'knum': knum, 'kmask': kmask,
                #         'prior_model': ba_model, 'prior_para': new_prior_terms, 'mprior': mprior_mask,
                #         'cconst_mask': cconst_mask}
                #
                # with open('ba_matlab_test.mat', "wb") as f:
                #     scipy.io.savemat(f, mdict = data)

                Pout = cs_ba_python(P, 0, F, 0, Z, P0, 6, 3, W.T[Zdata.T],
                                    cs_error_type.encode(), sfmCase.encode(),
                                    verbose, knum, kmask, ba_model, new_prior_terms,
                                    mprior_mask, cconst_mask)


                RT = np.reshape(Pout[0:6*F], (F, 6))
                for f in range(F):
                    R[2*f:2*f+2, :] = quaternion_matrix(RT[f,0:4])[:2,:3]

                T = np.reshape(RT[:,4:], (-1,1))
                S = np.reshape(Pout[6*F:], (len(pi), 3)).T

                self.rotations[i] = R
                self.translations[i] = T
                self.shapes[i] = S

                # util.plot_3d_point_cloud_vispy(S.T, colors_i.T)

            data = (self.rotations, self.translations, self.shapes)
            with open(ba_file, 'w') as f:
                pickle.dump(data, f, True)

    def refine(self):

        # default choice at the moment
        self.bundle_adjustment()

        """
        check 2d reprojection error after bundle adjustment
        """
        # for i in self.rotations.keys():
        #
        #     W_ba = np.dot(self.rotations[i], self.shapes[i]) + self.translations[i].reshape((-1,1))
        #
        #     W = self.data['W']
        #     Z = self.data['Zdata']
        #
        #     Wi = W[:, self.points[i][0]]
        #     Zi = Z[:, self.points[i][0]]
        #     Zi = Zi[0::2,:]
        #
        #     util.plot_traj_test(Wi, W_ba, Zi, self.data['images'])

        """ save stitching result using depth based to solve spatial flipping """
        stitching_file = self.save_path + '/stitched.pkl'
        self.stitching(stitching_file, rot_based=0)

        """ save stitching result using rotation based method to solve spatial flipping """
        stitching_file = self.save_path + '/stitched_rot_based.pkl'
        self.stitching(stitching_file, rot_based=1)

        """
        check 2d reprojection error after stitching, this should be the same as before stitching
        """
        # for i in self.rotations.keys():
        #
        #     W3d = np.dot(self.rotations3d[i], self.shapes[i]) + self.translations3d[i].reshape((-1,1))
        #     W_stitch = np.hstack((W3d[0::3,:], W3d[1::3,:])).reshape((-1, self.shapes[i].shape[1]))
        #
        #     W = self.data['W']
        #     Z = self.data['Zdata']
        #
        #     Wi = W[:, self.points[i][0]]
        #     Zi = Z[:, self.points[i][0]]
        #     Zi = Zi[0::2,:]
        #
        #     util.plot_traj_test(Wi, W_stitch, Zi, self.data['images'])

    def stitching(self, stitching_file, rot_based = 0):

        """
        Assume that at this stage there is no temporal flipping within each patch. In other words,
        if the reconstruction is degenerate and the result is almost planar, we might have some frames
        wrongly flipped(due to the ambiguity the rotation is wrong, nonetheless the shape is fine).
        In this function, we assume this issue has been fixed and we focus on spatial flipping
        and depth ambiguity here.

        using bruteforce search for best spatial flippings and then use greedy approach to generate
        registered depth values
        """

        if (os.path.exists(stitching_file)):
            return

        """
        convert 2d data to 3d
        """
        self.rotations3d = {}
        self.translations3d = {}
        self.shapes_all = {}
        self.depths_all = {}

        for i in self.rotations.keys():

            self.rotations3d[i] = rotation2dTo3d( self.rotations[i] )
            self.translations3d[i] = np.hstack( ( self.translations[i].reshape(( self.num_frames , 2 )),
                                                  np.zeros((self.num_frames, 1)) )).reshape(-1)
            self.shapes_all[i] = self.get_shapes_all(self.rotations3d[i],
                                                     self.translations3d[i],
                                                     self.shapes[i])
            self.depths_all[i] = self.shapes_all[i][2::3,:]

        pre_stitching_file = self.save_path + '/pre_stitching.pkl'

        if( not os.path.exists(pre_stitching_file) ):
            data = {'rotations3d': self.rotations3d, 'translations3d': self.translations3d,
                    'shapes': self.shapes, 'colors': self.colors, 'points': self.points,
                    'images': self.data['images']}
            with open(pre_stitching_file, 'w') as f:
                pickle.dump(data, f, True)

        """
        get edges between patches and the cost on these edges
        """
        edges = {}
        G = nx.Graph()

        for i in self.rotations.keys():
            points_i = self.points[i][0]
            G.add_node(i,
                       depths=np.copy(self.depths_all[i]),
                       points=np.copy(points_i),
                       colors=self.colors[i],
                       uv=np.hstack((self.shapes_all[i][0::3,:],
                                     self.shapes_all[i][1::3,:])).reshape( (-1,len(points_i)) ),
                       shared_pnts=np.array([]))

        for i in self.rotations.keys():
            points_i = self.points[i][0]
            for j in self.rotations.keys():
                if( j > i ):
                    points_j = self.points[j][0]
                    common_number = len(np.intersect1d(points_i, points_j))
                    if( common_number > 0 ):
                        edges[(i, j)] = common_number
                        G.add_edge(i,j, weight = common_number)

        if(rot_based == 0):
            edges_cost = self.prepare_stitching_cost(edges)
        else:
            edges_cost = self.prepare_stitching_cost_rot(edges)

        """
        brute force search
        """
        cost = np.inf
        flip = np.zeros(self.num_parts)

        for loop in range(2**(self.num_parts-1)):

            flip_v = np.zeros(self.num_parts)
            loop_b = bin(loop)[2:]
            for ind in range(len(loop_b)):
                flip_v[-1-ind] = int(loop_b[-1-ind])

            loop_cost = 0
            for edge in edges.keys():
                i, j = edge

                if( flip_v[i] == flip_v[j] ):
                    loop_cost += edges_cost[edge][0]
                else:
                    loop_cost += edges_cost[edge][1]

            # loop_cost = self.stitching_cost(self.shapes_all, flip_v, edges)

            if(loop_cost < cost):
                cost = loop_cost
                flip = flip_v

        W_2ds = {}

        """
        plot the results before stitching
        """
        # self.plot_results(G, [0, 1])

        """
        flipping patches based on above results
        """
        for i in self.rotations.keys():

            # RS_2d = np.dot(self.rotations[i], self.shapes[i])
            W_2d = np.dot(self.rotations[i], self.shapes[i]) + self.translations[i].reshape((-1, 1))
            W_2ds[i] = W_2d

            if(flip[i]):

                self.rotations3d[i][:, 2] = -self.rotations3d[i][:, 2]
                self.rotations3d[i][2::3,:] = -self.rotations3d[i][2::3,:]

                self.shapes[i][2,:] = -self.shapes[i][2,:]

                # self.shapes_all[i][2::3,:] *= -1
                self.depths_all[i] *= -1

            # RS_3d = np.dot(self.rotations3d[i], self.shapes[i])
            # W_3d = np.dot(self.rotations3d[i], self.shapes[i]) + self.translations3d[i].reshape((-1, 1))
            # diff = np.hstack((W_3d[0::3,:], W_3d[1::3,:])).reshape(( RS_2d.shape[0], RS_2d.shape[1] )) - W_2d

        """
        compute the depths for each patch
        """
        points, depths = self.depth_registration(G, flip)

        # self.plot_results(G, [1])
        # self.plot_results(G, G.node.keys()[0])

        """
        now we can retrieve the corresponding registered depths for each patch
        """
        for i in self.rotations.keys():

            points_i = self.points[i][0]
            depths_i = self.depths_all[i]

            id = np.in1d(points, points_i)
            depths_res = depths[:, id]

            self.translations3d[i][2::3]  = np.mean(depths_res - depths_i, axis=1).reshape(-1)

            # W_3d = np.dot(self.rotations3d[i], self.shapes[i]) + self.translations3d[i].reshape((-1, 1))
            # diff = np.hstack((W_3d[0::3, :], W_3d[1::3, :])).reshape((-1, depths_id.shape[1])) - W_2ds[i]

            self.shapes_all[i] = self.get_shapes_all(self.rotations3d[i],
                                                     self.translations3d[i],
                                                     self.shapes[i])
            self.depths_all[i] = self.shapes_all[i][2::3,:]


        """ saving data """

        data = {'rotations3d': self.rotations3d, 'translations3d': self.translations3d,
                'shapes': self.shapes, 'colors': self.colors, 'points': self.points,
                'images': self.data['images'], 'shapes_all': self.shapes_all}

        with open(stitching_file, 'w') as f:
            pickle.dump(data, f, True)

    def plot_results(self, G, edge):

        """
        highlight overlapping points
        """

        if(len(edge) == 1):

            """
            plot node i
            """

            i = edge[0]

            uv_i = G.node[i]['uv']
            depths_i = G.node[i]['depths']
            colors_i = G.node[i]['colors']
            points_i = G.node[i]['points']

            vertices_i = np.hstack((uv_i.reshape((-1, 2 * depths_i.shape[1])),
                                    depths_i)).reshape((-1, depths_i.shape[1]))

            nframes = vertices_i.shape[0] / 3

            vertices = vertices_i.reshape((nframes, 3, -1))
            vertices = np.swapaxes(vertices, 1, 2)
            vertices = vertices.reshape((-1,3))

            colors = colors_i.T
            labels = np.ones((len(points_i), 1))*i

        else:

            """
            plot edge i-j
            """

            i = edge[0]
            j = edge[1]

            # vertices_i = np.dot(self.rotations3d[i], self.shapes[i]) + \
            #              self.translations3d[i].reshape((-1, 1))
            # vertices_i = vertices_i[0:3, :]

            uv_i = G.node[i]['uv']
            depths_i = G.node[i]['depths']
            colors_i = G.node[i]['colors']
            points_i = G.node[i]['points']

            vertices_i = np.hstack( (uv_i.reshape((-1, 2*depths_i.shape[1])),
                                     depths_i) ).reshape((-1, depths_i.shape[1]))

            uv_j = G.node[j]['uv']
            depths_j = G.node[j]['depths']
            colors_j = G.node[j]['colors']
            points_j = G.node[j]['points']

            vertices_j = np.hstack( (uv_j.reshape((-1, 2*depths_j.shape[1])),
                                     depths_j) ).reshape((-1, depths_j.shape[1]))

            """
            retrieve common points
            """
            common, idi, idj =  self.my_intersect(points_i, points_j)

            vertices_i = vertices_i[:, idi]
            colors_i = colors_i[:, idi]
            vertices_j = vertices_j[:, idj]
            colors_j = colors_j[:, idj]

            vertices = np.concatenate((vertices_i.T, vertices_j.T))
            colors = np.concatenate((colors_i.T, colors_j.T))

            labels_i = np.ones((vertices_i.shape[1], 1))*i
            labels_j = np.ones((vertices_j.shape[1], 1))*j
            labels = np.concatenate((labels_i, labels_j))

            vertices = vertices.reshape((len(labels_i) + len(labels_j), -1, 3))
            nframes = vertices.shape[1]

            vertices = np.swapaxes(vertices, 0, 1)
            vertices = vertices.reshape((-1,3))

        vertices[:,2] = vertices[:,2] - np.min(vertices[:,2]) + 100

        depth_util.point_cloud_plot(vertices, colors, 0, 720, 1280, labels=labels, nframes=nframes)

    def depth_registration(self, G, flip = None):

        """ given depths input, returns registration result """
        """ flip the depths of each patch first if necessary """
        if(flip != None):
            for i in range(len(flip)):
                if(flip[i]):
                    G.node[i]['depths'] *= -1

        """do the registration with a greedy strategy, starting with the edge with most common points"""
        while(len(G.edges()) > 0):

            i, j, _ = sorted(G.edges(data=True), key=lambda (a, b, data): -data['weight'])[0]

            # self.plot_results(G, [i, j])

            uv_i = G.node[i]['uv']
            depths_i = G.node[i]['depths']
            points_i = G.node[i]['points']
            colors_i = G.node[i]['colors']

            uv_j = G.node[j]['uv']
            depths_j = G.node[j]['depths']
            points_j = G.node[j]['points']
            colors_j = G.node[j]['colors']

            common, idx, idy = self.my_intersect(points_i, points_j)

            if(len(points_j) > len(points_i)):
                i, j = j, i
                idx, idy = idy, idx
                uv_i, uv_j = uv_j, uv_i
                depths_i, depths_j = depths_j, depths_i
                points_i, points_j = points_j, points_i
                colors_i, colors_j = colors_j, colors_i

            ### whether point in points_j is in points_i
            mask = np.in1d(points_j, points_i, invert=True)
            pnts_to_add = points_j[mask]
            ids_to_add = np.where(mask)[0]
            shared_pnts = points_j[np.logical_not(mask)]

            if(flip != None):

                diff_ij = np.mean( depths_i[:, idx] - depths_j[:, idy], axis = 1).reshape((-1,1))
                depths_j = depths_j + diff_ij

            else:
                diff_ij = np.mean(depths_i[:, idx] - depths_j[:, idy], axis=1).reshape((-1,1))
                cost = np.linalg.norm(depths_i[:, idx] - depths_j[:, idy] - diff_ij, 'fro')
                diff_ij_flip = np.mean(depths_i[:, idx] + depths_j[:, idy], axis=1).reshape((-1,1))
                cost_flip = np.linalg.norm(depths_i[:, idx] + depths_j[:, idy] - diff_ij_flip, 'fro')

                if(cost_flip < cost):
                    depths_j = -depths_j + diff_ij_flip
                else:
                    depths_j = depths_j + diff_ij

            points_i = np.concatenate((points_i, pnts_to_add))
            depths_i = np.concatenate((depths_i, depths_j[:, ids_to_add]), axis=1)
            uv_i = np.concatenate((uv_i, uv_j[:, ids_to_add]), axis=1)
            colors_i = np.concatenate((colors_i, colors_j[:, ids_to_add]), axis=1)

            G.node[i]['uv'] = uv_i
            G.node[i]['points'] = points_i
            G.node[i]['depths'] = depths_i
            G.node[i]['colors'] = colors_i
            G.node[i]['shared_pnts'] = shared_pnts

            # plot node i
            # self.plot_results(G, [i])

            """ remove node j and update graphs """
            nbors = np.append(G.edge[i].keys(), G.edge[j].keys())
            to_remove = np.array([i,j])
            mask = np.in1d(nbors, to_remove, invert=True)
            nbors = np.unique(nbors[mask])
            G.remove_node(j)
            for nbor in nbors:
                points_nbor = G.node[nbor]['points']
                common_number = len(np.intersect1d(points_i, points_nbor))
                G.add_edge(i, nbor, weight = common_number)

        ### the node remained
        node = G.node[G.node.keys()[0]]

        points = node['points']
        depths = node['depths']
        idx = np.argsort(points)

        return points[idx], depths[:, idx]

    def my_intersect(self, x, y):

        common = np.intersect1d(x, y)

        index = np.argsort(x)
        sorted_x = x[index]
        sorted_index = np.searchsorted(sorted_x, common)
        idx = np.take(index, sorted_index, mode="clip")

        index = np.argsort(y)
        sorted_y = y[index]
        sorted_index = np.searchsorted(sorted_y, common)
        idy = np.take(index, sorted_index, mode="clip")

        return common, idx, idy

    def prepare_stitching_cost(self, edges):
        """
        compute all possible edge costs
        """
        edges_cost = {}

        for edge in edges.keys():

            i, j = edge
            points_i = self.points[i][0]
            points_j = self.points[j][0]

            common, idx, idy = self.my_intersect(points_i, points_j)
            depths_i = self.shapes_all[i][2::3, idx]
            depths_j = self.shapes_all[j][2::3, idy]

            diff_ij = np.mean(depths_i - depths_j, axis = 1).reshape((-1,1))
            cost = np.linalg.norm(depths_i - depths_j - diff_ij, 'fro')

            diff_ij_flip = np.mean(depths_i + depths_j, axis=1).reshape((-1,1))
            cost2 = np.linalg.norm(depths_i + depths_j - diff_ij_flip, 'fro')

            edges_cost[edge] = [cost, cost2]

        return edges_cost

    def prepare_stitching_cost_rot(self, edges):
        """
        compute all possible edge costs based on  rotation difference
        """
        edges_cost = {}

        for edge in edges.keys():

            i, j = edge
            rot_i = self.rotations3d[i]
            rot_j = self.rotations3d[j]

            rel_rot_i = self.get_relative_rot(rot_i)
            rel_rot_j = self.get_relative_rot(rot_j)

            cost = self.get_rot_diff(rel_rot_i, rel_rot_j)

            rel_rot_j_flip = rel_rot_j
            rel_rot_j_flip[:,2] *= -1
            rel_rot_j_flip[2::3, :] *= -1

            cost2 = self.get_rot_diff(rel_rot_i, rel_rot_j_flip)

            edges_cost[edge] = [cost, cost2]

        return edges_cost

    def get_relative_rot(self, rot):

        """
        compute the relative rotations between current frame and previous frame
        """
        F = rot.shape[0] / 3
        rel_rot = np.zeros_like(rot)

        rel_rot[0:3,:] = np.eye(3)

        for f in range(1,F):
            pre_rot = rot[3*f-3:3*f, :]
            cur_rot = rot[3*f:3*f+3, :]
            rel_rot[3*f:3*f+3,:] = np.dot(cur_rot, pre_rot.T)

        return  rel_rot

    def get_rot_diff(self, rot1, rot2):
        """
        compute the difference between two sets of rotations,
        this is computed as the angle needed to go from rot1 to rot2
        """

        cost = 0
        F = rot1.shape[0] / 3

        for f in range(F):

            r1 = rot1[3*f:3*f + 3, :]
            r2 = rot2[3*f:3*f + 3, :]
            r12 = np.dot(r2, r1.T)

            temp = cv2.Rodrigues(r12)[0]
            cost += np.linalg.norm(temp)

        return cost

    # def stitching_cost(self, shapes_all, flip_v, edges):
    #
    #     """
    #     just sum over the costs along all the edges
    #     """
    #
    #     cost = 0
    #
    #     for edge in edges.keys():
    #
    #         i, j = edge
    #         points_i = self.points[i][0]
    #         points_j = self.points[j][0]
    #         points_ij = np.intersect1d(points_i, points_j)
    #         depths_i = shapes_all[i][2::3, points_ij]
    #         depths_j = shapes_all[j][2::3, points_ij]
    #         if(flip_v[i]):
    #             depths_i = -depths_i
    #         if (flip_v[j]):
    #             depths_j = -depths_j
    #
    #         diff_ij = np.mean(depths_i - depths_j, axis = 1)
    #         cost += np.linalg.norm(depths_i - depths_j - diff_ij, 'fro')
    #
    #     return cost

    def get_shapes_all(self, R, T, S):

        S_all = np.zeros((self.num_frames*3, S.shape[1]))

        for f in range(self.num_frames):

            S_all[3*f:3*f+3] = np.dot(R[3*f:3*f+3], S) + T[3*f:3*f+3].reshape((3,-1))

        return  S_all



















        










