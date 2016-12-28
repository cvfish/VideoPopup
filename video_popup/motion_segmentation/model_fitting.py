import math
import numpy as np
import cPickle as pickle

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

from persp_model_fitting import PerspModelFitting

from video_popup.utils import util

import segmentation as graph_cut

class ModelFitting(object):

    def __init__(self, data, para, options):
        """
        copy over the data and and set up the parameters for graphcut
        """
        W, Z, M, s, images, labels = data
        self.data = {"W":W, "Z":Z, "M":M, "s":s, "labels":labels}
        self.images = images

        graphcut_para = para.graph_cut_para
        neighbors_num = graphcut_para.neighbors_num

        breaking_lambda, outlier_lambda, Ma, Mb, Mc = \
            graphcut_para.pointwise_breaking_lambda, \
            graphcut_para.pointwise_outlier_lambda, \
            graphcut_para.pairwise_breaking_ma, \
            graphcut_para.pairwise_breaking_mb, \
            graphcut_para.pairwise_breaking_mc

        pairwise_weight, pairwise_sigma = graphcut_para.pairwise_weight, \
                                          graphcut_para.pairwise_sigma

        pairwise_breaking = Mc / (1 + np.exp( (M[0:neighbors_num,:] -Ma) / Mb) )
        pointwise_breaking = breaking_lambda * np.mean(pairwise_breaking, axis = 0)
        pointwise_outlier = outlier_lambda * np.mean(pairwise_breaking, axis = 0)
        pairwise_connection = pairwise_weight * np.exp( - M[0:neighbors_num,:] / pairwise_sigma ** 2)

        # do some normalization for pointwise parameters
        pointwise_breaking = pointwise_breaking * np.sum(Z, axis = 0)
        pointwise_outlier = pointwise_outlier * np.sum(Z, axis = 0)

        # do some normalization for pairwise parameters
        overlap_weight = np.zeros(pairwise_breaking.shape)
        for i in range(neighbors_num):
            temp = np.logical_and(np.logical_and(Z, Z[:, np.maximum(s[i,:], 0)]), s[i,:] >= 0)
            overlap_weight[i,:] = np.sum(temp, axis = 0)

        pairwise_breaking = pairwise_breaking * overlap_weight
        pairwise_connection = pairwise_connection * overlap_weight

        self.params = para
        self.save_path = options['save_path']
        self.show_iter = options['show_iteration']

        self.pointwise_breaking = pointwise_breaking
        self.pointwise_outlier = pointwise_outlier
        self.pairwise_connection = pairwise_connection
        self.pairwise_breaking = pairwise_breaking

        # create fitting model object
        fitting_para = self.params
        persp_fitting_para = fitting_para.persp_fitting_para
        # ortho_fitting_para = fitting_para.ortho_fitting_para
        # subspace_fitting_para = fitting_para.subspace_fitting_para

        if(fitting_para.fitting_model == fitting_para.PERSP):
            self.model = PerspModelFitting(self.data, persp_fitting_para)
            # self.model = persp_model_fitting(self.data, persp_fitting_para)
        # elif(fitting_para.fitting_model == fitting_para.ORTHO):
        #     self.model = ortho_model_fitting(self.data, ortho_fitting_para)
        # elif(fitting_para.fitting_model == fitting_para.SUBSPACE):
        #     self.model = subspace_model_fitting(self.data, subspace_fitting_para)

        self.assignment = []
        self.labels = []
        self.outliers = []
        self.inliers = []

    def init_proposals(self):
        """
        propose initial models
        """
        s = self.data['s']
        F, P = self.data["Z"].shape
        num = self.params.init_proposal_num

        init_save_file = self.save_path + '/init{:g}/init_proposal_data.pkl'.format(num)

        try:

            util.ensure_dir(init_save_file)

            with open(init_save_file, 'r') as f:
                self.model_points, self.model_paras, self.unary_cost = pickle.load(f)

        except:

            step = np.maximum(math.floor(P / num), 1)

            model_points = {}

            # images = self.images
            # W = self.data['W']

            for i in range(num):

                neighbor = [p for p in s[:, i*step] if p > -1]
                #neighbor = np.asarray(neighbor)

                # img = mpimg.imread(images[0])
                # plt.imshow(img)
                # mask = W[0, neighbor] != 0
                # plt.scatter(x=W[0, neighbor[mask]], y=W[1, neighbor[mask]], c='red', s=5, marker='o')

                #model_points[i] = np.concatenate( ([i*step], neighbor ), axis=0).astype('int32')
                model_points[i] = [i*step] + neighbor
                #model_points.append( [i*step] + neighbor )

                # plt.scatter(x=W[0, i*step], y=W[1, i*step], c='green', s=5, marker='+')

            model_points, not_fittable_points = self.model.fittable_pnts(model_points)

            # """
            # plot all proposals
            # """
            # util.plot_traj(self.data['W'], self.data['Z'], self.images)
            # util.plot_proposals(self.data['W'], self.data['Z'], self.model_points, self.images)

            self.model.fit(model_points)

            # some points may not be able to generate models, we need to filter those out
            # keys = self.model.get_model_points.keys()
            # mapping = np.zeros(max(keys) + 1, dtype=np.int)
            # mapping[keys] = range( len( keys) )
            # for key in keys:
            #     self.model.model_points[ mapping[key] ] = self.model.model_points.pop(key)
            #     self.model.fh_models[ mapping[key] ] = self.model.fh_models.pop(key)

            self.model_points = self.model.get_model_points()
            self.model_paras = self.model.get_models()
            self.unary_cost = self.model.compute_unary_cost()

            with open(init_save_file, 'w') as f:
                pickle.dump((self.model_points, self.model_paras, self.unary_cost), f, True )



        self.labels = np.zeros(P, dtype=np.int)

    def run(self):

        for i in range(self.params.iters_num):

            print 'iteration' + '' + str(i)

            assignment, labels, outliers, inliers = self.update()

            if( np.array_equal(self.assignment, assignment) ) and \
                    (np.array_equal(self.labels,labels)):
                break

            # plot result
            # if(self.show_iter):
            #     util.plot_nbor(self.data['W'], self.data['Z'], self.data['s'], self.images,
            #                    labels=labels, show_edge=0, show_overlap=1, show_broken=1,
            #                    assignment=assignment)


            self.assignment = assignment
            self.labels = labels
            self.outliers = outliers
            self.inliers = inliers

        return assignment, labels, outliers, inliers

    def update(self):
        """
        call graph cut and update labels
        """

        # compute unary cost from the models MxP
        M, P = self.unary_cost.shape

        gc_para = self.params.graph_cut_para

        pairwise_nbor_num = gc_para.neighbors_num
        overlap_nbor_num = gc_para.overlap_neighbor_num
        lambda_weight = gc_para.lambda_weight
        overlap_cost = gc_para.overlap_cost

        unary = np.reshape(self.unary_cost.flatten('F'), (P, M) )
        overlap_nbor = self.data['s'][0:overlap_nbor_num, :]
        pairwise_nbor = self.data['s'][0:pairwise_nbor_num, :]
        pairwise_cost = self.pairwise_connection
        label_costs = self.params.mdl * np.ones(P)
        interior_labels = self.labels
        ppthresh = self.pointwise_outlier
        pbthresh = self.pairwise_breaking

        # set up graph cut optimization
        # input: labels
        # output: assignment, new_labels

        # graph cut engine: alpha, multi or allgc
        if(gc_para.engine == 0):
            assignment, new_labels = graph_cut.expand(
                unary,
                pairwise_nbor,
                pairwise_cost,
                interior_labels,
                label_costs,
                ppthresh
            )
        elif(gc_para.engine == 1):
            assignment, new_labels = graph_cut.multi(
                unary,
                overlap_nbor,
                interior_labels,
                lambda_weight,
                label_costs,
                ppthresh,
                pbthresh,
                overlap_cost
            )
        elif(gc_para.engine == 2):
            assignment, new_labels = graph_cut.allgc(
                unary,
                overlap_nbor,
                pairwise_nbor,
                pairwise_cost,
                interior_labels,
                lambda_weight,
                label_costs,
                ppthresh,
                pbthresh,
                overlap_cost
            )

        # change assignment back to M * P matrix
        assignment = assignment.T
        assignment = assignment.astype(np.int)
        new_labels = new_labels.astype(np.int)

        # outliers will not appear in the assignment corresponding to its labels
        outliers = assignment[new_labels, range(P)] == 0
        inliers = np.logical_not( outliers )

        new_model_points = {}

        for m in range(M):
            mask = assignment[m,:] > 0
            pnts = [i for i in np.where(mask)[0]]
            if(len(pnts) > 0):
                new_model_points[m] = pnts

        ## self.model_points = {key: self.model_points[key] for key in new_model_points}

        fixed_models = []

        for m, pnts in new_model_points.iteritems():

            if( self.model_points[m] == pnts ):
                fixed_models.append(m)

        changed_model_points = \
            {key: new_model_points[key] for key in new_model_points if key not in fixed_models}

        changed_model_points, not_fittable_pnts = self.model.fittable_pnts( changed_model_points )

        ## refit the models, only for changed models
        self.model.fit(changed_model_points)

        fixed_models = fixed_models + not_fittable_pnts.keys()
        keys = fixed_models + changed_model_points.keys()

        self.model_points = new_model_points
        self.model_paras = {key: self.model_paras[key] for key in fixed_models}
        self.model_paras.update( {changed_model_points.keys()[key]: self.model.get_models()[key]
                                  for key in self.model.get_models().keys() } )

        mapping = np.zeros(max(keys) + 1, dtype=np.int)
        mapping[keys] = range( len( keys) )

        assignment = assignment[keys, :]
        labels = mapping[new_labels]

        self.unary_cost = np.vstack((self.unary_cost[fixed_models, :],
                                    self.model.compute_unary_cost()))

        ## update model_points and model_paras
        self.model_points = {mapping[key]: self.model_points[key] for key in keys}
        self.model_paras = {mapping[key]: self.model_paras[key] for key in keys}

        return assignment, labels, outliers, inliers