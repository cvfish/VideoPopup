import cv2
import numpy as np

class PerspModelFitting(object):

    def __init__(self, data, persp_fitting_para):

        self.data = data
        self.para = persp_fitting_para
        self.model_points = {}
        self.fh_models = {}
        self.inlier_mask = {}

    def fittable_pnts(self, model_points):

        Z = self.data['Z']

        not_fittable = []

        for m, pnts in model_points.iteritems():

            if (len(pnts) <= 10):

                not_fittable.append(m)

            else:

                pair_vis = np.sum(Z[0:-1, pnts] * Z[1:, pnts], axis=1) >= 10

                if(not any(pair_vis)):

                    not_fittable.append(m)

        return {k:model_points[k] for k in model_points if not k in not_fittable}, \
               {k:model_points[k] for k in not_fittable}

    def fit(self, model_points):

        self.model_points = {}
        self.fh_models = {}
        self.inlier_mask = {}

        Z = self.data['Z']
        k = 0

        for m, pnts in model_points.iteritems():

            if (len(pnts) > 10):

                pair_vis = np.sum(Z[0:-1, pnts] * Z[1:, pnts], axis=1) >= 10

                self.model_points[k] = pnts
                self.fh_models[k], self.inlier_mask[k] = self.model_fitting(pnts, pair_vis)
                k = k + 1

    def model_fitting(self, pnts, pair_vis):

        W = self.data['W']
        Z = self.data['Z']

        model = {}
        inlier_mask = {}

        for frame, vis in enumerate(pair_vis):

            if (vis):

                mask = np.where(np.sum(Z[frame:frame + 2, pnts], axis=0) == 2)[0]
                Wp = W[2 * frame:2 * frame + 4, [ pnts[i] for i in mask ]]
                model[frame], inlier_mask[frame] = self.model_fitting_frame(Wp[0:2, :], Wp[2:4, :])

        # fill up missing frames
        not_vis = np.logical_not(pair_vis)
        diff = np.reshape(np.where(not_vis)[0], (-1,1)) - np.reshape(np.where(pair_vis)[0],(1,-1))
        closest_vis_for_nvis = np.argmin(np.abs(diff), axis=1)

        vis_frames = np.where(pair_vis)[0]
        for ind, frame in enumerate(np.where(not_vis)[0]):
            model[frame] = model[ vis_frames[ closest_vis_for_nvis[ind] ] ]
            inlier_mask[frame] = inlier_mask[ vis_frames[ closest_vis_for_nvis[ind] ] ]

        return model, inlier_mask

    def model_fitting_frame(self, p1, p2):

        # [p_2; 1]^T F [p_1; 1] = 0
        # F, maskF = cv2.findFundamentalMat(np.transpose(p1.astype(np.float32)),
        #                                   np.transpose(p2.astype(np.float32)),
        #                                   cv2.FM_RANSAC)
        F, maskF = cv2.findFundamentalMat(np.transpose(p1.astype(np.float32)),
                                          np.transpose(p2.astype(np.float32)),
                                          cv2.FM_8POINT)

        # print "Fundamental matrix: \n"
        # print F
        # print "Mask: \n"
        # print maskF

        # H, maskH = cv2.findHomography(p1,p2)
        # sigma = self.para['sigma']
        # lambda1 = self.para['lambda1']
        # lambda2 = self.para['lambda2']

        return F, maskF

    def compute_unary_cost(self):

        # number of points
        P = self.data['W'].shape[1]
        # number of models
        M = len(self.fh_models)
        unary_cost = np.zeros((M, P))

        W = self.data['W']
        Z = self.data['Z']
        fh_models = self.fh_models

        for m, fh_model in fh_models.iteritems():

            for frame, fh in fh_model.iteritems():

                mask = np.logical_and(Z[frame, :], Z[frame + 1, :])

                if any(mask):
                    unary_cost[m, np.where(mask)[0]] += self.get_fitting_error(W[2 * frame:2 * frame + 4, np.where(mask)[0]], fh)

        return unary_cost

    def get_fitting_error(self, Wf, fh):

        P = Wf.shape[1]

        # compute sampson error for fundamental matrix fitting
        # x1 = np.concatenate((Wf[0:2, :], np.ones((1, P))), axis=0)
        # x2 = np.concatenate((Wf[2:4, :], np.ones((1, P))), axis=0)

        x1 = np.vstack(( Wf[0:2, :], np.ones((1, P)) ) )
        x2 = np.vstack(( Wf[2:4, :], np.ones((1, P)) ) )

        Fx1 = np.dot(fh, x1)
        Ftx2 = np.dot(fh.T, x2)


        x2tFx1 = np.sum(x2 * Fx1, axis=0)

        cost = x2tFx1 ** 2 / (Fx1[0, :] ** 2 + Fx1[1, :] ** 2 + Ftx2[0, :] ** 2 + Ftx2[1, :] ** 2)

        return cost

    def get_models(self):

        return self.fh_models

    def get_model_points(self):

        return self.model_points

    def get_inlier_mask(self):

        return  self.inlier_mask