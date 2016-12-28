import ortho_reconstruction as ortho

import numpy as np

from video_popup.utils import util

class Reconstruction(object):

    def __init__(self, data, para, options):

        self.data = data
        self.para = para

        self.save_path = options['save_path']
        self.try_list = options['try_list']

    def run(self):

        labels_parts = self.data['labels_parts']
        labels_objects = self.data['labels_objects']

        # labels_objects_unique = np.unique(labels_objects)
        # for object_id in labels_objects_unique:
        for object_id in self.try_list:

            # we will not try to reconstruct parts with less than 50 pnts
            object_pnts = np.where(labels_objects == object_id)[0]
            object_parts = labels_parts[labels_objects == object_id]
            object_parts_unique = np.unique(object_parts)


            for part in object_parts_unique:

                part_pnts = np.where(object_parts == part)[0]
                part_num = len(part_pnts)

                if(part_num < self.para.thresh):
                    object_pnts = np.delete(object_pnts, part_pnts)
                    object_parts = np.delete(object_parts, part_pnts)

            object_parts_unique = np.unique(object_parts)

            W = self.data['W']
            Z = self.data['Z']
            s = self.data['s']
            colors = self.data['colors']
            assignment = self.data['assignment']
            images = self.data['images']

            # create the neighborhood structure for an object
            object_s = np.copy( s[:, object_pnts] )
            mapping = np.zeros(len(labels_objects) + 1)
            mapping[:] = -1
            mapping[ object_pnts ] = range( len(object_pnts) )
            object_s = mapping[ object_s ]

            object_W = W[:, object_pnts]
            object_Z = Z[:, object_pnts]
            object_colors = colors[:, object_pnts]

            object_assign = assignment[object_parts_unique][:, object_pnts]

            #### check each part
            # util.plot_traj(object_W, object_Z, images, frame_step=10)

            # for part in object_parts_unique:
            #     ind = object_parts == part
            #     util.plot_traj(object_W[:, ind], object_Z[:,ind], images, frame_step=10)

            self.object_reconstruction(object_id, object_W, object_Z, object_s,
                                  object_parts, object_assign, images, object_colors)

    def object_reconstruction(self, object_id, W, Z, s, labels, assign, images, colors):

        print object_id

        save_path = self.save_path + '/object_{:02d}'.format(object_id)

        if(self.para.method == self.para.ORTHO_PIECEWISE_STITCHING or
           self.para.method == self.para.ORTHO_PIECEWISE_GLOBAL or
           self.para.method == self.para.ORTHO_GLOBAL):

            engine = ortho.OrthoReconstruction(W, Z, s, labels, assign, images, colors,
                                               self.para.ortho_reconstruction_para,
                                               save_path = save_path)

        engine.initialize()

        engine.refine()
