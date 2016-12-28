""" Visualization Test using vispy """

import numpy as np

import math

import matplotlib.image as mpimg
import matplotlib.cm as cmx
import matplotlib.colors as colors

import cv2

import vispy.scene

from vispy import io
from vispy.scene import visuals
from vispy.visuals import CompoundVisual

from ImageVisual3D import ImageVisual3D
from MyMarkerVisual import MyMarkerNode

from video_popup.utils import util

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

class CameraVisual(CompoundVisual):
    """
    This visual is used for showing cameras
    """
    def __init__(self, fx, fy, u0, v0, length = 1, nW = 0, nH = 0, rot = 0, trans = 0, image = 0):

        self.fx = fx; self.fy = fy; self.u0 = u0; self.v0 = v0
        self.length = length; self.nW = nW; self.nH = nH
        self.rot = rot; self.trans = trans; self.image = image

        self.lines = vispy.visuals.LineVisual()
        self.lines2 = vispy.visuals.LineVisual()

        front_vertices = self.set_data(fx, fy, u0, v0, length, nW, nH, rot, trans)

        if (isinstance(image, np.ndarray)):
            self.image_visual = ImageVisual3D()
            self.image_visual.set_data(front_vertices, image)
            CompoundVisual.__init__(self, [self.lines, self.image_visual])
        else:
            CompoundVisual.__init__(self, [self.lines, self.lines2])

    def set_data(self, fx, fy, u0, v0, length = 1, nW = 0, nH = 0, rot = 0, trans = 0):

        center, ul, ur, ll, lr = self.get_camera(fx, fy, u0, v0, length, nW, nH, rot, trans)

        line_data = np.zeros((14,3))
        line_data[0] = center; line_data[1] = ul
        line_data[2] = center; line_data[3] = ur
        line_data[4] = center; line_data[5] = ll
        line_data[6] = center; line_data[7] = lr
        line_data[8] = ul; line_data[9] = ur
        line_data[10] = ll; line_data[11] = ul
        line_data[12] = ur; line_data[13] = lr

        self.lines.set_data(pos = line_data, color = (1, 0, 0, 1), width = 1, connect = 'segments')
        self.lines2.set_data(pos= line_data, color = (1, 0, 0, 1), width = 1, connect = 'segments')

        front_vertices = np.vstack((ul, ur, ll, lr)).astype(np.float32)

        return front_vertices

    """ computer camera center and four front points """
    def get_camera(self, fx, fy, u0, v0, length = 1, nW = 0, nH = 0, rot = 0, trans = 0):

        if(nW == 0):
            nW = 2*u0

        if(nH == 0):
            nH = 2*v0

        center = np.zeros((1,3))
        ray = np.zeros((1, 3))
        ray[0,2] = 1

        # ul = center + ray * length - u0 / fx * length - v0 / fy * length
        # ur = center + ray * length + (nW - u0) / fx * length - v0 / fy * length
        # ll = center + ray * length - u0 / fx * length + (nH - v0) / fy * length
        # lr = center + ray * length + (nW - u0) / fx * length + (nH - v0) / fy * length

        ul = center + ray * length
        ul[0,0] += - u0 / fx * length
        ul[0,1] += - v0 / fy * length

        ur = center + ray * length
        ur[0,0] += (nW - u0) / fx * length
        ur[0,1] += - v0 / fy * length

        ll = center + ray * length
        ll[0,0] += - u0 / fx * length
        ll[0,1] += (nH - v0) / fy * length

        lr = center + ray * length
        lr[0,0] += (nW - u0) / fx * length
        lr[0,1] += (nH - v0) / fy * length

        if(isinstance(rot, np.ndarray) and isinstance(trans, np.ndarray)):

            # angle, axis = angle_axis(rot)
            # rot_mat = vispy.util.transforms.rotate(angle, axis)

            trans = trans.reshape((-1, 1))
            rot_mat = cv2.Rodrigues(rot)[0]
            rot_mat = rot_mat.T

            center = np.dot( rot_mat, center.T - trans)
            ul = np.dot(rot_mat, ul.T - trans)
            ur = np.dot(rot_mat, ur.T - trans)
            ll = np.dot(rot_mat, ll.T - trans)
            lr = np.dot(rot_mat, lr.T - trans)

            center = center.T; ul = ul.T; ur = ur.T; ll = ll.T; lr = lr.T

        return  center, ul, ur, ll, lr

    def update_size(self, length):

        front_vertices = self.set_data(self.fx, self.fy, self.u0, self.v0,
                                       length,
                                       self.nW, self.nH, self.rot, self.trans)

        if (isinstance(self.image, np.ndarray)):
            #self.image_visual.set_data(front_vertices, self.image)
            self.image_visual.update_data(front_vertices)

CameraNode = visuals.create_visual_node(CameraVisual)

""" get points from reconstruction results """
def get_points(reconstruction, has_color = 1):

    num_points = len(reconstruction.points)
    position  = np.ones((num_points, 3)).astype(np.float32)
    color = np.ones((num_points, 4)).astype(np.float32)

    num = 0
    for key, value in reconstruction.points.iteritems():

        position[num, 0] = value.coordinates[0]
        position[num, 1] = value.coordinates[1]
        position[num, 2] = value.coordinates[2]

        if(has_color):
            color[num, 0] = value.color[0] / 255.0
            color[num, 1] = value.color[1] / 255.0
            color[num, 2] = value.color[2] / 255.0
        else:
            color[num, 1] = 0
            color[num, 2] = 0

        num += 1

    return  position, color

def angle_axis(rot):
    """ try to decompose axis and angle from axis-angle representation """
    if(np.linalg.norm(rot) == 0):
        angle = 0
        axis = np.ones_like(rot)
        print 'rotate zero angle'

    else:
        angle = np.linalg.norm(rot) * 180 / math.pi
        # axis = rot / angle
        axis = rot / np.linalg.norm(rot)

    return  angle, axis

class MyOrthoSceneCanvas(vispy.scene.SceneCanvas):

    def __init__(self, scene_reconstructions, image_files, K, show=True):

        # read images
        self.images = {}
        self.num_frames = len(image_files)
        frame = 0
        for image in image_files:
            self.images[frame] = mpimg.imread(image)
            frame += 1

        self.K = K
        self.nH = self.images.itervalues().next().shape[0]
        self.nW = self.images.itervalues().next().shape[1]

        # prepare point clouds
        self.point_clouds = {}

        num_objects = len(scene_reconstructions['rotations'])
        cmap = get_colors(num_objects + 2)

        object_index = 0

        for label in range(num_objects):

            vertices = scene_reconstructions['shapes'][label].T
            colors = scene_reconstructions['colors'][label].T
            points = scene_reconstructions['points'][label][0]

            point_cloud = {}
            point_cloud['scale'] = 1.0
            # point_cloud['vertices'] = np.ascontiguousarray(vertices)
            # point_cloud['colors'] = np.ascontiguousarray(colors)
            # point_cloud['vertices'] = vertices.astype(np.float32)
            # point_cloud['colors'] = colors.astype(np.float32)

            point_cloud['vertices'] = np.ascontiguousarray(vertices.astype(np.float32))
            point_cloud['colors'] = np.ascontiguousarray(colors.astype(np.float32))

            point_cloud['vertex_ids'] = points
            point_cloud['seg_color'] = cmap[object_index,0:3].astype(np.float32)
            point_cloud['rots'] = scene_reconstructions['rotations'][label]
            point_cloud['trans'] = scene_reconstructions['translations'][label]

            self.point_clouds[label] = point_cloud

            object_index += 1

            #util.plot_3d_point_cloud_vispy(vertices, colors)

        # initialize cameras

        # parameters
        self.selected_color = cmap[num_objects+1, 0:3].astype(np.float32)

        self.show_sel_color = False

        self.show_color = True
        self.show_gt = False

        self.fix_camera = True
        self.sel_object = -1

        self.cur_frame = 0
        self.show_camera = False
        self.camera_length = 1

        self.sel_object = 0
        self.depths = np.zeros((num_objects, self.num_frames))
        self.flippings = np.zeros(num_objects, dtype=np.bool)
        self.flippings_pframe = np.zeros((num_objects, self.num_frames), dtype=np.bool)

        self.show_text = False
        self.text_size = 12

        self.point_size = 5.0

        # create canvas
        vispy.scene.SceneCanvas.__init__(self, keys='interactive', show=True, bgcolor='w')

        self.unfreeze()
        self.view = self.central_widget.add_view()

        ## add nodes to view
        self.create_nodes()

        ## transform these nodes
        self.transform_nodes()

        self.view.camera = 'arcball'

        if(show):
            self.app.run()

    def create_nodes(self):

        self.point_clouds_nodes = {}
        for label in self.point_clouds:

            # node = visuals.Markers()
            # node.set_data( self.point_clouds[label]['vertices'],
            #                edge_color = None,
            #                face_color = self.point_clouds[label]['colors'],
            #                size = self.point_size
            #                )

            node = MyMarkerNode()
            node.set_data(self.point_clouds[label]['vertices'],
                          self.point_clouds[label]['colors'],
                          self.point_clouds[label]['seg_color'],
                          ref_color = self.selected_color,
                          selected_color = self.selected_color,
                          size=self.point_size,
                          scale=self.point_clouds[label]['scale'])

            self.view.add(node)
            self.point_clouds_nodes[label] = node

        self.shot_nodes = {}
        for frame in range(self.num_frames):
            self.shot_nodes[frame] = CameraNode(self.K[0,0],
                                                self.K[1,1],
                                                self.K[0,2],
                                                self.K[1,2],
                                                length = self.camera_length,
                                                nW = self.nW,
                                                nH = self.nH,
                                                image = self.images[frame],
                                                parent=self.view.scene)

    def update_params(self):

        for label in self.point_clouds_nodes:
            node = self.point_clouds_nodes[label]

            if (label == self.sel_object):
                node.update_sel_color(self.show_sel_color)
            else:
                node.update_sel_color(False)

        self.transform_nodes()

    def update_color_or_size(self):

        for label in self.point_clouds_nodes:
            node = self.point_clouds_nodes[label]

            node.update_color(not self.show_color)

            node.update_point_size(self.point_size)

    def update_camera(self):

        for frame in range(self.num_frames):
            node = self.shot_nodes[frame]
            node.update_size(self.camera_length)

            if(frame == self.cur_frame):
                node._visible = self.show_camera
            else:
                node._visible = False

    def transform_nodes(self):

        """
        important, make sure when call node.transform.translate(trans), the shape of trans should be (3,)
        """

        if(self.fix_camera):

            ### we transform each point cloud according to the pose
            for label in self.point_clouds_nodes:

                node = self.point_clouds_nodes[label]

                node.transform = vispy.visuals.transforms.MatrixTransform()

                rot = self.point_clouds[label]['rots'][self.cur_frame*3:self.cur_frame*3+3, :]
                trans = self.point_clouds[label]['trans'][self.cur_frame*3:self.cur_frame*3+3]

                angle, axis = angle_axis(cv2.Rodrigues(rot)[0].reshape(-1))
                node.transform.rotate(angle, axis)

                # """ flipping goes here """
                flag = self.flippings[label] != self.flippings_pframe[label, self.cur_frame]
                flipping_matrix = np.identity(4)
                if(flag):
                    flipping_matrix[2,2] = -1
                temp = node.transform.matrix

                temp = np.dot(temp, flipping_matrix)
                node.transform.matrix = temp

                """ depth translation goes here """
                node.transform.translate(trans)
                # node.transform.translate(np.array([0.0, 0.0, self.depths[label, self.cur_frame]]))

            ### and set the transformations of all cameras to identity
            for frame in range(self.num_frames):

                node = self.shot_nodes[frame]
                node.transform = vispy.visuals.transforms.MatrixTransform()

                if (frame != self.cur_frame):
                    node._visible = False
                else:
                    node._visible = self.show_camera

    def on_key_press(self, event):

        print event.text

        if event.text == 'q':
            img = self.render()
            io.write_png("../../render/render{:02d}.png".format(self.cur_frame), img)

        if event.text == 'b':
            self.point_size = self.point_size * 1.2
            self.update_color_or_size()

        if event.text == 's':
            self.point_size = self.point_size / 1.2
            self.update_color_or_size()

        if event.text == 'j':
            self.camera_length = self.camera_length * 2
            self.update_camera()

        if event.text == 'l':
            self.camera_length = self.camera_length * 0.5
            self.update_camera()

        if event.text == 't':
            self.show_camera = not self.show_camera
            self.update_camera()

        if event.key == 'Right':

            if(self.cur_frame < self.num_frames - 1):

                self.cur_frame += 1
                print "frame{:d}".format(self.cur_frame)

                self.transform_nodes()

                print event.text + ' inside'

        if event.key == 'Left':

            if (self.cur_frame > 0):

                self.cur_frame -= 1
                print "frame{:d}".format(self.cur_frame)

                self.transform_nodes()

                print event.text + ' inside'

        if event.text == 'c':
            self.show_color = not self.show_color
            self.update_color_or_size()


    def set_camera_visiblity(self, current_vis):

        self.show_camera = current_vis
        self.update_camera()

    def set_tweaking_params(self, object_id, depths, show_sel_object,
                            current_frame, flippings, flippings_pframe):

        self.sel_object = object_id
        self.depths = depths
        self.show_sel_color = show_sel_object

        self.cur_frame = current_frame
        self.flippings = flippings
        self.flippings_pframe = flippings_pframe

        self.update_params()

    def set_color(self, show_color):

        self.show_color = show_color
        self.update_color_or_size()