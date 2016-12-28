""" Visualization Test using vispy """

import numpy as np

import math
import collections
import matplotlib.image as mpimg
import matplotlib.cm as cmx
import matplotlib.colors as colors

import cv2

from vispy import io
from vispy.scene import visuals
from vispy.visuals import CompoundVisual
import vispy.scene

from ImageVisual3D import ImageVisual3D
from MyMarkerVisual import MyMarkerNode

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

        u0 = u0 + 0.0
        v0 = v0 + 0.0

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

            trans = trans.reshape((-1, 1))

            """
            either way works
            """

            # angle, axis = angle_axis(rot)
            # rot_mat = vispy.util.transforms.rotate(angle, axis)
            # rot_mat = rot_mat[0:3,0:3]

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
        axis = rot / angle

    return  angle, axis

""" plot cameras """
def attach_cameras_to_view(reconstruction, view, data_path, length = 1, font_size = 12):

    nW = reconstruction.cameras.values()[0].width
    nH = reconstruction.cameras.values()[0].height

    fx = 2000.0; fy = 2000.0; u0 = nW / 2; v0 = nH / 2

    ordered_shots = collections.OrderedDict(sorted(reconstruction.shots.items()))
    num_shots = len(ordered_shots)

    ## record all camera nodes
    camera_nodes = {}
    text_nodes = {}

    for shot in range(num_shots):

        rot = ordered_shots.values()[shot].pose.rotation
        trans = ordered_shots.values()[shot].pose.translation

        # img = mpimg.imread(data_path+'/images/'+ordered_shots.values()[shot].id)
        img = mpimg.imread(data_path[ordered_shots.values()[shot].id])

        if (img.dtype.type is np.uint8 or img.dtype.type is np.uint16):
            img = img.astype(np.float32) / 255.0

        camera_nodes[shot] = CameraNode(fx, fy, u0, v0, length=length, nW=nW, nH=nH,
                                        rot=rot, trans=trans, image=img, parent=view.scene)

        ### show text

        origin = ordered_shots.values()[shot].pose.get_origin()

        text_nodes[shot] = visuals.Text(ordered_shots.values()[shot].id, color='red', bold=True,
                                        parent=view.scene, pos=origin, font_size=font_size)

    return camera_nodes, text_nodes

    # rot0 = ordered_shots.values()[0].pose.rotation
    # angle0, axis0 = angle_axis(rot0)
    # trans0 = ordered_shots.values()[0].pose.translation
    #
    # rot1 = ordered_shots.values()[1].pose.rotation
    # angle1, axis1 = angle_axis(rot1)
    # trans1 = ordered_shots.values()[1].pose.translation
    #
    # img = mpimg.imread('/home/cvfish/Work/code/mapillary/OpenSfM/data/berlin/images/02.jpg')
    # if (img.dtype.type is np.uint8 or img.dtype.type is np.uint16):
    #     img = img.astype(np.float32) / 255.0
    #
    # # test_node0 = CameraNode(fx, fy, u0, v0, length = 1, nW=nW, nH=nH,
    # #                        rot=rot0, trans = trans0, parent=view.scene)
    # #
    # test_node1 = CameraNode(fx, fy, u0, v0, length=2, nW=nW, nH=nH,
    #                         rot=rot1, trans=trans1, image=img,
    #                         parent=view.scene)
    #
    # # node0 = CameraNode(fx, fy, u0, v0, length = 1, nW=nW, nH=nH, parent=view.scene)
    # # node0.transform = vispy.visuals.transforms.MatrixTransform()
    # # node0.transform.rotate(angle0, axis0)
    # # node0.transform.translate(trans0)
    #
    # node1 = CameraNode(fx, fy, u0, v0, length=1, nW=nW, nH=nH, image=img,
    #                    parent=view.scene)
    #
    # node1.transform = vispy.visuals.transforms.MatrixTransform()
    # node1.transform.translate(-trans1)
    # node1.transform.rotate(-angle1, axis1)

# class MyCanvas(object):
class MyCanvas(vispy.scene.SceneCanvas):

    def __init__(self, reconstruction, data_path, has_color = 1, show_camera = 1):

        self.data_path = data_path
        self.has_color = has_color
        self.show_camera = show_camera

        pos, color = get_points(reconstruction, has_color)

        # create scatter object and fill in the data
        self.scatter = visuals.Markers()
        self.scatter.set_data(pos, edge_color=None, face_color=color, size=5)

        # self.canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
        # self.view = self.canvas.central_widget.add_view()

        vispy.scene.SceneCanvas.__init__(self, keys='interactive', show=True)
        self.unfreeze()
        self.view = self.central_widget.add_view()

        self.view.add(self.scatter)

        # size for plotting cameras
        self.length = 1
        self.font_size = 12

        if (show_camera):
            self.camera_nodes, self.text_nodes = \
                attach_cameras_to_view(reconstruction, self.view, self.data_path,
                                       length=self.length, font_size=self.font_size)

        self.view.camera = 'arcball'  # or try 'turntable'

        # add a colored 3D axis for orientation
        self.axis = visuals.XYZAxis(parent = self.view.scene)

        self.visible = True

        # self.canvas.app.run()
        self.app.run()

    def update_data(self, reconstruction):

        pos, color = get_points(reconstruction, self.has_color)
        self.scatter.set_data(pos, edge_color=None, face_color=color, size=5)

        ## set previous camera nodes to none
        if(self.show_camera):

            prev_cam_nodes = self.camera_nodes
            prev_text_nodes = self.text_nodes
            for shot in range(len(prev_cam_nodes)):
                prev_cam_nodes[shot].parent = None
                prev_text_nodes[shot].parent = None

            self.camera_nodes, self.text_nodes = \
                attach_cameras_to_view(reconstruction, self.view, self.data_path,
                                       length=self.length, font_size=self.font_size)

    def update_camera_size(self, length):

        self.length = length
        if(self.show_camera):
            for shot in range(len(self.camera_nodes)):
                self.camera_nodes[shot].update_size(self.length)

    def update_text_size(self, font_size):

        self.font_size = font_size
        for shot in range(len(self.text_nodes)):
            self.text_nodes[shot].font_size = self.font_size

    def on_key_press(self, event):

        if event.text == 'f':

            self.update_camera_size(self.length * 2)
            self.update()

        if event.text == 's':

            self.update_camera_size(self.length * 0.5)
            self.update()

        if event.text == 'j':

            self.update_text_size(self.font_size * 2)
            self.update()

        if event.text == 'l':

            self.update_text_size(self.font_size * 0.5)
            self.update()

        if event.text == 't':

            if(self.visible):

                for shot in range(len(self.camera_nodes)):
                    self.camera_nodes[shot]._visible = False
                for shot in range(len(self.text_nodes)):
                    self.text_nodes[shot]._visible = False
                self.axis._visible = False

                self.visible = False
                self.update()

            else:

                for shot in range(len(self.camera_nodes)):
                    self.camera_nodes[shot]._visible = True
                for shot in range(len(self.text_nodes)):
                    self.text_nodes[shot]._visible = True
                self.axis._visible = True

                self.visible = True
                self.update()

class MySceneCanvas(vispy.scene.SceneCanvas):

    def __init__(self, scene_reconstructions, image_files, K, show=True):

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

        # cmap = np.array([[1.0000, 0.2857, 0, 1.0],
        #                  [1.0000, 0.5714, 0, 1.0],
        #                  [1.0000, 0.8571, 0, 1.0],
        #                  [0.8571, 1.0000, 0, 1.0],
        #                  [0.5714, 1.0000, 0, 1.0],
        #                  [0, 1.0000, 0.8571, 1.0],
        #                  [0, 0.8571, 1.0000, 1.0],
        #                  [0, 0.5714, 1.0000, 1.0],
        #                  [0, 0.2857, 1.0000, 1.0],
        #                  [0, 0, 1.0000, 1.0],
        #                  [0.2857, 0, 1.0000, 1.0],
        #                  [0.8571, 0, 1.0000, 1.0],
        #                  [0.8571, 0, 0.5000, 1.0],
        #                  [0.5571, 0, 0.5000, 1.0]])

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
            point_cloud['seg_color'] = cmap[object_index, 0:3].astype(np.float32)
            point_cloud['rots'] = scene_reconstructions['rotations'][label]
            point_cloud['trans'] = scene_reconstructions['translations'][label]

            self.point_clouds[label] = point_cloud

            object_index += 1

        # parameters
        self.ref_color = cmap[num_objects, 0:3].astype(np.float32)
        self.selected_color = cmap[num_objects + 1, 0:3].astype(np.float32)

        self.show_ref_color = False
        self.show_sel_color = False

        self.show_color = True
        self.show_gt = False
        # self.fix_camera = True
        # self.fix_object = -1

        self.fix_camera = True
        self.fix_object = -1
        self.sel_object = -1

        self.cur_frame = 0

        self.show_camera = False
        self.show_all_cameras = False

        self.camera_length = 1

        self.show_text = False
        self.text_size = 12

        self.point_size = 5.0

        # create canvas
        #vispy.scene.SceneCanvas.__init__(self, keys='interactive', show=True)
        vispy.scene.SceneCanvas.__init__(self, keys='interactive', show=True, bgcolor='w')

        self.unfreeze()
        self.view = self.central_widget.add_view()

        ## add nodes to view
        self.create_nodes()

        ## transform these nodes
        self.transform_nodes()

        self.view.camera = 'arcball'
        # self.view.camera = 'turntable'

        if (show):
            self.app.run()

    # def __init__(self, scene_reconstructions, data, K, show=True):
    #
    #     # read images
    #     self.images = {}
    #     image_files = collections.OrderedDict(sorted(data.image_files.items()))
    #     self.num_frames = len(image_files)
    #     frame = 0
    #     for key, value in image_files.items():
    #         self.images[frame] = mpimg.imread(value)
    #         frame += 1
    #
    #     self.K = K
    #     self.nH = self.images.itervalues().next().shape[0]
    #     self.nW = self.images.itervalues().next().shape[1]
    #
    #     # prepare point clouds
    #     self.point_clouds = {}
    #
    #     num_objects = len(scene_reconstructions)
    #     cmap = get_colors(num_objects + 2)
    #
    #     object_index = 0
    #
    #     for label in scene_reconstructions:
    #
    #         reconstruction = scene_reconstructions[label][0]
    #         points = reconstruction.points
    #         num_points = len(points)
    #         vertices = np.ones((num_points, 3)).astype(np.float32)
    #         colors = np.ones((num_points, 3)).astype(np.float32)
    #         vertex_ids = np.ones((num_points, 1))
    #         index = 0
    #         for id in points:
    #             vertices[index] = points[id].coordinates
    #             colors[index] = points[id].color
    #             vertex_ids[index] = id
    #             index += 1
    #
    #         point_cloud = {}
    #         point_cloud['scale'] = 1.0
    #         point_cloud['vertices'] = vertices
    #         point_cloud['colors'] = colors / 255
    #         point_cloud['vertex_ids'] = vertex_ids
    #         point_cloud['seg_color'] = cmap[object_index,0:3].astype(np.float32)
    #         point_cloud['shots'] = collections.OrderedDict(sorted(scene_reconstructions[label][0].shots.items()))
    #
    #         self.point_clouds[label] = point_cloud
    #
    #         object_index += 1
    #
    #     # initialize cameras
    #
    #
    #     # parameters
    #     self.ref_color = cmap[num_objects, 0:3].astype(np.float32)
    #     self.selected_color = cmap[num_objects+1, 0:3].astype(np.float32)
    #
    #     self.show_ref_color = False
    #     self.show_sel_color = False
    #
    #     self.show_color = True
    #     self.show_gt = False
    #     # self.fix_camera = True
    #     # self.fix_object = -1
    #
    #     self.fix_camera = True
    #     self.fix_object = -1
    #     self.sel_object = -1
    #
    #     self.cur_frame = 0
    #
    #     self.show_camera = False
    #     self.show_all_cameras = False
    #
    #     self.camera_length = 1
    #
    #     self.show_text = False
    #     self.text_size = 12
    #
    #     self.point_size = 5.0
    #
    #     # create canvas
    #     vispy.scene.SceneCanvas.__init__(self, keys='interactive', show=True)
    #
    #     self.unfreeze()
    #     self.view = self.central_widget.add_view()
    #
    #
    #     ## add nodes to view
    #     self.create_nodes()
    #
    #     ## transform these nodes
    #     self.transform_nodes()
    #
    #     self.view.camera = 'arcball'
    #     #self.view.camera = 'turntable'
    #
    #     if(show):
    #         self.app.run()

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
                          ref_color = self.ref_color,
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

    def update_scales(self):

        for label in self.point_clouds_nodes:
            node = self.point_clouds_nodes[label]
            # node.set_data( self.point_clouds[label]['vertices'] * self.point_clouds[label]['scale'],
            #                edge_color = None,
            #                face_color = self.point_clouds[label]['colors']
            #                if self.show_color else self.point_clouds[label]['seg_color'],
            #                size = self.point_size
            #                )
            node.update_scale(self.point_clouds[label]['scale'])

            print self.sel_object
            print self.show_sel_color

            if(label == self.sel_object):
                node.update_sel_color(self.show_sel_color)
            else:
                node.update_sel_color(False)

        self.transform_nodes()

    def update_color_or_size(self):
        for label in self.point_clouds_nodes:
            node = self.point_clouds_nodes[label]
            # node.set_data( self.point_clouds[label]['vertices'] * self.point_clouds[label]['scale'],
            #                edge_color = None,
            #                face_color = self.point_clouds[label]['colors']
            #                if self.show_color  else self.point_clouds[label]['seg_color'],
            #                size = self.point_size
            #                )

            node.update_color(not self.show_color)

            if(label == self.fix_object):
                node.update_ref_color(self.show_ref_color)
            else:
                node.update_ref_color(False)

            node.update_point_size(self.point_size)

            # node._data.['a_bg_color'] =
            # self.point_clouds[label]['colors'] if self.show_color else self.point_clouds[label]['seg_color']

    def update_camera(self):

        for frame in range(self.num_frames):
            node = self.shot_nodes[frame]
            node.update_size(self.camera_length)

            if(frame == self.cur_frame):
                node._visible = self.show_camera or self.show_all_cameras
            else:
                node._visible = self.show_all_cameras

    def transform_nodes(self):

        if(self.fix_camera):

            ### we transform each point cloud according to the pose
            for label in self.point_clouds_nodes:

                node = self.point_clouds_nodes[label]
                node.transform = vispy.visuals.transforms.MatrixTransform()

                rot = self.point_clouds[label]['rots'][self.cur_frame*3:self.cur_frame*3+3, :]
                trans = self.point_clouds[label]['trans'][self.cur_frame*3:self.cur_frame*3+3]
                trans = trans * self.point_clouds[label]['scale']
                angle, axis = angle_axis(cv2.Rodrigues(rot)[0].reshape(-1))

                node.transform.rotate(angle, axis)
                node.transform.translate(trans)

            ### and set the transformations of all cameras to identity
            for frame in range(self.num_frames):

                node = self.shot_nodes[frame]
                node.transform = vispy.visuals.transforms.MatrixTransform()

                if (frame != self.cur_frame):
                    node._visible = False
                    #print "frame{:d} not visible".format(frame)
                else:
                    node._visible = self.show_camera or self.show_all_cameras
                    #print "frame{:d} visible".format(frame)

        else:

            ### find the object which we treat as reference
            if(self.fix_object == -1):
                print "error, no object has been set as reference"
            else:

                ref_rot = self.point_clouds[self.fix_object]['rots'][self.cur_frame*3:self.cur_frame*3+3, :]
                ref_trans = self.point_clouds[self.fix_object]['trans'][self.cur_frame*3:self.cur_frame*3+3]
                ref_trans = ref_trans * self.point_clouds[self.fix_object]['scale']
                ref_angle, ref_axis = angle_axis(cv2.Rodrigues(ref_rot)[0].reshape(-1))

                for label in self.point_clouds_nodes:

                    node = self.point_clouds_nodes[label]
                    node.transform = vispy.visuals.transforms.MatrixTransform()

                    if(self.fix_object == label):
                        ### reference object, identity transform
                        continue
                    else:
                        ### other objects, transform to reference object's coordinate system
                        rot = self.point_clouds[label]['rots'][self.cur_frame * 3:self.cur_frame * 3 + 3, :]
                        trans = self.point_clouds[label]['trans'][self.cur_frame * 3:self.cur_frame * 3 + 3]
                        trans = trans * self.point_clouds[label]['scale']
                        angle, axis = angle_axis(cv2.Rodrigues(rot)[0].reshape(-1))

                        node.transform.rotate(angle, axis)
                        node.transform.translate(trans)
                        node.transform.translate(-ref_trans)
                        node.transform.rotate(-ref_angle, ref_axis)

                ### has to transform the camera nodes as well
                for frame in self.shot_nodes:

                    node = self.shot_nodes[frame]
                    node.transform = vispy.visuals.transforms.MatrixTransform()

                    ref_rot = self.point_clouds[self.fix_object]['rots'][frame*3:frame*3+3, :]
                    ref_trans = self.point_clouds[self.fix_object]['trans'][frame*3:frame*3+3]
                    ref_trans = ref_trans * self.point_clouds[self.fix_object]['scale']
                    ref_angle, ref_axis = angle_axis(cv2.Rodrigues(ref_rot)[0].reshape(-1))

                    node.transform.translate(-ref_trans)
                    node.transform.rotate(-ref_angle, ref_axis)

                    if(frame == self.cur_frame):
                        node._visible = self.show_camera or self.show_all_cameras
                    else:
                        node._visible = self.show_all_cameras

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

        # if event.text == 's':
        #     if(self.show_color):
        #         self.show_color = False
        #         self.update_color_or_size()

        if event.text == 'c':

            self.show_color = not self.show_color
            self.update_color_or_size()

    def set_camera_visiblity(self, current_vis, all_cameras_vis):

        self.show_camera = current_vis
        self.show_all_cameras = all_cameras_vis
        self.update_camera()

    def set_object_scales(self, object_id, scales, show_sel_object):

        if object_id == -1:
            self.sel_object = object_id
            self.update_scales()
            return

        self.sel_object = object_id
        self.point_clouds[object_id]['scale'] = scales[object_id]
        self.show_sel_color = show_sel_object

        self.update_scales()

    def set_reference_object(self, object_id, show_ref_color):

        if(object_id == -1):
            self.fix_camera = True
        else:
            self.fix_camera = False
            self.fix_object = object_id

        self.transform_nodes()

        self.show_ref_color = show_ref_color
        self.update_color_or_size()

    def update_color(self, show_color):

        self.show_color = show_color
        self.update_color_or_size()

def simple_viewer(reconstruction, data_path, has_color = 1, show_camera = 1):

    pos, color = get_points(reconstruction, has_color)

    #
    # Make a canvas and add simple view
    #
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()

    # create scatter object and fill in the data
    scatter = visuals.Markers()
    scatter.set_data(pos, edge_color=None, face_color=color, size=5)

    view.add(scatter)

    if(show_camera):
        attach_cameras_to_view(reconstruction, view, data_path)

    view.camera = 'arcball'  # or try 'turntable'

    # add a colored 3D axis for orientation
    visuals.XYZAxis(parent=view.scene)

    canvas.app.run()