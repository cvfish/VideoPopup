# OpenGL viewer based on VisPy

import matplotlib.cm as cmx

import numpy as np

from vispy import io
from vispy import app, gloo
from vispy.util.transforms import perspective, translate, rotate
from vispy.util.quaternion import Quaternion

vert = """
// Uniforms
// ------------------------------------
uniform   mat4 u_model;
uniform   mat4 u_view;
uniform   mat4 u_projection;
uniform   vec4 u_color;
uniform   bool u_plot_seg;
uniform   float u_point_size;
// Attributes
// ------------------------------------
attribute vec3 a_position;
attribute vec4 a_color;
attribute vec4 a_seg_color;
// Varying
// ------------------------------------
varying vec4 v_color;
void main()
{
    if(u_plot_seg)
    {
        v_color = a_seg_color * u_color;
    }
    else
        v_color = a_color * u_color;
    gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);
    gl_PointSize = u_point_size;
}
"""


frag = """
// Varying
// ------------------------------------
varying vec4 v_color;
void main()
{
    gl_FragColor = v_color;
}
"""

def ortho_projection(nH, nW, near=0.1, far=100000):

    P = np.zeros((4,4)).astype(np.float32)

    P[0,0] = 2.0 / nW
    P[1,1] = 2.0 / nH

    P[0,3] = -1
    P[1,3] = 1

    P[2,2] = 2.0 / (near - far)
    P[2,3] = (near + far) / (near - far)

    P[3,3] = 1

    return  P

def projection_from_intrinsics(K, nH, nW, near=0.1, far=100000):

    P = np.zeros((4,4)).astype(np.float32)

    au = K[0,0]
    u0 = K[0,2]
    av = K[1,1]
    v0 = K[1,2]

    factor = 1.0

    u0 = nW / 2.0 + factor * (u0 - nW / 2.0)
    v0 = nH / 2.0 + factor * (v0 - nH / 2.0)

    P[0,0] = 2 * factor * au / nW
    P[1,1] = 2 * factor * av / nH

    P[0,2] = (1.0 - (2 * u0 / nW))
    P[1,2] = -(1.0 - (2 * v0 / nH))

    P[2,2] = (near + far) / (near - far)
    P[2,3] = (2 * near * far) / (near - far)

    P[3,2] = -1.0

    return P

def _arcball(x, y, w, h):
    """Convert x,y coordinates to w,x,y,z Quaternion parameters
    Adapted from:
    linalg library
    Copyright (c) 2010-2015, Renaud Blanch <rndblnch at gmail dot com>
    Licence at your convenience:
    GPLv3 or higher <http://www.gnu.org/licenses/gpl.html>
    BSD new <http://opensource.org/licenses/BSD-3-Clause>
    """
    r = (w + h) / 2.
    x, y = -(2. * x - w) / r, -(2. * y - h) / r
    h = np.sqrt(x*x + y*y)
    return (0., x/h, y/h, 0.) if h > 1. else (0., x, y, np.sqrt(1. - h*h))

"""
all the transformations, including rotate, translate, projection
generates transposed matrices, so when do matrix multiplication,
we need to transpose matrix first, and also transpose afterwards.
"""

class Canvas(app.Canvas):

    def __init__(self, vertices, colors, K, nH, nW, faces = 0, image_grid = 0, labels = 0,
                 nframes = 1, edge_thresh = 1000):

        app.Canvas.__init__(self, keys='interactive', size=(nW, nH))

        numPnts = vertices.shape[0] / nframes

        self.frame = 0
        self.nframes = nframes
        self.npoints = numPnts

        self.has_labels = 0
        self.per_frame_label = 0
        self.faces_buf = {}

        if( isinstance(labels, np.ndarray) ):

            self.per_frame_label = 1

            self.has_labels = 1

            seg_colors = np.zeros([np.prod(labels.shape), 4]).astype(np.float32)
            cmap = cmx.rainbow(np.linspace(0, 1, np.maximum( np.max(labels) + 1,
                                                            len(np.unique(labels) ) ) ) )

            # cmap = np.array([[1.0000,    0.2857,         0,  1.0],
            #                 [1.0000,    0.5714,         0,  1.0],
            #                 [1.0000,    0.8571,         0,  1.0],
            #                 [0.8571,    1.0000,         0,  1.0],
            #                 [0.5714,    1.0000,         0,  1.0],
            #                 [     0,    1.0000,    0.8571,  1.0],
            #                 [     0,    0.8571,    1.0000,  1.0],
            #                 [     0,    0.5714,    1.0000,  1.0],
            #                 [     0,    0.2857,    1.0000,  1.0],
            #                 [     0,         0,    1.0000,  1.0],
            #                 [0.2857,         0,    1.0000,  1.0],
            #                 [0.8571,         0,    1.0000,  1.0]])

            for i in np.unique(labels):
                seg_colors[:,3] = 1.0
                if(i != -1):
                    mask = labels == i
                    seg_colors[mask.reshape(-1),:] = cmap[i,:]

            self.seg_colors = seg_colors

        self.has_faces = 0
        self.draw_points = 1

        if( isinstance(faces, dict) ):

            self.has_faces = 1
            for f in range(nframes):
                self.faces = faces[f]
                vertices_f = vertices[f*numPnts:(f+1)*numPnts,:]
                mask = self.remove_long_edges(self.faces, vertices_f, edge_thresh)
                faces_temp = self.faces[mask, :]
                self.faces_buf[f] = gloo.IndexBuffer(faces_temp)

        elif(image_grid):

            self.has_faces = 1

            ### we have two types of edges
            indices = np.array(range(nH*nW)).astype(np.uint32).reshape((-1,1))
            triangles_type1 = np.hstack((indices, indices+1, indices + nW))
            triangles_type2 = np.hstack((indices, indices+nW-1, indices + nW))
            mask1 = np.ones((nH,nW),dtype=bool)
            mask2 = np.ones((nH,nW),dtype=bool)
            mask1[:,-1] = False
            mask1[-1,:] = False
            mask2[:, 0] = False
            mask2[-1,:] = False

            self.faces = np.vstack((
                triangles_type1[mask1.reshape(-1), :],
                triangles_type2[mask2.reshape(-1), :]))

            ## all the frames use the same faces
            if(self.has_labels):
                if(not self.per_frame_label):
                    # remove connections between different labels
                    labels_temp = labels.reshape(-1)
                    mask01 = labels_temp[self.faces[:,0]] == labels_temp[self.faces[:,1]]
                    mask02 = labels_temp[self.faces[:,0]] == labels_temp[self.faces[:,2]]
                    mask12 = labels_temp[self.faces[:,1]] == labels_temp[self.faces[:,2]]
                    mask = np.logical_and(mask01, mask02)
                    mask = np.logical_and(mask, mask12)
                    self.faces = self.faces[mask,:]

            for f in range(nframes):
                vertices_f = vertices[f*numPnts:(f+1)*numPnts,:]
                mask = self.remove_long_edges(self.faces, vertices_f, edge_thresh)
                faces_temp = self.faces[mask, :]
                self.faces_buf[f] = gloo.IndexBuffer(faces_temp)

            # different frames have different faces
            if(self.has_labels):
                if(self.per_frame_label):
                    for f in range(nframes):

                        labels_temp = labels[f*numPnts:(f+1)*numPnts].reshape(-1)
                        mask01 = labels_temp[self.faces[:,0]] == labels_temp[self.faces[:,1]]
                        mask02 = labels_temp[self.faces[:,0]] == labels_temp[self.faces[:,2]]
                        mask12 = labels_temp[self.faces[:,1]] == labels_temp[self.faces[:,2]]
                        mask = np.logical_and(mask01, mask02)
                        mask = np.logical_and(mask, mask12)
                        faces_temp = self.faces[mask,:]

                        vertices_f = vertices[f*numPnts:(f+1)*numPnts,:]
                        mask = self.remove_long_edges(faces_temp, vertices_f, edge_thresh)
                        faces_temp = faces_temp[mask,:]

                        self.faces_buf[f] = gloo.IndexBuffer(faces_temp)

        self.vertices = vertices
        self.colors = np.hstack((colors,
                                 np.ones((colors.shape[0], 1)).astype(np.float32)))

        self.update_color = 0
        if(colors.shape[0] == numPnts * nframes):
            self.update_color = 1

        # data contains the data to render
        self.data = np.zeros(numPnts, [('a_position', np.float32, 3),
                                       ('a_color',    np.float32, 4),
                                       ('a_seg_color',    np.float32, 4)])

        self.data['a_position'] = self.vertices[0:numPnts,:]
        self.data['a_color'] = self.colors[0:numPnts,:]

        if(self.has_labels):
            self.data['a_seg_color'] = self.seg_colors[0:numPnts,:]
        else:
            self.data['a_seg_color'] = self.colors[0:numPnts,:]

        self.center = np.mean(vertices, axis=0)

        self.program = gloo.Program(vert, frag)

        self.program.bind(gloo.VertexBuffer(self.data))
        # self.program['a_color'] = self.data['a_color']
        # self.program['a_position'] = self.data['a_position']
        # self.program['a_seg_color'] = self.data['a_seg_color']

        self.view = translate((0, 0, 0))
        self.model = np.eye(4, dtype=np.float32)

        gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
        # gloo.set_viewport(0, 0, nW, nH)

        if(isinstance(K, np.ndarray)):
            projection = projection_from_intrinsics(K, nH, nW)
        else:
            projection = ortho_projection(nH, nW)

        #print projection
        self.projection = projection.T
        # self.projection = np.ascontiguousarray(projection)
        # self.projection = np.ascontiguousarray(projection.T)

        # self.projection = perspective(45.0, self.size[0] /
        #                               float(self.size[1]), 2.0, 1000.0)


        self.program['u_projection'] = self.projection

        self.program['u_model'] = self.model
        self.program['u_view'] = self.view

        self.program['u_color'] = 1, 1, 1, 1

        self.program['u_plot_seg'] = 0

        self.program['u_point_size'] = 5

        self.nH = nH
        self.nW = nW

        self.last_x = 0
        self.last_y = 0

        self.substract_center = translate((-self.center[0],
                                           -self.center[1],
                                           -self.center[2]))

        self.add_center = translate((self.center[0],
                                     self.center[1],
                                     self.center[2]))

        self.translate_X = self.center[0]
        self.translate_Y = self.center[1]
        self.translate_Z = self.center[2]

        self.scale = - self.center[2] / 100

        self.rot_X = 0
        self.rot_Y = 0
        self.rot_Z = 0

        self.quaternion = Quaternion()

        self.theta = 0
        self.phi = 0

        #gloo.set_clear_color('black')
        gloo.set_clear_color('white')

        gloo.set_state('opaque')
        gloo.set_state(depth_test=True)
        # gloo.set_polygon_offset(1, 1)

        self.show()

    def remove_long_edges(self, faces, vertices, edge_thresh):

        pnts1 = vertices[faces[:,0],:]
        pnts2 = vertices[faces[:,1],:]
        pnts3 = vertices[faces[:,2],:]

        lengths12 = ((pnts1 - pnts2)**2).sum(axis = 1)
        lengths23 = ((pnts2 - pnts3)**2).sum(axis = 1)
        lengths13 = ((pnts1 - pnts3)**2).sum(axis = 1)

        mask = np.logical_and( lengths12 < edge_thresh,
                               np.logical_and( lengths23 < edge_thresh,
                                               lengths13 < edge_thresh) )

        return  mask

    def on_resize(self, event):

        gloo.set_viewport(0, 0, event.physical_size[0], event.physical_size[1])

        # print event.physical_size[0], event.physical_size[1]

    def update_camera(self):

        self.view = translate((self.translate_X,
                               self.translate_Y,
                               self.translate_Z))

        self.program['u_view'] = self.view

        self.update()

    def _normalize(self, x_y):

        x, y = x_y
        w, h = float(self.nW), float(self.nH)
        return x/(w/2.)-1., y/(h/2.)-1.

    def on_mouse_move(self, event):

        if event.is_dragging and not event.modifiers:

            # x0, y0 = event.press_event.pos
            # x1, y1 = event.last_event.pos
            # x, y = event.pos
            # x0_norm, y0_norm = self._normalize(event.press_event.pos)

            x1_norm, y1_norm = self._normalize(event.last_event.pos)
            x_norm,  y_norm = self._normalize(event.pos)
            dx, dy = x_norm - x1_norm, -(y_norm - y1_norm)

            button = event.press_event.button

            if button == 1:

                # w = self.nW
                # h = self.nH
                #
                # self.quaternion = (self.quaternion *
                #                Quaternion(*_arcball(x1, y1, w, h)) *
                #                Quaternion(*_arcball(x, y, w, h)))
                #
                # self.model = self.quaternion.get_matrix()

                self.rot_Y += 10*dx
                self.rot_X -= 10*dy

                rotx = rotate(self.rot_X,(1,0,0))
                roty = rotate(self.rot_Y,(0,1,0))
                rotz = rotate(self.rot_Z,(0,0,1))

                self.model = np.dot(self.substract_center,
                                    np.dot(rotz,
                                           np.dot(
                                               rotx,
                                               roty)))

                self.program['u_model'] = self.model

            elif button == 2:

                # self.translate_X += 10 * dx
                # self.translate_Y += 10 * dy

                self.translate_X += 10 * dx * self.scale
                self.translate_Y += 10 * dy * self.scale

            elif button == 3:

                self.rot_Z += 30*dy

                rotx = rotate(self.rot_X,(1,0,0))
                roty = rotate(self.rot_Y,(0,1,0))
                rotz = rotate(self.rot_Z,(0,0,1))

                self.model = np.dot(self.substract_center,
                                    np.dot(rotz,
                                           np.dot(
                                               rotx,
                                               roty)))

                self.program['u_model'] = self.model

            self.update_camera()

    def on_mouse_wheel(self, event):

        # self.translate_Z += event.delta[1]

        self.translate_Z += event.delta[1] * self.scale

        self.update_camera()

    def on_key_press(self, event):

        if event.text == 'q':

            img = self.render()
            io.write_png("../../render/render{:02d}.png".format(self.frame), img)

        if event.text == ' ':

            self.translate_X = self.center[0]
            self.translate_Y = self.center[1]
            self.translate_Z = self.center[2]

            self.rot_X = 0
            self.rot_Y = 0
            self.rot_Z = 0

            # self.quaternion = Quaternion()
            # self.model = np.eye(4, dtype=np.float32)

            self.model = self.substract_center

            self.program['u_model'] = self.model

            self.update_camera()

        if event.text == 'f':

            if(self.has_faces and self.draw_points):

                self.draw_points = 0
                self.update()

        if event.text == 'p':

            self.draw_points = 1
            self.update()

        if event.text == 'b':

            self.program['u_point_size'] *= 1.2
            self.update()

        if event.text == 's':

            self.program['u_point_size'] *= 1/1.2
            self.update()

        if event.key == 'Right':

            if(self.frame < self.nframes - 1):

                self.frame += 1

                print "frame{:d}".format(self.frame)

                self.data['a_position'] = self.vertices[self.frame * self.npoints:(self.frame+1) * self.npoints,:]
                # self.program['a_position'].set_data( self.data['a_position'] )

                if(self.update_color):

                    self.data['a_color'] =  self.colors[self.frame * self.npoints:(self.frame+1) * self.npoints,:]

                    if(self.has_labels):
                        self.data['a_seg_color'] =  self.seg_colors[self.frame * self.npoints:(self.frame+1) * self.npoints,:]
                    else:
                        self.data['a_seg_color'] = self.data['a_color']
                    # self.program['a_color'].set_data( self.data['a_color'] )

                self.program.bind(gloo.VertexBuffer(self.data))

                self.update()

        if event.key == 'Left':

            if(self.frame > 0):

                self.frame -= 1

                print "frame{:d}".format(self.frame)

                self.data['a_position'] = self.vertices[self.frame * self.npoints:(self.frame+1) * self.npoints,:]
                # self.program['a_position'].set_data( self.data['a_position'] )

                if(self.update_color):

                    self.data['a_color'] =  self.colors[self.frame * self.npoints:(self.frame+1) * self.npoints,:]

                    if(self.has_labels):
                        self.data['a_seg_color'] =  self.seg_colors[self.frame * self.npoints:(self.frame+1) * self.npoints,:]
                    else:
                        self.data['a_seg_color'] = self.data['a_color']
                    # self.program['a_color'].set_data( self.data['a_color'] )

                self.program.bind(gloo.VertexBuffer(self.data))

                self.update()

        if event.text == 'c':

            if(self.has_labels and self.program['u_plot_seg'] == 0):
                self.program['u_plot_seg'] = 1
                self.update()
            else:
                self.program['u_plot_seg'] = 0
                self.update()

    def on_draw(self, event):

        gloo.clear()

        if(self.draw_points):
            self.program.draw('points')
        else:
            if(self.has_faces):
                self.program.draw('triangles', self.faces_buf[self.frame])
            else:
                pass

def app_call(vertices, colors, K, nH, nW, image_grid = 0, labels = 0, nframes=1, edge_thresh = 1000):

        Canvas(np.ascontiguousarray(vertices).astype(np.float32),
               np.ascontiguousarray(colors).astype(np.float32),
               K, nH, nW, image_grid = image_grid, labels = labels,
               nframes = nframes, edge_thresh = edge_thresh)

        app.run()