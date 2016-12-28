## inherit from Visual class to add support for scaling and switching colors


import numpy as np

from vispy.visuals import Visual
from vispy import gloo

from vispy.scene import visuals

vert = """
uniform  vec3 u_color;
uniform  vec3 u_seg_color;
uniform  vec3 u_ref_color;
uniform  vec3 u_sel_color;
uniform  float u_scale;
uniform  float u_point_size;
uniform  bool u_plot_seg;
uniform  bool u_plot_ref;
uniform  bool u_plot_sel;
// Attributes
// ------------------------------------
attribute vec3 a_position;
attribute vec3 a_color;

// Varying
// ------------------------------------
varying vec4 v_color;
void main()
{
    if(u_plot_seg)
    {
        v_color = vec4(u_seg_color * u_color, 1.0);
    }
    else
        v_color = vec4(a_color * u_color, 1.0);

    if(u_plot_ref)
        v_color = vec4(u_ref_color * u_color, 1.0);

    if(u_plot_sel)
        v_color = vec4(u_sel_color * u_color, 1.0);

    //a_position.x *= u_scale;
    //a_position.y *= u_scale;
    //a_position.z *= u_scale;

    gl_Position = $transform(vec4(u_scale*a_position.x, u_scale * a_position.y, u_scale * a_position.z, 1.0));
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

#gl_FragColor = v_color;

class MyMarkerVisual(Visual):

    def __init__(self):

        Visual.__init__(self, vcode=vert, fcode=frag)

    def set_data(self, vertices, color, seg_color, ref_color, selected_color, size = 5.0, scale = 1.0):

        # self.test = 1
        #
        # self.vertices = vertices
        # self.color = color
        # self.seg_color = seg_color.reshape((1,3))

        self.shared_program['a_position'] = gloo.VertexBuffer(vertices)
        self.shared_program['a_color'] = gloo.VertexBuffer(color)
        self.shared_program['u_seg_color'] = seg_color.reshape((1,3))
        self.shared_program['u_ref_color'] = ref_color.reshape((1, 3))
        self.shared_program['u_sel_color'] = selected_color.reshape((1, 3))
        self.shared_program['u_point_size'] = size
        self.shared_program['u_scale'] = scale

        self.shared_program['u_color'] = 1, 1, 1

        # plot color by default
        self.shared_program['u_plot_seg'] = 0
        self.shared_program['u_plot_ref'] = 0
        self.shared_program['u_plot_sel'] = 0

        self._draw_mode = 'points'

    def update_color(self, plot_seg = 0):

        self.shared_program['u_plot_seg'] = plot_seg
        self.update()

    def update_ref_color(self, plot_ref = 0):

        self.shared_program['u_plot_ref'] = plot_ref
        self.update()

    def update_sel_color(self, plot_sel = 0):

        self.shared_program['u_plot_sel'] = plot_sel
        self.update()

    def update_point_size(self, point_size):

        self.shared_program['u_point_size'] = point_size
        self.update()

    def update_scale(self, scale):

        print scale

        self.shared_program['u_scale'] = scale
        self.update()

    def _prepare_transforms(self, view):
        # xform = view.transforms.get_transform()
        # view.view_program.vert['transform'] = xform
        view.view_program.vert['transform'] = view.get_transform()


MyMarkerNode = visuals.create_visual_node(MyMarkerVisual)