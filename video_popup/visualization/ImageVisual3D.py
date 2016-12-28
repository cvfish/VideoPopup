import numpy as np

from vispy.visuals import Visual
from vispy import gloo

"""
 0-------1
 |       |
 |       |
 2-------3
"""

vert = """
// Attributes
attribute vec3 a_position;
attribute vec2 a_texcoord;

// Varyings
varying vec2 v_texcoord;

// Main
void main (void)
{
    v_texcoord = a_texcoord;
    gl_Position = $transform(vec4(a_position.x, a_position.y, a_position.z, 1.0));
}
"""

frag = """
uniform sampler2D u_texture;
varying vec2 v_texcoord;
void main()
{
    gl_FragColor = texture2D(u_texture, v_texcoord);
    gl_FragColor.a = 1.0;
}

"""

class ImageVisual3D(Visual):

    def __init__(self):

        Visual.__init__(self, vcode=vert, fcode=frag)
        # self.set_data(vertices, image)

    def set_data(self, vertices, image):

        self.faces = np.array([[0, 1, 2],
                               [1, 2, 3]]).astype(np.uint32)
        self._faces = gloo.IndexBuffer()
        self._faces.set_data(self.faces)
        self._index_buffer = self._faces

        self.textures = np.array([[0.0, 0.0], [1.0, 0.0],
                                  [0.0, 1.0], [1.0, 1.0]]).astype(np.float32)

        self.shared_program['a_position'] = gloo.VertexBuffer(vertices)
        self.shared_program['a_texcoord'] = gloo.VertexBuffer(self.textures)
        self.shared_program['u_texture'] = gloo.Texture2D(image)

        self._draw_mode = 'triangles'

    def update_data(self, vertices):

        self.shared_program['a_position'] = gloo.VertexBuffer(vertices)

    def _prepare_transforms(self, view):
        # xform = view.transforms.get_transform()
        # view.view_program.vert['transform'] = xform
        view.view_program.vert['transform'] = view.get_transform()

