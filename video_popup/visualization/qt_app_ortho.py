### qt application

from PyQt4 import QtGui, QtCore

import numpy as np

import vispy_util_ortho

class ShowWidget(QtGui.QWidget):
    """
    widget for controlling showing cameras, segmentation, etc
    """

    signal_show_params = QtCore.pyqtSignal(dict, name='updateShow')

    def __init__(self, parent=None):
        super(ShowWidget, self).__init__(parent)

        self.gb = QtGui.QGroupBox(u"Visualization")
        self.gb.setCheckable(False)

        gb_lay = QtGui.QGridLayout()

        self.lC = []
        self.lC.append(QtGui.QCheckBox("Show Camera", self.gb))
        self.lC.append(QtGui.QCheckBox("Show Segmentation", self.gb))

        # Layout
        for pos in range(len(self.lC)):
            gb_lay.addWidget(self.lC[pos], pos, 0)
            self.lC[pos].stateChanged.connect(self.update_param)

        self.gb.setLayout(gb_lay)

        vbox = QtGui.QVBoxLayout()
        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.gb)
        hbox.addStretch(1.0)
        vbox.addLayout(hbox)
        vbox.addStretch(1.0)

        self.setLayout(vbox)

        self.show_params = {}
        self.show_params['show_camera'] = False
        self.show_params['show_segmentation'] = False

    def update_param(self, option):

        """
        update param and emit a signal
        """
        self.show_params['show_camera'] = self.lC[0].isChecked()
        self.show_params['show_segmentation'] = self.lC[1].isChecked()

        self.signal_show_params.emit(self.show_params)

class TweakWidget(QtGui.QWidget):

    """
    Widget for manually changing object scales
    """
    signal_tweak_params = QtCore.pyqtSignal(dict, name='tweakChanged')

    def __init__(self, num_objects, num_frames, parent=None):
        super(TweakWidget, self).__init__(parent)

        self.num_objects = num_objects
        self.num_frames = num_frames

        self.gb = QtGui.QGroupBox(u"Tweaking Object")
        self.gb.setCheckable(False)

        # self.show_selected_label = QtGui.QLabel("Show Selected", self.gb)
        self.show_selected_cb = QtGui.QCheckBox("Show Selected", self.gb)
        self.show_selected_cb.stateChanged.connect(self.update_param)

        self.selected_object_label = QtGui.QLabel("Selected Object", self.gb)
        self.selected_object_sp = QtGui.QSpinBox(self.gb)
        self.selected_object_sp.setMinimum(-1)
        self.selected_object_sp.setMaximum( num_objects-1 )
        self.selected_object_sp.setValue(-1)
        self.selected_object_sp.valueChanged.connect(self.update_object_or_frame)

        """ depth per frame"""
        self.depth_label = QtGui.QLabel("Object Depth(per frame)", self.gb)
        self.depth_sp = QtGui.QDoubleSpinBox(self.gb)
        self.depth_sp.setDecimals(2)
        self.depth_sp.setSingleStep(1)
        self.depth_sp.setLocale(QtCore.QLocale(QtCore.QLocale.English))
        self.depth_sp.setMinimum(0)
        self.depth_sp.setMaximum(1000)
        self.depth_sp.setValue(0)
        self.depth_sp.valueChanged.connect(self.update_param)

        self.slider_label = QtGui.QLabel("Slider Depth(per frame)", self.gb)
        self.depth_slider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.depth_slider.setMinimum(0)
        self.depth_slider.setMaximum(1000)
        self.depth_slider.setValue(0)
        self.connect(self.depth_slider, QtCore.SIGNAL('valueChanged(int)'), self.update_param)

        """ flipping per object """
        self.flip_cb = QtGui.QCheckBox("Flip Object", self.gb)
        self.flip_cb.stateChanged.connect(self.update_param)

        """flipping per frame, per object """
        self.flip_frame_cb = QtGui.QCheckBox("Flip Object(per frame)", self.gb)
        self.flip_frame_cb.stateChanged.connect(self.update_param)

        """frame slider"""
        self.frame_slider_label = QtGui.QLabel("Frame", self.gb)
        self.frame_slider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.frame_slider.setMinimum(1)
        self.frame_slider.setMaximum(num_frames)
        self.frame_slider.setValue(1)
        self.connect(self.frame_slider, QtCore.SIGNAL('valueChanged(int)'), self.update_object_or_frame)

        gb_lay = QtGui.QGridLayout()

        # gb_lay.addWidget(self.show_selected_label, 0, 0)
        # gb_lay.addWidget(self.show_selected_cb, 0, 1)
        gb_lay.addWidget(self.show_selected_cb, 0, 0)

        gb_lay.addWidget(self.selected_object_label, 1, 0)
        gb_lay.addWidget(self.selected_object_sp, 1, 1)

        gb_lay.addWidget(self.depth_label, 2, 0)
        gb_lay.addWidget(self.depth_sp, 2, 1)

        gb_lay.addWidget(self.slider_label, 3, 0)
        gb_lay.addWidget(self.depth_slider, 3, 1)

        gb_lay.addWidget(self.flip_cb, 4, 0)
        gb_lay.addWidget(self.flip_frame_cb, 5, 0)

        gb_lay.addWidget(self.frame_slider_label, 6, 0)
        gb_lay.addWidget(self.frame_slider, 6, 1)

        self.gb.setLayout(gb_lay)
        vbox = QtGui.QVBoxLayout()
        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.gb)
        hbox.addStretch(1.0)
        vbox.addLayout(hbox)
        vbox.addStretch(1.0)

        self.setLayout(vbox)

        self.tweak_params = {}
        self.tweak_params['object_id'] = -1
        self.tweak_params['object_depths'] = np.zeros((num_objects, num_frames))
        self.tweak_params['show_sel_object'] = self.show_selected_cb.isChecked()
        self.tweak_params['current_frame'] = 0
        self.tweak_params['object_flippings'] = np.zeros(num_objects, dtype=bool)
        self.tweak_params['object_flippings_pframe'] = np.zeros((num_objects, num_frames), dtype=bool)

        self.current_object = 0
        self.current_frame = 0
        self.object_or_frame_change = False

    def update_object_or_frame(self, option):

        self.current_object = self.selected_object_sp.value()
        self.current_frame = self.frame_slider.value() - 1

        self.object_or_frame_change = True

        flip = self.tweak_params['object_flippings'][self.current_object]
        depth = self.tweak_params['object_depths'][self.current_object, self.current_frame]
        flip_frame = self.tweak_params['object_flippings_pframe'][self.current_object,self.current_frame]

        self.depth_slider.setValue( depth  )
        self.depth_sp.setValue( depth )
        self.flip_cb.setChecked( flip )
        self.flip_frame_cb.setChecked( flip_frame )

        self.object_or_frame_change = False

        self.tweak_params['object_id'] = self.current_object
        self.tweak_params['current_frame'] = self.current_frame

        self.signal_tweak_params.emit(self.tweak_params)

    def update_param(self, option):
        """
        update param and emit a signal
        """

        if(self.object_or_frame_change):
            pass
        else:
            self.tweak_params['show_sel_object'] = self.show_selected_cb.isChecked()

            obj = self.current_object; frame = self.current_frame
            if(type(option) is int):
                self.tweak_params['object_depths'][obj, frame] = self.depth_slider.value()
                self.depth_sp.setValue(self.depth_slider.value())
            else:
                self.tweak_params['object_depths'][obj, frame] = self.depth_sp.value()
                self.depth_slider.setValue(self.depth_sp.value())

            """ update flipping as well """
            self.tweak_params['object_flippings'][obj] = self.flip_cb.isChecked()
            self.tweak_params['object_flippings_pframe'][obj, frame] = self.flip_frame_cb.isChecked()

        self.signal_tweak_params.emit(self.tweak_params)

class MainWindow(QtGui.QMainWindow):

    def __init__(self, scene_reconstructions, data, K, show=False):
        QtGui.QMainWindow.__init__(self)

        self.num_objects = len(scene_reconstructions['points'])
        self.num_frames = scene_reconstructions['rotations'][0].shape[0]/3

        self.resize(700, 500)
        self.setWindowTitle("Piecewise Orthographic Rigid Reconstruction")

        self.show_widget = ShowWidget(self)
        self.show_widget.signal_show_params.connect(self.update_view)

        self.tweak_widget = TweakWidget( self.num_objects, self.num_frames, self)
        self.tweak_widget.signal_tweak_params.connect(self.update_view)

        self.splitter_v = QtGui.QSplitter(QtCore.Qt.Vertical)
        self.splitter_v.addWidget(self.show_widget)
        self.splitter_v.addWidget(self.tweak_widget)

        # Create MySceneCanvas
        self.canvas = vispy_util_ortho.MyOrthoSceneCanvas(scene_reconstructions, data, K, show=False)
        self.canvas.create_native()
        self.canvas.native.setParent(self)
        self.canvas.measure_fps(0.1, self.show_fps)

        # Central Widget
        splitter1 = QtGui.QSplitter(QtCore.Qt.Horizontal)
        splitter1.addWidget(self.splitter_v)
        splitter1.addWidget(self.canvas.native)

        self.setCentralWidget(splitter1)

        # FPS message in statusbar:
        self.status = self.statusBar()
        self.status.showMessage("...")

    def update_view(self, params):

        if('object_depths' in params.keys()):
            self.canvas.set_tweaking_params(params['object_id'],
                                            params['object_depths'],
                                            params['show_sel_object'],
                                            params['current_frame'],
                                            params['object_flippings'],
                                            params['object_flippings_pframe'])

        if('show_camera' in params.keys()):
            self.canvas.set_camera_visiblity(params['show_camera'])
            self.canvas.set_color(not params['show_segmentation'])

    def show_fps(self, fps):

        self.status.showMessage("FPS - %.2f " % fps)

