### qt application

from PyQt4 import QtGui, QtCore

import numpy as np

import vispy_util_persp

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
        self.lC.append(QtGui.QCheckBox("Show All Cameras", self.gb))
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
        self.show_params['show_all_cameras'] = False
        self.show_params['show_segmentation'] = False

    def update_param(self, option):

        """
        update param and emit a signal
        """
        self.show_params['show_camera'] = self.lC[0].isChecked()
        self.show_params['show_all_cameras'] = self.lC[1].isChecked()
        self.show_params['show_segmentation'] = self.lC[2].isChecked()

        self.signal_show_params.emit(self.show_params)

class ReferenceWidget(QtGui.QWidget):

    signal_ref_params = QtCore.pyqtSignal(dict, name='updateRef')

    def __init__(self, num_objects, parent=None):
        super(ReferenceWidget, self).__init__(parent)

        self.num_objects = num_objects
        self.gb = QtGui.QGroupBox(u"Reference Object")
        self.gb.setCheckable(False)

        gb_lay = QtGui.QGridLayout()

        self.ref_object_label = QtGui.QLabel("Ref Object ID", self.gb)
        self.ref_object_sp = QtGui.QSpinBox(self.gb)
        self.ref_object_sp.setMinimum(-1)
        self.ref_object_sp.setMaximum( num_objects-1 )
        self.ref_object_sp.setValue(-1)
        self.ref_object_sp.valueChanged.connect(self.update_param)


        self.show_ref_cb = QtGui.QCheckBox("Show Ref",self.gb)
        self.show_ref_cb.stateChanged.connect(self.update_param)

        gb_lay.addWidget(self.show_ref_cb, 0, 0)
        gb_lay.addWidget(self.ref_object_label, 1, 0)
        gb_lay.addWidget(self.ref_object_sp, 1, 1)

        self.gb.setLayout(gb_lay)
        vbox = QtGui.QVBoxLayout()
        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.gb)
        hbox.addStretch(1.0)
        vbox.addLayout(hbox)
        vbox.addStretch(1.0)

        self.setLayout(vbox)

        self.ref_params = {}
        self.ref_params['ref_object'] = -1
        self.ref_params['show_ref_object'] = 0

    def update_param(self, option):
        """
        update param and emit a signal
        """
        self.ref_params['ref_object'] = self.ref_object_sp.value()
        self.ref_params['show_ref_object'] = self.show_ref_cb.isChecked()

        self.signal_ref_params.emit(self.ref_params)

class ScaleWidget(QtGui.QWidget):

    """
    Widget for manually changing object scales
    """
    signal_scale_parameters = QtCore.pyqtSignal(dict, name='scaleChanged')

    def __init__(self, num_objects, parent=None):
        super(ScaleWidget, self).__init__(parent)

        self.num_objects = num_objects
        self.gb = QtGui.QGroupBox(u"Scaling Object")
        self.gb.setCheckable(False)

        # self.show_selected_label = QtGui.QLabel("Show Selected", self.gb)
        self.show_selected_cb = QtGui.QCheckBox("Show Selected", self.gb)
        self.show_selected_cb.stateChanged.connect(self.update_param)

        self.selected_object_label = QtGui.QLabel("Selected Object", self.gb)
        self.selected_object_sp = QtGui.QSpinBox(self.gb)
        self.selected_object_sp.setMinimum(-1)
        self.selected_object_sp.setMaximum( num_objects-1 )
        self.selected_object_sp.setValue(-1)
        self.selected_object_sp.valueChanged.connect(self.update_object)

        self.scale_label = QtGui.QLabel("Object Scale", self.gb)
        self.scale_sp = QtGui.QDoubleSpinBox(self.gb)
        self.scale_sp.setDecimals(2)
        self.scale_sp.setSingleStep(1)
        self.scale_sp.setLocale(QtCore.QLocale(QtCore.QLocale.English))
        self.scale_sp.setMinimum(0.01)
        self.scale_sp.setMaximum(1000)
        self.scale_sp.setValue(1)
        self.scale_sp.valueChanged.connect(self.update_param)

        self.slider_label = QtGui.QLabel("Slider Scale", self.gb)
        self.scale_slider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.scale_slider.setMinimum(1)
        self.scale_slider.setMaximum(1000)
        self.scale_slider.setValue(1)
        self.connect(self.scale_slider, QtCore.SIGNAL('valueChanged(int)'), self.update_param)

        gb_lay = QtGui.QGridLayout()

        # gb_lay.addWidget(self.show_selected_label, 0, 0)
        # gb_lay.addWidget(self.show_selected_cb, 0, 1)
        gb_lay.addWidget(self.show_selected_cb, 0, 0)

        gb_lay.addWidget(self.selected_object_label, 1, 0)
        gb_lay.addWidget(self.selected_object_sp, 1, 1)

        gb_lay.addWidget(self.scale_label, 2, 0)
        gb_lay.addWidget(self.scale_sp, 2, 1)

        gb_lay.addWidget(self.slider_label, 3, 0)
        gb_lay.addWidget(self.scale_slider, 3, 1)

        self.gb.setLayout(gb_lay)
        vbox = QtGui.QVBoxLayout()
        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.gb)
        hbox.addStretch(1.0)
        vbox.addLayout(hbox)
        vbox.addStretch(1.0)

        self.setLayout(vbox)

        self.scale_params = {}
        self.scale_params['object_id'] = -1
        self.scale_params['object_scales'] = np.ones(num_objects)
        self.scale_params['show_sel_object'] = self.show_selected_cb.isChecked()
        # self.slider_scales = np.ones(num_objects)
        # self.spin_scales = np.ones(num_objects)

        self.current_object = 0
        self.object_change = False

    def update_object(self, option):

        self.current_object = self.selected_object_sp.value()

        self.object_change = True
        self.scale_slider.setValue( self.scale_params['object_scales'][self.current_object]  )
        self.scale_sp.setValue( self.scale_params['object_scales'][self.current_object] )
        self.object_change = False

        self.scale_params['object_id'] = self.current_object

        self.signal_scale_parameters.emit(self.scale_params)

    def update_param(self, option):
        """
        update param and emit a signal
        """

        self.scale_params['show_sel_object'] = self.show_selected_cb.isChecked()

        if(self.object_change):
            pass
        else:
            # self.slider_scales[self.current_object] = self.scale_slider.value()
            # self.spin_scales[self.current_object] = self.scale_sp.value()

            if(type(option) is int):
                self.scale_params['object_scales'][self.current_object] = self.scale_slider.value()
                self.scale_sp.setValue(self.scale_slider.value())
            else:
                self.scale_params['object_scales'][self.current_object] = self.scale_sp.value()
                self.scale_slider.setValue(self.scale_sp.value())

            # if(self.scale_slider.value() > self.scale_sp.value()):
            #     self.scale_params['object_scales'][self.current_object] = self.scale_slider.value()
            #     self.scale_sp.setValue(self.scale_slider.value())
            # elif(self.scale_slider.value() < self.scale_sp.value()):
            #     self.scale_params['object_scales'][self.current_object] = self.scale_sp.value()
            #     self.scale_slider.setValue(self.scale_sp.value())

        self.signal_scale_parameters.emit(self.scale_params)

class MainWindow(QtGui.QMainWindow):

    def __init__(self, scene_reconstructions, data, K, show=False):
        QtGui.QMainWindow.__init__(self)

        self.num_objects = len(scene_reconstructions)

        self.resize(700, 500)
        self.setWindowTitle("Piecewise Rigid Reconstruction")

        self.show_widget = ShowWidget(self)
        self.show_widget.signal_show_params.connect(self.update_view)

        self.ref_widget = ReferenceWidget( self.num_objects, self)
        self.ref_widget.signal_ref_params.connect(self.update_view)

        self.scale_widget = ScaleWidget(self.num_objects, self)
        self.scale_widget.signal_scale_parameters.connect(self.update_view)

        self.splitter_v = QtGui.QSplitter(QtCore.Qt.Vertical)
        self.splitter_v.addWidget(self.show_widget)
        self.splitter_v.addWidget(self.ref_widget)
        self.splitter_v.addWidget(self.scale_widget)

        # Create MySceneCanvas
        self.canvas = vispy_util_persp.MySceneCanvas(scene_reconstructions, data, K, show=False)
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

        if('object_scales' in params.keys()):
            self.canvas.set_object_scales(params['object_id'], params['object_scales'], params['show_sel_object'])

        if('ref_object' in params.keys()):
            self.canvas.set_reference_object(params['ref_object'], params['show_ref_object'])

        if('show_camera' in params.keys()):
            self.canvas.set_camera_visiblity(params['show_camera'], params['show_all_cameras'])
            self.canvas.update_color(not params['show_segmentation'])

    def show_fps(self, fps):

        self.status.showMessage("FPS - %.2f " % fps)










