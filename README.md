#VideoPopup

This is the implementation of the following ECCV2014 paper:

    Video Pop-up: Monocular 3D Reconstruction of Dynamic Scenes
    Chris Russell*, Rui Yu*, Lourdes Agapito

For more information about this work, please visit the [project website](http://www0.cs.ucl.ac.uk/staff/R.Yu/video_popup/VideoPopup2.html).

This github repository is maintained by Rui Yu (R.Yu@cs.ucl.ac.uk).
Contact me if you have any questions.

#1. Building the System

VideoPopup has been tested in Ubuntu 14.04 only.

###1.1 Requirements

Vispy
We use [vispy](https://github.com/vispy/vispy) for 3D visualization

OpenSfM
See this [Dockerfile](https://github.com/paulinus/opensfm-docker-base/blob/master/Dockerfile) for the commands to install all dependencies for OpenSfM.
Our code base already contains a modified version of OpenSfM, so do not need to install OpenSfM yourself

###1.2
  To compile the system, do the following:

```
  ./build.sh
```

#2. Data

One example sequence is available at [google drive](https://drive.google.com/open?id=0B8-9V4y1N7pxdzdCOXAweWk3OU0).

#3. Examples

After building the code and downloading the data, you are ready to try the provided examples.

###3.1 Motion Segmentation
   See video_popup/motion_segmentation/video_popup_motseg.py

###3.2 Depth Reconstruction from two consecutive frames
   See video_popup/depth_reconstruction/depth_reconstruction_test.py

###3.3 Piecewise Orthographic Reconstruction
   See video_popup/reconstruction/reconstruction_bird.py

###3.4 Piecewise Perspective Reconstruction
   See video_popup/reconstruction/reconstruction_tm_kitti.py

#4. Visualization

###4.1 Motion Segmentation Results
   See video_popup/motion_segmentation/segmentation_check.py

###4.1 Piecewise Orthographic Reconstruction
   See video_popup/reconstruction/reconstruction_bird_check.py

###3.4 Piecewise Perspective Reconstruction
   See video_popup/reconstruction/reconstruction_tm_kitti_check.py

#5. GUI Usage

Keyboard interactions

```
  'q': save rendering image
  'c': switch between image color and segmentation color
  's': decrease point size
  'b': increase point size
  'Right': next frame
  'Left: previous frame
```

If you use vispy_viewer.py, you can also do

```
  ' ': reset to original viewpoint
  'p': show pointcloud
  'f': show mesh faces
```


------
