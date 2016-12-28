#VideoPopup

This is the implementation of the following ECCV2014 paper:

    Video Pop-up: Monocular 3D Reconstruction of Dynamic Scenes
    Chris Russell*, Rui Yu*, Lourdes Agapito
    *Joint first authorship

and the reconstruction part of:

    Dense Monocular Depth Estimation in Complex Dynamic Scenes
    Rene Ranftl, Vibhav Vineet, Qifeng Chen, Vladlen Koltun1

For more information about this work, please visit the [project website](http://www0.cs.ucl.ac.uk/staff/R.Yu/video_popup/VideoPopup2.html).

This github repository is maintained by Rui Yu (R.Yu@cs.ucl.ac.uk).
Contact me if you have any questions.

#1. Building the System

VideoPopup has been tested in Ubuntu 14.04 only.

###1.1 Requirements

Vispy: we use [vispy](https://github.com/vispy/vispy) for 3D visualization

OpenSfM: see this [Dockerfile](https://github.com/paulinus/opensfm-docker-base/blob/master/Dockerfile) for the commands to install all dependencies for OpenSfM.

Our code base already contains a modified version of OpenSfM, so do not need to install OpenSfM yourself.

###1.2 Build VideoPopup

  To compile the system, do the following:

```
  ./build.sh
```

#2. Data

One example sequence is available at [google drive](https://drive.google.com/open?id=0B8-9V4y1N7pxdzdCOXAweWk3OU0).

```
put data folder under root directory
if you want to create new tracks file, check data/Kitti/05_2f/runBroxMalik.sh
```

#3. Examples

After building the code and downloading the data, you are ready to try the provided examples.

###3.1 Motion Segmentation
<!-- ![bird-segmentation-result](https://github.com/cvfish/VideoPopup/blob/master/render/bird/image.jpg) -->
<!-- ![two-men-segmentation-result](https://github.com/cvfish/VideoPopup/blob/master/render/two-men/image.png) -->
![segmentation-result](https://github.com/cvfish/VideoPopup/blob/master/render/image.png)
```
video_popup/motion_segmentation/video_popup_motseg.py
video_popup/motion_segmentation/segmentation_check.py
```

###3.2 Depth Reconstruction from two consecutive frames
![depth-result](https://github.com/cvfish/VideoPopup/blob/master/render/depth.png)
```
video_popup/depth_reconstruction/depth_reconstruction_test.py
```

###3.3 Piecewise Orthographic Reconstruction
<!-- ![bird-sparse-result](https://github.com/cvfish/VideoPopup/blob/master/render/bird/sparse_render00.png) -->
<!-- ![bird-dense-result](https://github.com/cvfish/VideoPopup/blob/master/render/bird/dense_render00.png) -->
![bird-result](https://github.com/cvfish/VideoPopup/blob/master/render/bird.png)
```
video_popup/reconstruction/reconstruction_bird.py
video_popup/reconstruction/reconstruction_bird_check.py
```

###3.4 Piecewise Perspective Reconstruction
<!-- ![two-men-sparse-result](https://github.com/cvfish/VideoPopup/blob/master/render/two-men/sparse_render00.png) -->
<!-- ![two-men-dense-result](https://github.com/cvfish/VideoPopup/blob/master/render/two-men/dense_render00.png) -->
![bird-result](https://github.com/cvfish/VideoPopup/blob/master/render/two-men.png)
```
video_popup/reconstruction/reconstruction_tm_kitti.py
video_popup/reconstruction/reconstruction_tm_kitti_check.py
```

#4. GUI Usage

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
