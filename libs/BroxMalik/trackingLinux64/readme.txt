This program is provided for research purposes only. Any commercial
use is prohibited. If you are interested in a commercial use, please 
contact the copyright holder. 

________________________________________________________________

Point tracking binary for 64 bit Linux (CPU version)
________________________________________________________________

(c) Thomas Brox 2011

This program is distributed WITHOUT ANY WARRANTY. 

If you use this program, you should cite the following paper:

N. Sundaram, T. Brox, K. Keutzer. Dense point trajectories by 
GPU-accelerated large displacement optical flow, European 
Conference on Computer Vision (ECCV), 2010.

---------------

Build:

g++ tracking.cpp -I. -L. -lldof -o tracking 

Usage:

./tracking bmfFile startFrame numberOfFrames sampling

bmfFile is a text file with a very short header, comprising the 
number of images in the sequence and 1. After the header all 
image files of the sequences are listed separated by line breaks. 
See cars1.bmf for an example. All input files must be in the 
PPM format (P6). 

startFrame is the frame where the computation is started. 
Usually this is frame 0. If you want to start later in the 
sequence you may specify another value.

numberOfFrames is the number of frames for which you want to 
run the tracker. Make sure that the value is not larger than 
the total number of frames of your sequence.

sampling specifies the subsampling parameter. If you specify 8, 
only every 8th pixel in x and y direction is taken into account. 
If you specify 1, the sampling will be dense. Since the tracker 
is based on dense optical flow, the choice of this parameter 
has little effect on the computation speed, but the size of the
trajectory file can become huge in case of dense trajectories. 

The output can be found in a subdirectory of the directory where 
the input sequence is stored. It comprises a text file 
TracksNumberOfFrames.dat with all the trajectories. For further 
details how to interpret the text file have a look at
readWriteTracks.cpp. Additionally, the directory contains a 
visualization of the tracked points Tracking???.ppm. 

__________________________________________________________________

Bugs
__________________________________________________________________

Please report bugs to brox@informatik.uni-freiburg.de

