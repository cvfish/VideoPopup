all:

	swig -python segmentation.i

	# g++ -g -shared -fPIC segmentation.cpp segmentation_wrap.c -w -o _segmentation.so \
	# -I/usr/local/lib/python2.7/dist-packages/numpy/core/include \
	# -I/usr/include/python2.7/

	g++ -O3 -shared -fPIC segmentation.cpp segmentation_wrap.c -w -o _segmentation.so \
	   -I/usr/local/lib/python2.7/dist-packages/numpy/core/include \
	   -I/usr/include/python2.7/
