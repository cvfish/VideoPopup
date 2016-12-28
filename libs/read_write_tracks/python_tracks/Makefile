all:
	swig -python tracks.i

	g++ -g -shared -fPIC tracks_io.cpp tracks_wrap.c -w -o _tracks.so \
	  -I/usr/local/lib/python2.7/dist-packages/numpy/core/include \
	  -I/usr/include/python2.7/
