all:

	swig -python neighbors.i

	# g++ -g -shared -fPIC neighbors.cpp neighbors_wrap.c -w -o _neighbors.so \
	# 	-I/usr/local/lib/python2.7/dist-packages/numpy/core/include \
	# 	-I/usr/include/python2.7/

	g++ -O3 -shared -fPIC -fopenmp neighbors.cpp neighbors_wrap.c -w -o _neighbors.so \
	   -I/usr/local/lib/python2.7/dist-packages/numpy/core/include \
	   -I/usr/include/python2.7/
