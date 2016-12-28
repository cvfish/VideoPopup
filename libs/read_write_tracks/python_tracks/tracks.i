%module tracks

%{
  #define SWIG_FILE_WITH_INIT
  #include "tracks_io.h"
%}

%include "numpy.i"

%init %{
  import_array();
%}

%pythoncode %{

import numpy as np
def read_tracks(file_name, measurements, labels):
  with open(file_name) as f:
    frames = int(next(f))
    tracks = int(next(f))
  measurements = measurements.astype(np.double)
  labels = labels.astype(np.int32)
  readTracks(file_name, measurements, labels)
  return measurements, labels

def write_tracks(file_name, measurements, labels):
  frames, tracks = measurements.shape
  measurements = measurements.astype(np.double)
  labels = labels.astype(np.int32)
  writeTracks(file_name, measurements, labels)

%}

%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {
  (double* measurements, int frames, int tracks)
 }

%apply (int* INPLACE_ARRAY1, int DIM1) {
  (int *labels, int labels_num)
    }

%include "tracks_io.h"
