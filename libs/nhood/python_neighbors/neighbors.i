%module neighbors

%{
    #define SWIG_FILE_WITH_INIT
    #include "neighbors_python.h"
%}

%include "numpy.i"

%init %{
    import_array();
%}

%pythoncode %{

import numpy as np

def nhood_old(measurement_matrix, visibility_matrix,
              velocity_weight, neighbors_num,
              top_k, pframe_thresh,
              max_occ, occ_penalty,
              bottom_vector, top_vector,
              distances, neighbors):

    measurement_matrix = measurement_matrix.astype(np.double)
    visibility_matrix = visibility_matrix.astype(np.int32)

    bottom_vector = bottom_vector.astype(np.int32)
    top_vector = top_vector.astype(np.int32)

    distances = distances.astype(np.double)
    neighbors = neighbors.astype(np.int32)

    get_nhood_old(measurement_matrix, visibility_matrix,
                  velocity_weight, neighbors_num,
                  top_k, pframe_thresh, max_occ,
                  occ_penalty, bottom_vector, top_vector,
                  distances, neighbors)

    return distances, neighbors

def nhood(measurement_matrix, visibility_matrix,
          velocity_weight, neighbors_num,
          top_k, pframe_thresh,
          max_occ, occ_penalty,
          bottom_vector, top_vector,
          distances, neighbors,
          color_matrix, color_weight):

    measurement_matrix = measurement_matrix.astype(np.double)
    visibility_matrix = visibility_matrix.astype(np.int32)

    color_matrix = color_matrix.astype(np.double)

    bottom_vector = bottom_vector.astype(np.int32)
    top_vector = top_vector.astype(np.int32)

    distances = distances.astype(np.double)
    neighbors = neighbors.astype(np.int32)

    get_nhood(measurement_matrix, visibility_matrix,
              velocity_weight, neighbors_num,
              top_k, pframe_thresh, max_occ,
              occ_penalty, bottom_vector, top_vector,
              distances, neighbors,
              color_matrix, color_weight)

    return distances, neighbors

%}


%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {
    (double* measurement_matrix, int frames2F, int pointsNM),
    (double* color_matrix, int channels, int pointsNC),
    (double* distances, int distancesM, int distancesN)
}

%apply (int* INPLACE_ARRAY2, int DIM1, int DIM2) {
    (int* visibility_matrix, int frames, int pointsNV),
    (int* neighbors, int neighborsM, int neighborsN)
}

%apply (int* INPLACE_ARRAY1, int DIM1) {
    (int* bottom_vector, int pointsB),
    (int* top_vector, int pointsT)
}

%include "neighbors_python.h"
