%module test

%{
    #define SWIG_FILE_WITH_INIT
    #include "test.h"
%}

%include "numpy.i"

%init %{
    import_array();
    %}

%pythoncode %{

import numpy as np

def array_test(array1, array2):

  array1 = array1.astype(np.double)
  array2 = array2.astype(np.double)

  print_test(array1, array2);

  return 0

%}

%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {
    (double* array1, int array1m, int array1n),
    (double* array2, int array2m, int array2n)
}

%include "test.h"
