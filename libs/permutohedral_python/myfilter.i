%module myfilter

%{
    #define SWIG_FILE_WITH_INIT
    #include "myfilter.h"
%}

%include "numpy.i"

%init %{
    import_array();
%}

%exception _myfilter{
    $action
    if (PyErr_Occurred()) SWIG_fail;
}

%pythoncode %{
import _myfilter
import numpy as np
def filter(control_image, perturb_image,SpatialStdev=5, ColorStdev=0.125):
      """Wrapper/interface to Fast High-Dimensional Filtering Using the Permutohedral Lattice
         Andrew Adams   Jongmin Baek    Abe Davis
         http://graphics.stanford.edu/papers/permutohedral/
         Takes as input a width*height*3 control image, and an image to perturb.
       """
      control_image=control_image.astype(np.double)
      perturb_image=perturb_image.astype(np.double)
      assert(control_image.shape[:2]==perturb_image.shape[:2])
      assert(control_image.shape[2]==3)
      out=np.empty_like(perturb_image)
      _filter1(control_image,perturb_image,out,SpatialStdev,ColorStdev)
      return out
%}

%apply (double* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) {
       (double* control_image,  int cx, int cy, int cd),
       (double* perturb_image,  int px, int py, int pd),
       (double* out_image, int ox, int oy, int od)}

%include "myfilter.h"
