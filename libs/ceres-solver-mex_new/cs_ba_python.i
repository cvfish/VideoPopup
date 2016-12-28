%module cs_ba_python

%{
    #define SWIG_FILE_WITH_INIT
    #include "cs_ba_python.h"
%}

%include "numpy.i"

%init %{
    import_array();
    %}

%pythoncode %{

import numpy as np

def cs_ba_python(n, ncon, m, mcon, vmask, p0, cnp, pnp, x, errorType, sfmCase,
                 verbose, knum, kmask, prior_model, prior_para, mprior_mask,
                 cconst_mask):

  """same interface in python as the one used in matlab, refer to cs_ba.m for details"""

  vmask = np.reshape(vmask, (-1), order='F').astype(np.double)
  p0 = p0.astype(np.double)
  x = x.astype(np.double)
  kmask = kmask.astype(np.double)
  prior_para = prior_para.astype(np.double)
  mprior_mask = mprior_mask.astype(np.double)
  cconst_mask = cconst_mask.astype(np.double)

  pOut = np.copy(p0)

  _cs_ba(pOut, n, ncon, m, mcon, vmask,
         p0, cnp, pnp, x, errorType, sfmCase,
         verbose, knum, kmask, prior_model, prior_para, mprior_mask,
         cconst_mask)

  return pOut

%}


%apply (double* INPLACE_ARRAY1, int DIM1) {
    (double* pOut, int pOutn),
    (double* vmask, int vmaskn),
    (double* p0, int p0n),
    (double* x, int xn),
    (double* kmask, int kmaskn),
    (double* prior_para, int prior_paran),
    (double* mprior_mask, int mprior_maskn),
    (double* cconst_mask, int cconst_maskn)
}

%include "cs_ba_python.h"
