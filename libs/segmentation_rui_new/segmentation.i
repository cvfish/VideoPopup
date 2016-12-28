%module segmentation

%{
    #define SWIG_FILE_WITH_INIT
    #include "segmentation.h"
  %}

%include "numpy.i"

%init %{
    import_array();
    %}

%pythoncode %{

import numpy as np

def allgc(unary, overlap_nbor, pairwise_nbor, pairwise_cost,
          interior_labels, lambda_weight, label_costs, ppthresh,
          pbthresh, overlap_cost):

    """ Newest graph-cut segmentation algorithm that supports edge-breaking and sparsity in overlapping models.
        unary: [points * models] unary cost, overlap_nbor: [overlap_nbor_num * points] neighborhood, pairwise_nbor:
        [pairwise_nbor_num * points], pairwise_cost: [pairwise_nbor_num * points ] pairwise cost, interior_labels:
        [1*points] vector current interior labeling, lambda: overlapping parameter, label_costs: [1*points] vector mdl label costs,
        ppthresh: [1*points] vector per point outlier threshold, pbthresh: [overlap_nbor_num * points / 1 * points] 2d array
        edge breaking threshold(per edge or per point, corresponding to different strategies),
        overlap_cost: cost per pairwise overlapping labels.
    """
    from numpy import zeros

    mask_out = zeros(unary.shape)
    labels_out = zeros(unary.shape[0])

    overlap_nbor = overlap_nbor.astype(np.double)
    pairwise_nbor = pairwise_nbor.astype(np.double)
    pairwise_cost = pairwise_cost.astype(np.double)
    interior_labels = interior_labels.astype(np.double)
    label_costs = label_costs.astype(np.double)
    ppthresh = ppthresh.astype(np.double)
    pbthresh = pbthresh.astype(np.double)

    _allgc(mask_out, labels_out, unary, overlap_nbor, pairwise_nbor,
           pairwise_cost, interior_labels, lambda_weight, label_costs, ppthresh,
           pbthresh, overlap_cost)

    return mask_out, labels_out

def expand(unary, pairwise_nbor, pairwise_cost, interior_labels, label_costs, ppthresh):

    from numpy import zeros
    mask_out = zeros(unary.shape)
    labels_out = zeros(unary.shape[0])

    pairwise_nbor = pairwise_nbor.astype(np.double)
    pairwise_cost = pairwise_cost.astype(np.double)
    interior_labels = interior_labels.astype(np.double)
    label_costs = label_costs.astype(np.double)
    ppthresh = ppthresh.astype(np.double)

    _expand(mask_out, labels_out, unary, pairwise_nbor, pairwise_cost,
            interior_labels, label_costs, ppthresh)

    return mask_out, labels_out

def multi(unary, overlap_nbor, interior_labels, lambda_weight, label_costs, ppthresh, pbthresh, overlap_cost):

    from numpy import zeros
    mask_out = zeros(unary.shape)
    labels_out = zeros(unary.shape[0])

    overlap_nbor = overlap_nbor.astype(np.double)
    interior_labels = interior_labels.astype(np.double)
    label_costs = label_costs.astype(np.double)
    ppthresh = ppthresh.astype(np.double)
    pbthresh = pbthresh.astype(np.double)

    _multi(mask_out, labels_out, unary, overlap_nbor, interior_labels,
           lambda_weight, label_costs, ppthresh, pbthresh, overlap_cost)

    return mask_out, labels_out

%}


%apply (double* INPLACE_ARRAY1, int DIM1) {
    (double* labels_out, int labels_outn),
    (double* interior_labels, int interior_labelsn),
    (double* label_costs, int label_costsn),
    (double* ppthresh, int ppthreshn)
}

%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {
    (double* mask_out, int maskm, int maskn),
    (double* pairwise_cost, int pairwise_costm, int pairwise_costn),
    (double* unary, int unarym, int unaryn),
    (double* pbthresh, int pbthreshm, int pbthreshn),
    (double* overlap_nbor, int overlap_nborm, int overlap_nborn),
    (double* pairwise_nbor, int pairwise_nborm, int pairwise_nborn)
}

%include "segmentation.h"
