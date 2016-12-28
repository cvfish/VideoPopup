
void _allgc(double* mask_out, int maskm, int maskn,
            double* labels_out, int labels_outn,
            double* unary, int unarym, int unaryn,
            double* overlap_nbor, int overlap_nborm, int overlap_nborn,
            double* pairwise_nbor, int pairwise_nborm, int pairwise_nborn,
            double* pairwise_cost, int pairwise_costm, int pairwise_costn,
            double* interior_labels, int interior_labelsn,
            double lambda, double* label_costs, int label_costsn,
            double* ppthresh, int ppthreshn,
            double* pbthresh, int pbthreshm, int pbthreshn,
            double overlap_cost);

void _expand(double* mask_out, int maskm, int maskn,
             double* labels_out, int labels_outn,
             double* unary, int unarym, int unaryn,
             double* pairwise_nbor, int pairwise_nborm, int pairwise_nborn,
             double* pairwise_cost, int pairwise_costm, int pairwise_costn,
             double* interior_labels, int interior_labelsn,
             double* label_costs, int label_costsn,
             double* ppthresh, int ppthreshn);

void _multi(double* mask_out, int maskm, int maskn,
            double* labels_out, int labels_outn,
            double* unary, int unarym, int unaryn,
            double* overlap_nbor, int overlap_nborm, int overlap_nborn,
            double* interior_labels, int interior_labelsn,
            double lambda, double* label_costs, int label_costsn,
            double* ppthresh, int ppthreshn);
