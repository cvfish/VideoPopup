#include "segmentation.h"

#include "class.cpp"

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
            double overlap_cost)
{

  int points_num = unarym;
  int overlap_nbor_num = overlap_nborm;
  int pairwise_nbor_num = pairwise_nborm;
  int models_num = unaryn;

  // need to make sure the order of pbthresh is right, i.e. column major
  hypothesis H(points_num,
               overlap_nbor_num,
               models_num,
               lambda,
               label_costs,
               pairwise_nbor_num,
               pbthresh);

  if( pbthresh && pbthreshm > 1 && pbthreshn > 1)
    {
      H.thresh = 0;
      H._mcost = new double[ pbthreshm * pbthreshn ];
      for(int i = 0; i < pbthreshm; ++i )
        for(int j = 0; j < pbthreshn; ++j )
          H._mcost[ pbthreshm * j + i ] = pbthresh[ pbthreshn * i + j ];
    }

  H.overlap_penalty = overlap_cost;

  // make sure unary's order is right,
  H.un = new double[ points_num * models_num ];
  for(int i = 0; i < points_num; ++i)
    {
      for(int j = 0; j < models_num; ++j)
        {
          // if(H.un[ points_num * j + i ] > ppthresh[i])
          //   H.un[ points_num * j + i ] = ppthresh[i];
          H.un[ points_num * j + i ] = min( ppthresh[i], unary[ models_num * i + j ] );
        }
    }

  for(int i = 0; i < points_num; ++i)
    H.label[i] = interior_labels[i];

  // make sure overlap_nbor's order is right
  for(int i = 0; i < overlap_nbor_num; ++i)
    for(int j = 0; j < points_num; ++j)
      H.neigh_m[ overlap_nbor_num * j + i ] = overlap_nbor[ points_num * i + j ];

  // make sure pairwise_nbor's order is right
  for(int i = 0; i < pairwise_nbor_num; ++i)
    for(int j = 0; j < points_num; ++j)
      H.neigh_p[ pairwise_nbor_num * j + i ] = pairwise_nbor[ points_num * i + j ];

  // make sure _ncost's order is right
  if(pairwise_cost && pairwise_costm > 0 && pairwise_costn > 0)
    {
      H._ncost = new double[ pairwise_costm * pairwise_costn ];
      for(int i = 0; i < pairwise_costm; ++i)
        for(int j = 0; j < pairwise_costn; ++j)
          H._ncost[ pairwise_costm * j + i ] = pairwise_cost[ pairwise_costn * i + j ];
    }

  // solve the problem
  H.solve();

  // get the result, make sure mask_out's order is right
  double* mask_temp = new double[ points_num * models_num ];
  H.annotate( mask_temp );

  for(int i = 0; i < points_num; ++i)
    for(int j = 0; j < models_num; ++j)
      {
        mask_out[ models_num * i + j ] = mask_temp[ points_num * j + i ];
        if( H.un[ points_num * j + i ] == ppthresh[ i ] )
          mask_out[ models_num * i + j ] = 0;
      }

  for(int i = 0; i < points_num; ++i)
    labels_out[ i ] = H.label[i];

  // delete allocated memory, cannot find a better way to convert python's row-major array to
  // column-major 2d array used in the c++ file, cannot just pass python's column-major array
  // for some reason
  {
    if( pbthresh && pbthreshm > 1 && pbthreshn > 1)
      {
        delete [](H._mcost);
      }
    delete [](H.un);
    if(pairwise_cost && pairwise_costm > 0 && pairwise_costn > 0)
      {
        delete [](H._ncost);
      }
    delete [] mask_temp;

  }

  return;

}


void _expand(double* mask_out, int maskm, int maskn,
             double* labels_out, int labels_outn,
             double* unary, int unarym, int unaryn,
             double* pairwise_nbor, int pairwise_nborm, int pairwise_nborn,
             double* pairwise_cost, int pairwise_costm, int pairwise_costn,
             double* interior_labels, int interior_labelsn,
             double* label_costs, int label_costsn,
             double* ppthresh, int ppthreshn)
{

  _allgc(mask_out, maskm, maskn,
         labels_out, labels_outn,
         unary, unarym, unaryn,
         NULL, 0, 0,
         pairwise_nbor, pairwise_nborm, pairwise_nborn,
         pairwise_cost, pairwise_costm, pairwise_costn,
         interior_labels, interior_labelsn,
         0,
         label_costs, label_costsn,
         ppthresh, ppthreshn,
         NULL, 0, 0,
         0);
}

void _multi(double* mask_out, int maskm, int maskn,
            double* labels_out, int labels_outn,
            double* unary, int unarym, int unaryn,
            double* overlap_nbor, int overlap_nborm, int overlap_nborn,
            double* interior_labels, int interior_labelsn,
            double lambda, double* label_costs, int label_costsn,
            double* ppthresh, int ppthreshn)
{
  _allgc(mask_out, maskm, maskn,
         labels_out, labels_outn,
         unary, unarym, unaryn,
         overlap_nbor, overlap_nborm, overlap_nborn,
         NULL, 0, 0,
         NULL, 0, 0,
         interior_labels, interior_labelsn,
         lambda,
         label_costs, label_costsn,
         ppthresh, ppthreshn,
         NULL, 0, 0,
         0);
}
