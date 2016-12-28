#include <assert.h>
#include "neighbors.h"

void get_nhood_old(double* measurement_matrix, int frames2F, int pointsNM,
                   int* visibility_matrix, int frames, int pointsNV,
                   double velocity_weight, int neighbors_num,
                   int top_k, double pframe_thresh,
                   int max_occ, double occ_penalty,
                   int* bottom_vector, int pointsB,
                   int* top_vector, int pointsT,
                   double* distances, int distancesM, int distancesN,
                   int* neighbors, int neighborsM, int neighborsN)
{

  assert(frames2F == 2*frames);
  assert(pointsNM == pointsNV);
  assert(pointsNV == pointsB);
  assert(pointsB == pointsT);
  assert(distancesM == neighbors_num);
  assert(distancesN == pointsNV);

  assert(neighborsM == distancesM);
  assert(neighborsN == distancesN);

  Neighbor nbor(frames, pointsNV);

  nbor.Initialize(measurement_matrix,
                  visibility_matrix,
                  velocity_weight,
                  neighbors_num,
                  top_k,
                  pframe_thresh,
                  max_occ,
                  occ_penalty,
                  bottom_vector,
                  top_vector);

  nbor.ComputeDistance_old(distances, neighbors);

}

void get_nhood(double* measurement_matrix, int frames2F, int pointsNM,
               int* visibility_matrix, int frames, int pointsNV,
               double velocity_weight, int neighbors_num,
               int top_k, double pframe_thresh,
               int max_occ, double occ_penalty,
               int* bottom_vector, int pointsB,
               int* top_vector, int pointsT,
               double* distances, int distancesM, int distancesN,
               int* neighbors, int neighborsM, int neighborsN,
               double* color_matrix, int channels, int pointsNC,
               double color_weight)
{
  assert(frames2F == 2*frames);
  assert(pointsNM == pointsNV);
  assert(pointsNV == pointsNC);
  assert(pointsNC == pointsB);
  assert(pointsB == pointsT);
  assert(channels == 3);

  assert(distancesM == neighbors_num);
  assert(distancesN == pointsNV);

  assert(neighborsM == distancesM);
  assert(neighborsN == distancesN);

  Neighbor nbor(frames, pointsNV);

  nbor.Initialize(measurement_matrix,
                  visibility_matrix,
                  velocity_weight,
                  neighbors_num,
                  top_k,
                  pframe_thresh,
                  max_occ,
                  occ_penalty,
                  bottom_vector,
                  top_vector,
                  color_matrix,
                  color_weight);

  nbor.ComputeDistance(distances, neighbors);

}
