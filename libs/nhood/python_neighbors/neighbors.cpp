#include "neighbors.h"
#include <memory>
#include <algorithm>
#include <iostream>
#include <vector>
#include <queue>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <time.h>

using namespace std;

Neighbor::Neighbor(int frames, int points)
{
  frames_ = frames;
  points_ = points;
}

Neighbor::~Neighbor()
{
  FreeMemory();
}

void Neighbor::Initialize(double* measurement_matrix, int* visibility_matrix,
                          double velocity_weight, int neighbors_num, int top_k,
                          double pframe_thresh, int max_occ, double occ_penalty,
                          int* bottom_vector, int* top_vector)
{
  measurement_matrix_ = measurement_matrix;
  visibility_matrix_ = visibility_matrix;

  velocity_weight_ = velocity_weight;
  neighbors_num_ = neighbors_num;

  top_k_ = top_k;
  pframe_thresh_ = pframe_thresh;
  max_occ_ = max_occ;
  occ_penalty_ = occ_penalty;

  bottom_vector_ = bottom_vector;
  top_vector_ = top_vector;

  AllocateMemory();
}

void Neighbor::Initialize(double* measurement_matrix, int* visibility_matrix,
                          double velocity_weight, int neighbors_num, int top_k,
                          double pframe_thresh, double max_occ, double occ_penalty,
                          int* bottom_vector, int* top_vector,
                          double* color_matrix, double color_weight)
{

  measurement_matrix_ = measurement_matrix;
  visibility_matrix_ = visibility_matrix;

  color_matrix_ = color_matrix;
  velocity_weight_ = velocity_weight;

  top_k_ = top_k;
  neighbors_num_ = neighbors_num;

  // cout << "neighbors_num " << neighbors_num_ << endl;

  pframe_thresh_ = pframe_thresh;

  occ_penalty_ = occ_penalty;
  max_occ_ = max_occ;
  color_weight_ = color_weight;

  bottom_vector_ = bottom_vector;
  top_vector_ = top_vector;

  AllocateMemory();
}

void Neighbor::AllocateMemory()
{
  top_k_dist_ = new double[top_k_];
  top_k_vdist_ = new double[top_k_];
}

void Neighbor::FreeMemory()
{
  if(top_k_dist_)
    delete[] top_k_dist_;

  if(top_k_vdist_)
    delete[] top_k_vdist_;
}

void Neighbor::ComputeDistance(double* distances, int* neighbors)
{
  distances_ = distances;
  neighbors_ = neighbors;

  int elements_num = points_ * neighbors_num_;
  thresh_ = top_k_ * pframe_thresh_;
  std::fill_n(distances_, elements_num, thresh_);
  std::fill_n(neighbors_, elements_num, -1);

  int* V = visibility_matrix_;
  double* W = measurement_matrix_;

// clock_t t1 = clock();

// #pragma omp parallel for
  for(int i = 0; i < points_; ++i)
    {

      for(int j = i+1; j != points_; ++j)
        {
          int bottom = max(0, max(bottom_vector_[i], bottom_vector_[j]));
          int top = min(frames_, min(top_vector_[i], top_vector_[j]));
          int occ_frames = bottom + frames_ - top;

          if( top < bottom)
            continue;

          double cost = occ_frames * occ_penalty_ * top_k_;

          if(color_weight_ > 0)
            for(int k = 0; k < 3; ++k)
              {
                double color_diff = color_matrix_[ k*points_ + i ] - color_matrix_[ k*points_ + j ];
                cost += color_diff * color_diff * color_weight_ * top_k_;
              }

          std::fill_n(top_k_dist_, top_k_, 0);
          std::fill_n(top_k_vdist_, top_k_, 0);

          double upper_bound = max(distances_[ (neighbors_num_ - 1)*points_ + i ],
                                   distances_[ (neighbors_num_ - 1)*points_ + j ]);

          for(int frame = bottom; frame <= top && cost < upper_bound && occ_frames < max_occ_; ++frame)
            {

              if(V[frame * points_ + i] && V[frame * points_ + j])
                {

                  double diffx = W[2*frame*points_ + i] - W[2*frame*points_ + j];
                  double diffy = W[(2*frame + 1)*points_ + i] - W[(2*frame + 1)*points_ + j];
                  double dist = diffx*diffx + diffy * diffy;

                  if(dist > top_k_dist_[top_k_ - 1])
                    {
                      int m = top_k_ - 1;
                      cost -= top_k_dist_[m];
                      cost += dist;
                      while(m > 0 && dist > top_k_dist_[ m-1 ]){
                        top_k_dist_[m] = top_k_dist_[ m-1 ];
                        --m;
                      }
                      top_k_dist_[m] = dist;
                    }

                  if(frame < frames_ - 1 && V[(frame + 1) * points_ + i] && V[(frame + 1)*points_ + j])
                    {
                      double diffvx = diffx - W[(2*frame+2)*points_ + i] + W[(2*frame+2) *points_ + j];
                      double diffvy = diffy - W[(2*frame+3)*points_ + i] + W[(2*frame+3) *points_ + j];
                      double vdist = (diffvx * diffvx + diffvy * diffvy) * velocity_weight_;

                      if(vdist > top_k_vdist_[top_k_ - 1])
                        {
                          int n = top_k_ - 1;
                          cost -= top_k_vdist_[n];
                          cost += vdist;
                          while(n > 0 && vdist > top_k_vdist_[ n-1 ]){
                            top_k_vdist_[n] = top_k_vdist_[ n-1 ];
                            --n;
                          }
                          top_k_vdist_[n] = vdist;
                        }
                    }
                }
              else
                {
                  ++occ_frames;
                  cost += occ_penalty_ * top_k_;
                }
            }

          if(occ_frames >= max_occ_ || cost >= upper_bound)
            continue;

          if( cost < distances_[ (neighbors_num_ - 1)*points_ + i ] )
            {
              int k = neighbors_num_ - 1;
              while(k > 0 && cost < distances_[ (k-1)*points_ + i ] ){
                distances_[ k*points_ + i ] = distances_[ (k-1)*points_ + i ];
                neighbors_[ k*points_ + i ] = neighbors_[ (k-1)*points_ + i ];
                --k;
              }

              distances_[ k*points_ + i ] = cost;
              neighbors_[ k*points_ + i ] = j;
            }

          if(cost < distances_[ (neighbors_num_ - 1)*points_ + j ])
            {
              int k = neighbors_num_ - 1;
              while(k > 0 && cost < distances_[ (k-1)*points_ + j ]){
                distances_[ k*points_ + j ] = distances_[ (k-1)*points_ + j ];
                neighbors_[ k*points_ + j ] = neighbors_[ (k-1)*points_ + j ];
                --k;
              }

              distances_[ k*points_ + j ] = cost;
              neighbors_[ k*points_ + j ] = i;
            }

        }


      if( i % 1000 == 0)
        {

          cout << "computing neighbors, point " << i << endl;
          cout << "distances: ";
          for(int k = 0; k < neighbors_num_; ++k)
            {
              cout << distances_[k*points_ + i] << " ";
            }
          cout << endl;
          cout << "neighbors: ";
          for(int k = 0; k < neighbors_num_; ++k)
            {
              cout << neighbors_[k*points_ + i] << " ";
            }
          cout << endl;
        }
    }

  // clock_t t2 = clock();
  // cout << (float)(t2 -t1)/CLOCKS_PER_SEC << endl;

  for (int i = 0; i != neighbors_num_*points_; ++i)
    distances_[i] /= top_k_;

}

void Neighbor::ComputeDistance_old(double* distances, int* neighbors)
{
  distances_ = distances;
  neighbors_ = neighbors;

  int elements_num = points_ * neighbors_num_;
  thresh_ = top_k_ * pframe_thresh_;
  std::fill_n(distances_, elements_num, thresh_);
  std::fill_n(neighbors_, elements_num, -1);

  int* V = visibility_matrix_;
  double* W = measurement_matrix_;

  for(int i = 0; i < points_; ++i)
    {

      for(int j = i+1; j != points_; ++j)
        {
          int bottom = max(0, max(bottom_vector_[i], bottom_vector_[j]));
          int top = min(frames_, min(top_vector_[i], top_vector_[j]));
          int occ_frames = bottom + frames_ - top;

          if( top < bottom)
            continue;

          double cost = occ_frames * occ_penalty_ * top_k_;

          std::fill_n(top_k_dist_, top_k_, 0);
          std::fill_n(top_k_vdist_, top_k_, 0);

          double upper_bound = max(distances_[ (neighbors_num_ - 1)*points_ + i ],
                                   distances_[ (neighbors_num_ - 1)*points_ + j ]);

          for(int frame = bottom; frame <= top && cost < upper_bound && occ_frames < max_occ_; ++frame)
            {

              if(V[frame * points_ + i] && V[frame * points_ + j])
                {

                  double diffx = W[2*frame*points_ + i] - W[2*frame*points_ + j];
                  double diffy = W[(2*frame + 1)*points_ + i] - W[(2*frame + 1)*points_ + j];
                  double dist = diffx*diffx + diffy * diffy;

                  if(dist > top_k_dist_[top_k_ - 1])
                    {
                      int m = top_k_ - 1;
                      cost -= top_k_dist_[m];
                      cost += dist;
                      while(m > 0 && dist > top_k_dist_[ m-1 ]){
                        top_k_dist_[m] = top_k_dist_[ m-1 ];
                        --m;
                      }
                      top_k_dist_[m] = dist;
                    }

                  if(frame < frames_ - 1 && V[(frame + 1) * points_ + i] && V[(frame + 1)*points_ + j])
                    {
                      double diffvx = diffx - W[(2*frame+2)*points_ + i] + W[(2*frame+2) *points_ + j];
                      double diffvy = diffy - W[(2*frame+3)*points_ + i] + W[(2*frame+3) *points_ + j];
                      double vdist = (diffvx * diffvx + diffvy * diffvy) * velocity_weight_;

                      if(vdist > top_k_vdist_[top_k_ - 1])
                        {
                          int n = top_k_ - 1;
                          cost -= top_k_vdist_[n];
                          cost += vdist;
                          while(n > 0 && vdist > top_k_vdist_[ n-1 ]){
                            top_k_vdist_[n] = top_k_vdist_[ n-1 ];
                            --n;
                          }
                          top_k_vdist_[n] = vdist;
                        }
                    }
                }
              else
                {
                  ++occ_frames;
                  cost += occ_penalty_ * top_k_;
                }
            }

          if(occ_frames > max_occ_ || cost >= upper_bound)
            continue;

          if( cost < distances_[ (neighbors_num_ - 1)*points_ + i ] )
            {
              int k = neighbors_num_ - 1;
              while(k > 0 && cost < distances_[ (k-1)*points_ + i ] ){
                distances_[ k*points_ + i ] = distances_[ (k-1)*points_ + i ];
                neighbors_[ k*points_ + i ] = neighbors_[ (k-1)*points_ + i ];
                --k;
              }

              distances_[ k*points_ + i ] = cost;
              neighbors_[ k*points_ + i ] = j;
            }

          if(cost < distances_[ (neighbors_num_ - 1)*points_ + j ])
            {
              int k = neighbors_num_ - 1;
              while(k > 0 && cost < distances_[ (k-1)*points_ + j ]){
                distances_[ k*points_ + j ] = distances_[ (k-1)*points_ + j ];
                neighbors_[ k*points_ + j ] = neighbors_[ (k-1)*points_ + j ];
                --k;
              }

              distances_[ k*points_ + j ] = cost;
              neighbors_[ k*points_ + j ] = i;
            }

        }

      if( i % 1000 == 0)
        {
          cout << "computing neighbors, point " << i << endl;
          cout << "distances: ";
          for(int k = 0; k < neighbors_num_; ++k)
            {
              cout << distances_[k*points_ + i] << " ";
            }
          cout << endl;
          cout << "neighbors: ";
          for(int k = 0; k < neighbors_num_; ++k)
            {
              cout << neighbors_[k*points_ + i] << " ";
            }
          cout << endl;
        }


    }

  for (int i = 0; i != neighbors_num_*points_; ++i)
    distances_[i] /= top_k_;

}
