class Neighbor
{
public:
  Neighbor(int frames, int points);
  ~Neighbor();

  void Initialize(double* measurement_matrix, int* visibility_matrix,
                  double velocity_weight, int neighbors_num, int top_k,
                  double pframe_thresh, int max_occ, double occ_penalty,
                  int* bottom_vector, int* top_vector);

  void Initialize(double* measurement_matrix, int* visibility_matrix,
                  double velocity_weight, int neighbors_num, int top_k,
                  double pframe_thresh, double max_occ, double occ_penalty,
                  int* bottom_vector, int* top_vector,
                  double* color_matrix, double color_weight);

  void AllocateMemory();
  void FreeMemory();

  void ComputeDistance(double* distances, int* neighbors);
  void ComputeDistance_old(double* distances, int* neighbors);

private:

  // 2F*N measurement matrix
  double* measurement_matrix_;
  // F*N visibility matrix
  int* visibility_matrix_;
  // 3*N rgb matrix
  double* color_matrix_;
  // last visible frame of tracks
  int* top_vector_;
  // first visible frame of tracks
  int* bottom_vector_;

  int neighbors_num_;
  double velocity_weight_;
  double color_weight_;
  double occ_penalty_;
  // maximum occlusions allowed
  int max_occ_;
  // use top k distances for position and velocity
  int top_k_;
  // per frame threshold
  double pframe_thresh_;
  double thresh_;

  int frames_;
  int points_;

  // distance matrix output
  // neighbors x points matrix
  double* distances_;
  // neighbors output, index from 0, -1 means no neighbors
  // neighbors x points matrix
  int* neighbors_;

  // tempary variables
  // top k distances
  double* top_k_dist_;
  // top k velocity distances
  double* top_k_vdist_;

};
