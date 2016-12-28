// bundle adjustment error function created by Rui

// Orthographic case
// OrthoReprojectionErrorFull(Omarix,point)
// Omatrix = [a[0],a[1],a[2],a[3];
//           a[4],a[5],a[6],a[7]]
// OrthoReprojectionError(rotation,translation,point)
// OrthoReprojectionErrorWithQuaternions(quaternion,translation,point)

// Perspective case
// PerspReprojectionErrorFull(Pmatrix,point)
// Pmatrix = [a[0],a[1],a[2],a[3];
//           a[4],a[5],a[6],a[7];
//           a[8],a[9],a[10],a[11]]
// NormalizedPerspReprojectionError(rotation,translation,point),normarlize the measurement matrix with known intrinsics first
// NormalizedPerspReprojectionErrorWithQuaternions(quaternion,translation,point)
// PerspReprojectionErrorFixedK(rotation,translation,point),reprojection error with known intrinsics
// PerspReprojectionErrorFixedKWithQuaternions(quaternion,translation,point)
// PerspReprojectionError(K,rotation,translation,point)
// PerspReprojectionErrorWithQuaternions(K,quaternion,translation,point)


#ifndef CERES_EXAMPLES_RUI_REPROJECTION_ERROR_H_
#define CERES_EXAMPLES_RUI_REPROJECTION_ERROR_H_

#include "ceres/ceres.h"
#include "ceres/rotation.h"

#include <math.h>

#define OFFSET_ROT 3;
#define OFFSET_QUATERNION 4;
#define OFFSET_MOT_ORTHO 5;
#define OFFSET_MOT_PERSP 6;
#define OFFSET_MOT_ORTHO_QUATERNION 6;
#define OFFSET_MOT_PERSP_QUATERNION 7;

enum baType {
  BA_MOT,
  BA_STR,
  BA_MOTSTR
};

baType mapBA(std::string const& inString) {
  if (inString == "mot") return BA_MOT;
  if (inString == "str") return BA_STR;
  if (inString == "motstr") return BA_MOTSTR;
}

enum errorType {
  eOrthoReprojectionErrorFull,
  eOrthoReprojectionError,
  eOrthoReprojectionErrorWithQuaternions,
  ePerspReprojectionErrorFull,
  eNormalizedPerspReprojectionError,
  eNormalizedPerspReprojectionErrorWithQuaternions,
  ePerspReprojectionErrorFixedK,
  ePerspReprojectionErrorFixedKWithQuaternions,
  ePerspReprojectionError,
  ePerspReprojectionErrorWithQuaternions
};

errorType mapError(std::string const& inString) {
  if (inString == "OrthoReprojectionErrorFull") return eOrthoReprojectionErrorFull;
  if (inString == "OrthoReprojectionError") return eOrthoReprojectionError;
  if (inString == "OrthoReprojectionErrorWithQuaternions") return eOrthoReprojectionErrorWithQuaternions;
  if (inString == "PerspReprojectionErrorFull") return ePerspReprojectionErrorFull;
  if (inString == "NormalizedPerspReprojectionError") return eNormalizedPerspReprojectionError;
  if (inString == "NormalizedPerspReprojectionErrorWithQuaternions") return eNormalizedPerspReprojectionErrorWithQuaternions;
  if (inString == "PerspReprojectionErrorFixedK") return ePerspReprojectionErrorFixedK;
  if (inString == "PerspReprojectionErrorFixedKWithQuaternions") return ePerspReprojectionErrorFixedKWithQuaternions;
  if (inString == "PerspReprojectionError") return ePerspReprojectionError;
  if (inString == "PerspReprojectionErrorWithQuaternions") return ePerspReprojectionErrorWithQuaternions;
}

// Perspective Reprojection Error with Full Projection Matrix
// 3*4 projection matrix and 3*1 point

struct OrthoReprojectionErrorFull {

OrthoReprojectionErrorFull(double observed_x, double observed_y)
: observed_x(observed_x), observed_y(observed_y) {}

  template <typename T>
  bool operator()(const T* const camera,
		  const T* const point,
		  T* residuals) const {
    T p[3];

    // camera[0,1,2] are the translation.
    p[0] = camera[0]*point[0]+camera[1]*point[1]+camera[2]*point[2]+camera[3];
    p[1] = camera[4]*point[0]+camera[5]*point[1]+camera[6]*point[2]+camera[7];

    T xp = p[0];
    T yp = p[1];

    // Compute final projected point position.
    T predicted_x = xp;
    T predicted_y = yp;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - T(observed_x);
    residuals[1] = predicted_y - T(observed_y);

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double observed_x,
				     const double observed_y) {
    return (new ceres::AutoDiffCostFunction<OrthoReprojectionErrorFull, 2, 8, 3>(
										 new OrthoReprojectionErrorFull(observed_x, observed_y)));
  }

  double observed_x;
  double observed_y;

};

// Orthographic Reprojection Error
struct OrthoReprojectionError {
OrthoReprojectionError(double observed_x, double observed_y)
: observed_x(observed_x), observed_y(observed_y) {}

  template <typename T>
  bool operator()(const T* const camera_rotation,
		  const T* const camera_translation,
		  const T* const point,
		  T* residuals) const {
    // camera[0,1,2] are the angle-axis rotation.
    T p[3];
    ceres::AngleAxisRotatePoint(camera_rotation, point, p);

    // camera_translation are the xy translation.
    p[0] += camera_translation[0];
    p[1] += camera_translation[1];

    T xp = p[0];
    T yp = p[1];

    // Compute final projected point position.
    T predicted_x = xp;
    T predicted_y = yp;


    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - T(observed_x);
    residuals[1] = predicted_y - T(observed_y);

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double observed_x,
				     const double observed_y) {
    return (new ceres::AutoDiffCostFunction<OrthoReprojectionError, 2, 3, 2, 3>(
										new OrthoReprojectionError(observed_x, observed_y)));
  }

  double observed_x;
  double observed_y;

};

// point prior(shape prior)

struct PointNormPrior {

PointNormPrior(double alpha)
: alpha(alpha) {}

  template <typename T>
  bool operator()(const T* const point,
		  T* residuals) const {

    // The error is just the position

    T alpha_root = T(sqrt(alpha));

    residuals[0] = alpha_root*point[0];
    residuals[1] = alpha_root*point[1];
    residuals[2] = alpha_root*point[2];

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double alpha) {
    return (new ceres::AutoDiffCostFunction<PointNormPrior, 3, 3>(new PointNormPrior(alpha)));
  }

  double alpha;

};

// Depth relief prior for projective full matrix case
struct ProjFullDepthReliefPriorFast{
ProjFullDepthReliefPriorFast(double alpha_relief)
: alpha_relief(alpha_relief){}

  template <typename T>
  bool operator()(const T* const mean,
		  const T* const camera,
		  const T* const point,
		  T* residuals) const {


    T depth;
    depth = camera[8]*point[0]+camera[9]*point[1]+camera[10]*point[2]+camera[11];

    T alpha_relief_root = T(sqrt(alpha_relief));
    //residuals[0] = alpha_relief_root*(depth - mean[0])/mean[0];
    residuals[0] = alpha_relief_root*(depth-mean[0]);

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(double alpha_relief) {
    return (new ceres::AutoDiffCostFunction<ProjFullDepthReliefPriorFast, 1, 1, 12, 3>(new ProjFullDepthReliefPriorFast(alpha_relief)));
  }

  double alpha_relief;

};

// scale invariant relief prior
/* struct DepthReliefPrior { */

/* DepthReliefPrior(double n, double alpha_relief) */
/* : n(n),alpha_relief(alpha_relief){} */

/*   template <typename T> */
/*   bool operator()(const T* const T* motshape, */
/* 		  T* residuals) const { */

/*     T mean; */
/*     mean = 0; */
/*     residual[0] = 0; */
/*     for(int j=0; j<n; ++j){	 */
/*       T p[3]; */
/*       ceres::QuaternionRotatePoint(motshape[0],motshape[j+1],p); */
/*       mean += p[3]; */
/*       residual[0] += p[3]*p[3]; */
/*       } */
/*     mean = mean/n; */
/*     residual[0] = residual[0]/(mean*mean) - n; */


/*     return true; */
/*   } */

/*   // Factory to hide the construction of the CostFunction object from */
/*   // the client code. */
/*   static ceres::CostFunction* Create(const double n, double alpha_relief) { */
/*     return (new ceres::AutoDiffCostFunction<DepthReliefPrior, 3, 4, 2, 4,2>(new DepthReliefPrior(n,alpha_relief))); */
/*   } */

/*   double n; */
/*   double alpha_relief; */

/* }; */

// scale invariant relief prior
struct DepthReliefPriorFast{

DepthReliefPriorFast(double alpha_relief)
: alpha_relief(alpha_relief){}

  template <typename T>
  bool operator()(const T* const mean,
		  const T* const camera_rotation,
		  const T* const point,
		  T* residuals) const {

    T p1[3];
    ceres::QuaternionRotatePoint(camera_rotation, point, p1);

    T alpha_relief_root = T(sqrt(alpha_relief));
    //residuals[0] = alpha_relief_root*(p1[2]-mean[0])/mean[0];
    residuals[0] = alpha_relief_root*(p1[2]-mean[0]);

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(double alpha_relief) {
    return (new ceres::AutoDiffCostFunction<DepthReliefPriorFast, 1, 1, 4, 3>(new DepthReliefPriorFast(alpha_relief)));
  }

  double alpha_relief;

};

// orthographic motion prior(based on quaternion and translation difference between consecutive frames)
struct OrthoMotionSmoothPrior {

OrthoMotionSmoothPrior(double alpha_r, double alpha_t)
: alpha_r(alpha_r),alpha_t(alpha_t) {}

  template <typename T>
  bool operator()(const T* const camera_rotation_first,
		  const T* const camera_translation_first,
		  const T* const camera_rotation_second,
		  const T* const camera_translation_second,
		  T* residuals) const {

    T alpha_r_root = T(sqrt(alpha_r));
    T rot_diff =  (abs(camera_rotation_first[0]*camera_rotation_second[0] +
		       camera_rotation_first[1]*camera_rotation_second[1] +
		       camera_rotation_first[2]*camera_rotation_second[2] +
		       camera_rotation_first[3]*camera_rotation_second[3])-T(1));

    T trans_diff[2];
    T alpha_t_root = T(sqrt(alpha_t));
    trans_diff[0] = camera_translation_first[0]-camera_translation_second[0];
    trans_diff[1] = camera_translation_first[1]-camera_translation_second[1];

    residuals[0] = alpha_r_root*rot_diff;
    residuals[1] = alpha_t_root*trans_diff[0];
    residuals[2] = alpha_t_root*trans_diff[1];

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double alpha_r,
				     const double alpha_t) {
    return (new ceres::AutoDiffCostFunction<OrthoMotionSmoothPrior, 3, 4, 2, 4,2>(new OrthoMotionSmoothPrior(alpha_r, alpha_t)));
  }

  double alpha_r;
  double alpha_t;

};


// perspective motion prior(based on quaternion and translation difference between consecutive frames)
struct PerspMotionSmoothPrior {

PerspMotionSmoothPrior(double alpha_r, double alpha_t)
: alpha_r(alpha_r),alpha_t(alpha_t) {}

  template <typename T>
  bool operator()(const T* const camera_rotation_first,
		  const T* const camera_translation_first,
		  const T* const camera_rotation_second,
		  const T* const camera_translation_second,
		  T* residuals) const {

    T alpha_r_root = T(sqrt(alpha_r));
    T rot_diff =  (abs(camera_rotation_first[0]*camera_rotation_second[0] +
		       camera_rotation_first[1]*camera_rotation_second[1] +
		       camera_rotation_first[2]*camera_rotation_second[2] +
		       camera_rotation_first[3]*camera_rotation_second[3])-T(1));

    T trans_diff[3];
    T alpha_t_root = T(sqrt(alpha_t));
    trans_diff[0] = camera_translation_first[0]-camera_translation_second[0];
    trans_diff[1] = camera_translation_first[1]-camera_translation_second[1];
    trans_diff[2] = camera_translation_first[2]-camera_translation_second[2];

    residuals[0] = alpha_r_root*rot_diff;
    residuals[1] = alpha_t_root*trans_diff[0];
    residuals[2] = alpha_t_root*trans_diff[1];
    residuals[3] = alpha_t_root*trans_diff[2];

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double alpha_r,
				     const double alpha_t) {
    return (new ceres::AutoDiffCostFunction<PerspMotionSmoothPrior, 4, 4, 3, 4,3>(new PerspMotionSmoothPrior(alpha_r, alpha_t)));
  }

  double alpha_r;
  double alpha_t;

};


// depth difference prior(the depth value between consecutive frames)
struct DepthDiffPrior {

DepthDiffPrior(double alpha)
: alpha(alpha) {}

  template <typename T>
  bool operator()(const T* const camera_rotation_first,
		  const T* const camera_rotation_second,
		  const T* const point,
		  T* residuals) const {

    // The error is just the position

    T p1[3],p2[3];
    ceres::QuaternionRotatePoint(camera_rotation_first, point, p1);
    ceres::QuaternionRotatePoint(camera_rotation_second, point, p2);

    T alpha_root = T(sqrt(alpha));

    residuals[0] = alpha_root*(p1[2]-p2[2]);

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double alpha) {
    return (new ceres::AutoDiffCostFunction<DepthDiffPrior, 1, 4, 4, 3>(new DepthDiffPrior(alpha)));
  }

  double alpha;

};

// depth norm prior(the depth value between consecutive frames)
struct DepthNormPrior {

DepthNormPrior(double alpha)
: alpha(alpha) {}

  template <typename T>
  bool operator()(const T* const camera_rotation,
		  const T* const point,
		  T* residuals) const {

    // The error is just the position

    T p[3];
    ceres::QuaternionRotatePoint(camera_rotation, point, p);

    T alpha_root = T(sqrt(alpha));
    residuals[0] = alpha_root*p[2];

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double alpha) {
    return (new ceres::AutoDiffCostFunction<DepthNormPrior, 1, 4, 3>(new DepthNormPrior(alpha)));
  }

  double alpha;

};

// Orthographic Reprojection Error Using Quaternions

struct OrthoReprojectionErrorWithQuaternions {

OrthoReprojectionErrorWithQuaternions(double observed_x, double observed_y)
: observed_x(observed_x), observed_y(observed_y) {}

  template <typename T>
  bool operator()(const T* const camera_rotation,
		  const T* const camera_translation,
		  const T* const point,
		  T* residuals) const {
    // camera[0,1,2] are the angle-axis rotation.
    T p[3];
    ceres::QuaternionRotatePoint(camera_rotation, point, p);

    // camera_translation[0,1] are the xy translation.
    p[0] += camera_translation[0];
    p[1] += camera_translation[1];

    T xp = p[0];
    T yp = p[1];

    // Compute final projected point position.
    T predicted_x = xp;
    T predicted_y = yp;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - T(observed_x);
    residuals[1] = predicted_y - T(observed_y);

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double observed_x,
				     const double observed_y) {
    return (new ceres::AutoDiffCostFunction<OrthoReprojectionErrorWithQuaternions, 2, 4, 2, 3>(
											       new OrthoReprojectionErrorWithQuaternions(observed_x, observed_y)));
  }

  double observed_x;
  double observed_y;

};

// Perspective Reprojection Error with Full Projection Matrix
// 3*4 projection matrix and 3*1 point

struct PerspReprojectionErrorFull {

PerspReprojectionErrorFull(double observed_x, double observed_y)
: observed_x(observed_x), observed_y(observed_y) {}

  template <typename T>
  bool operator()(const T* const camera,
		  const T* const point,
		  T* residuals) const {
    T p[3];

    p[0] = camera[0]*point[0]+camera[1]*point[1]+camera[2]*point[2]+camera[3];
    p[1] = camera[4]*point[0]+camera[5]*point[1]+camera[6]*point[2]+camera[7];
    p[2] = camera[8]*point[0]+camera[9]*point[1]+camera[10]*point[2]+camera[11];

    T xp = p[0] / p[2];
    T yp = p[1] / p[2];

    // Compute final projected point position.
    T predicted_x = xp;
    T predicted_y = yp;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - T(observed_x);
    residuals[1] = predicted_y - T(observed_y);

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double observed_x,
				     const double observed_y) {
    return (new ceres::AutoDiffCostFunction<PerspReprojectionErrorFull, 2, 12, 3>(
										  new PerspReprojectionErrorFull(observed_x, observed_y)));
  }

  double observed_x;
  double observed_y;

};

// Perspective Reprojection Error, using normalized image measurements obtained with known intrinsics
struct NormalizedPerspReprojectionError {

NormalizedPerspReprojectionError(double observed_x, double observed_y)
: observed_x(observed_x), observed_y(observed_y) {}

  template <typename T>
  bool operator()(const T* const camera_rotation,
		  const T* const camera_translation,
		  const T* const point,
		  T* residuals) const {
    // camera[0,1,2] are the angle-axis rotation.
    T p[3];
    ceres::AngleAxisRotatePoint(camera_rotation, point, p);

    // camera[0,1,2] are the translation.
    p[0] += camera_translation[0];
    p[1] += camera_translation[1];
    p[2] += camera_translation[2];

    T xp = p[0] / p[2];
    T yp = p[1] / p[2];

    // Compute final projected point position.
    T predicted_x = xp;
    T predicted_y = yp;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - T(observed_x);
    residuals[1] = predicted_y - T(observed_y);

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double observed_x,
				     const double observed_y) {
    return (new ceres::AutoDiffCostFunction<NormalizedPerspReprojectionError, 2, 3, 3, 3>(
											  new NormalizedPerspReprojectionError(observed_x, observed_y)));
  }

  double observed_x;
  double observed_y;

};

// Perspective Reprojection Error Using Quaternions, using normalized image measurements obtained with known intrinsics
struct NormalizedPerspReprojectionErrorWithQuaternions {

NormalizedPerspReprojectionErrorWithQuaternions(double observed_x, double observed_y)
: observed_x(observed_x), observed_y(observed_y) {}

  template <typename T>
  bool operator()(const T* const camera_rotation,
		  const T* const camera_translation,
		  const T* const point,
		  T* residuals) const {
    // camera[0,1,2] are the angle-axis rotation.
    T p[3];
    ceres::QuaternionRotatePoint(camera_rotation, point, p);

    // camera_translation[0,1,2] are the translation.
    p[0] += camera_translation[0];
    p[1] += camera_translation[1];
    p[2] += camera_translation[2];

    T xp = p[0] / p[2];
    T yp = p[1] / p[2];

    // Compute final projected point position.
    T predicted_x = xp;
    T predicted_y = yp;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - T(observed_x);
    residuals[1] = predicted_y - T(observed_y);

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double observed_x,
				     const double observed_y) {
    return (new ceres::AutoDiffCostFunction<NormalizedPerspReprojectionErrorWithQuaternions, 2, 4, 3, 3>(                                                                                                         new NormalizedPerspReprojectionErrorWithQuaternions(observed_x, observed_y)));
  }

  double observed_x;
  double observed_y;

};

// Perspective Reprojection Error, obtained with known intrinsics
struct PerspReprojectionErrorFixedK {

PerspReprojectionErrorFixedK(double observed_x, double observed_y, double *intrinsics)
: observed_x(observed_x), observed_y(observed_y), intrinsics(intrinsics) {}

  template <typename T>
  bool operator()(const T* const camera_rotation,
		  const T* const camera_translation,
		  const T* const point,
		  T* residuals) const {
    // camera[0,1,2] are the angle-axis rotation.
    T p[3];
    ceres::AngleAxisRotatePoint(camera_rotation, point, p);

    // camera_translation[0,1,2] are the translation.
    p[0] += camera_translation[0];
    p[1] += camera_translation[1];
    p[2] += camera_translation[2];

    // intrinsics[0,1,2,3,4] : k11, k12, k22, k13, k23
    T xp = p[0]*T(intrinsics[0]) / p[2] + T(intrinsics[3]);
    T yp = p[1]*T(intrinsics[2]) / p[2] + T(intrinsics[4]);

    // Compute final projected point position.
    T predicted_x = xp;
    T predicted_y = yp;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - T(observed_x);
    residuals[1] = predicted_y - T(observed_y);

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double observed_x,
				     const double observed_y,
				     double *intrinsics) {
    return (new ceres::AutoDiffCostFunction<PerspReprojectionErrorFixedK, 2, 3, 3, 3>(
										      new PerspReprojectionErrorFixedK(observed_x, observed_y, intrinsics)));
  }

  double observed_x;
  double observed_y;
  double *intrinsics;

};

// Perspective Reprojection Error, obtained with known intrinsics
struct PerspReprojectionErrorFixedKWithQuaternions {

PerspReprojectionErrorFixedKWithQuaternions(double observed_x, double observed_y,double *intrinsics)
: observed_x(observed_x), observed_y(observed_y), intrinsics(intrinsics) {}

  template <typename T>
  bool operator()(const T* const camera_rotation,
		  const T* const camera_translation,
		  const T* const point,
		  T* residuals) const {
    // camera[0,1,2] are the angle-axis rotation.
    T p[3];
    ceres::QuaternionRotatePoint(camera_rotation, point, p);

    // camera_translation[0,1,2] are the translation.
    p[0] += camera_translation[0];
    p[1] += camera_translation[1];
    p[2] += camera_translation[2];

    // intrinsics[0,1,2,3,4] : k11, k12, k22, k13, k23
    T xp = p[0]*T(intrinsics[0]) / p[2] + T(intrinsics[3]);
    T yp = p[1]*T(intrinsics[2]) / p[2] + T(intrinsics[4]);

    // Compute final projected point position.
    T predicted_x = xp;
    T predicted_y = yp;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - T(observed_x);
    residuals[1] = predicted_y - T(observed_y);

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double observed_x,
				     const double observed_y,
				     double *intrinsics) {
    return (new ceres::AutoDiffCostFunction<PerspReprojectionErrorFixedKWithQuaternions, 2, 4, 3, 3>(
												     new PerspReprojectionErrorFixedKWithQuaternions(observed_x, observed_y, intrinsics)));
  }

  double observed_x;
  double observed_y;
  double *intrinsics;

};

// PerspReprojectionError(K,rotation,translation,point)
// Perspective Reprojection Error, obtained with known intrinsics
struct PerspReprojectionError {

PerspReprojectionError(double observed_x, double observed_y)
: observed_x(observed_x), observed_y(observed_y) {}

  template <typename T>
  bool operator()(const T* const camera_intrinsics,
		  const T* const camera_rotation,
		  const T* const camera_translation,
		  const T* const point,
		  T* residuals) const {
    // camera[0,1,2] are the angle-axis rotation.
    T p[3];
    ceres::AngleAxisRotatePoint(camera_rotation, point, p);

    // camera_translation[0,1,2] are the translation.
    p[0] += camera_translation[0];
    p[1] += camera_translation[1];
    p[2] += camera_translation[2];

    // intrinsics[0,1,2,3,4] : k11, k12, k22, k13, k23
    T xp = p[0]*camera_intrinsics[0] / p[2] + camera_intrinsics[3];
    T yp = p[1]*camera_intrinsics[2] / p[2] + camera_intrinsics[4];

    // Compute final projected point position.
    T predicted_x = xp;
    T predicted_y = yp;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - T(observed_x);
    residuals[1] = predicted_y - T(observed_y);

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double observed_x,
				     const double observed_y) {
    return (new ceres::AutoDiffCostFunction<PerspReprojectionError, 2, 5, 3, 3, 3>(
										   new PerspReprojectionError(observed_x, observed_y)));
  }

  double observed_x;
  double observed_y;

};

// PerspReprojectionErrorWithQuaternions(K,quaternion,translation,point)
struct PerspReprojectionErrorWithQuaternions {

PerspReprojectionErrorWithQuaternions(double observed_x, double observed_y)
: observed_x(observed_x), observed_y(observed_y) {}

  template <typename T>
  bool operator()(const T* const camera_intrinsics,
		  const T* const camera_rotation,
		  const T* const camera_translation,
		  const T* const point,
		  T* residuals) const {
    // camera[0,1,2] are the angle-axis rotation.
    T p[3];
    ceres::QuaternionRotatePoint(camera_rotation, point, p);

    // camera_translation[0,1,2] are the translation.
    p[0] += camera_translation[0];
    p[1] += camera_translation[1];
    p[2] += camera_translation[2];

    // intrinsics[0,1,2,3,4] : k11, k12, k22, k13, k23
    T xp = p[0]*camera_intrinsics[0] / p[2] + camera_intrinsics[3];
    T yp = p[1]*camera_intrinsics[2] / p[2] + camera_intrinsics[4];

    // Compute final projected point position.
    T predicted_x = xp;
    T predicted_y = yp;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - T(observed_x);
    residuals[1] = predicted_y - T(observed_y);

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double observed_x,
				     const double observed_y) {
    return (new ceres::AutoDiffCostFunction<PerspReprojectionErrorWithQuaternions, 2, 5, 4, 3, 3>(
												  new PerspReprojectionErrorWithQuaternions(observed_x, observed_y)));
  }

  double observed_x;
  double observed_y;

};

// motion prior for Tassos(efficient one,version 2)
struct MotionDiffWithPointv2{

MotionDiffWithPointv2(double alpha, double *vvt, double *u)
: alpha(alpha),vvt(vvt),u(u) {}

  template <typename T>
  bool operator()(const T* const camera_rotation_first,
                  const T* const camera_trans_first,
		  const T* const camera_rotation_second,
                  const T* const camera_trans_second,
		  T* residuals) const {

    // camera[0,1,2] are the angle-axis rotation.
    T point[12];
    point[0] = T(u[0]);
    point[1] = T(u[1]);
    point[2] = T(u[2]);
    point[3] = T(u[4]);
    point[4] = T(u[5]);
    point[5] = T(u[6]);
    point[6] = T(u[8]);
    point[7] = T(u[9]);
    point[8] = T(u[10]);
    point[9] = T(u[12]);
    point[10] = T(u[13]);
    point[11] = T(u[14]);

    T alpha_root = T(sqrt(alpha));
    //    T cost = T(0);

    for(int i=0; i<4; ++i){
      T p1[3];
      ceres::AngleAxisRotatePoint(camera_rotation_first, &point[3*i], p1);
      // camera_translation[0,1,2] are the translation.
      p1[0] += T(u[4*i+3])*camera_trans_first[0];
      p1[1] += T(u[4*i+3])*camera_trans_first[1];
      p1[2] += T(u[4*i+3])*camera_trans_first[2];

      /* p1[0] = camera_trans_first[0]; */
      /* p1[1] = camera_trans_first[1]; */
      /* p1[2] = camera_trans_first[2]; */


      T p2[3];
      ceres::AngleAxisRotatePoint(camera_rotation_second, &point[3*i], p2);
      // camera_translation[0,1,2] are the translation.
      p2[0] += T(u[4*i+3])*camera_trans_second[0];
      p2[1] += T(u[4*i+3])*camera_trans_second[1];
      p2[2] += T(u[4*i+3])*camera_trans_second[2];

      /* p2[0] = camera_trans_second[0]; */
      /* p2[1] = camera_trans_second[1]; */
      /* p2[2] = camera_trans_second[2]; */

      T p[3];
      p[0] = p1[0] - p2[0];
      p[1] = p1[1] - p2[1];
      p[2] = p1[2] - p2[2];

      residuals[3*i] = alpha_root*p[0];
      residuals[3*i+1] = alpha_root*p[1];
      residuals[3*i+2] = alpha_root*p[2];

      //      cost += T(alpha)*ceres::DotProduct(p,p);

    }

    //    residuals[0] = sqrt(cost);
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double alpha, double *vvt, double *u) {
    return (new ceres::AutoDiffCostFunction<MotionDiffWithPointv2, 12, 3, 3, 3, 3>(new MotionDiffWithPointv2(alpha,vvt,u)));
  }

  double alpha;
  double *vvt;
  double *u;
};


/* // Make fixed cameras or points constant */
/* ceres::SubsetParameterization *const_parameterization = NULL; */

/* std::vector<int> const_fixed_cam_index; */
/* for(int ind = 0; ind < cnp*mcon; ++ind) */
/*   const_fixed_cam_index.push_back(ind); */
/* const_parameterization = new ceres::SubsetParameterization(cnp*m+pnp*n,const_fixed_cam_index); */
/* problem.SetParameterization(p,const_parameterization); */

/* std::vector<int> const_fixed_point_index; */
/* for(int ind = cnp*m; ind < cnp*m+m+pnp*ncon; ++ind) */
/*   const_fixed_point_index.push_back(ind); */
/* const_parameterization = new ceres::SubsetParameterization(cnp*m+pnp*n,const_fixed_point_index); */
/* problem.SetParameterization(p,const_parameterization);   */

/* // BA for motion or struture only, mot, str , motstr */
/* std::vector<int> const_index; */
/* switch(ba){ */
/*  case BA_MOTSTR: */
/*    break; */
/*  case BA_MOT: */
/*    { */
/*      for(int ind = cnp*m; ind < cnp*m+pnp*n ; ++ind) */
/*        const_index.push_back(ind); */
/*      break; */
/*    } */
/*  case BA_STR: */
/*    { */
/*      for(int ind = 0; ind < cnp*m ; ++ind) */
/*        const_index.push_back(ind); */
/*      break; */
/*    } */
/*  } */
/* const_parameterization = new ceres::SubsetParameterization(cnp*m+pnp*n,const_index); */
/* problem.SetParameterization(p,const_parameterization); */

/* // Intrinsics mask */
/* std::vector<int> const_intrinsics_index; */
/* if(in_num > 0){ */
/*   for(int ind=0; ind<m; ++ind) */
/*     { */
/*       for(int in_iter=0;in_iter<in_num;++in_iter) */
/* 	if(!in_mask[in_iter]) */
/* 	  const_intrinsics_index.push_back(cnp*ind-in_num+in_iter);		 */
/*     } */
/*  } */
/* const_parameterization = new ceres::SubsetParameterization(cnp*m+pnp*n,const_intrinsics_index); */
/* problem.SetParameterization(p,const_parameterization);	   */


#endif  // CERES_EXAMPLES_RUI_REPROJECTION_ERROR_H_
