#include <cmath>
#include <cstdio>
#include <iostream>

using namespace std;

#include "rui_reprojection_error.h"

void add_ortho_prior(ceres::Problem &problem,double *p, int cnp, int pnp, int m, int n,
                     int prior_model, double *prior_para, int prior_paran, double *mprior_mask)
{

  ceres::CostFunction* cost_function;

  switch(prior_model){

  case 0:
    {
      // shape prior
      for(int j=0; j < n; ++j){
        cost_function =
          PointNormPrior::Create(prior_para[0]);
        problem.AddResidualBlock(cost_function,
                                 NULL /* squared loss */,
                                 &p[m*cnp+j*pnp]);
      }

      /* double hello; */
      /* double cost = 0.0; */
      /* std::vector<double> residuals; */
      /* std::vector<double> gradient; */

      /* std::vector<double*> parameter_blocks_1; */
      /* parameter_blocks_1.push_back(&p[m*cnp+j*pnp]); */
      /* // parameter_blocks_1.push_back(&p[m*cnp+j*pnp+1]); */
      /* // parameter_blocks_1.push_back(&p[m*cnp+j*pnp+2]); */


      /* auto evalOptions = ceres::Problem::EvaluateOptions(); */
      /* evalOptions.parameter_blocks = parameter_blocks_1; */
      /* problem.Evaluate(evalOptions, &cost, &residuals, &gradient, NULL); */
      /* int testsize = residuals.size(); */
      /* double test = 1.0; */

      /* hello = sqrt(1.0/pVisNum[j]*prior_para[0]); */
      break;
    }

  case 1:
    {
      // depth mean prior  prior
      for(int i=0; i < m; ++i){
        for(int j= 0; j < n; ++j){
          cost_function =
            DepthNormPrior::Create(1.0/m*prior_para[0]);
          problem.AddResidualBlock(cost_function,
                                   NULL /* squared loss */,
                                   &p[i*cnp],
                                   &p[m*cnp+j*pnp]);
        }
      }
      break;
    }

  case 2:
    {
      // depth norm prior
      for(int i=0; i<m; ++i){
        for(int j=0; j<n; ++j){
          cost_function =
            DepthNormPrior::Create(prior_para[0]);
          problem.AddResidualBlock(cost_function,
                                   NULL /* squared loss */,
                                   &p[i*cnp],
                                   &p[m*cnp+j*pnp]);
        }
      }
      break;
    }
  }

  // depth difference prior
  for(int i=0; i < m-1; ++i){
    if(0 == mprior_mask[i])
      continue;
    for(int j=0; j<n; ++j){
      cost_function =
        DepthDiffPrior::Create(prior_para[1]);
      problem.AddResidualBlock(cost_function,
                               NULL /* squared loss */,
                               &p[i*cnp],
                               &p[(i+1)*cnp],
                               &p[m*cnp+j*pnp]);

      /* double cost = 0.0; */
      /* std::vector<double> residuals; */
      /* std::vector<double> gradient; */
      /* problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, &residuals, &gradient, NULL); */
      /* int testsize = residuals.size(); */
      /* double test = 1.0; */
    }
  }

  // motion smoothness prior, only implemented for quaternion case
  for(int i=0; i < m-1 ; ++i){
    if(0 == mprior_mask[i])
      continue;
    if(prior_paran == 3)
      cost_function =
        OrthoMotionSmoothPrior::Create(prior_para[2],prior_para[2]);
    else
      cost_function =
        OrthoMotionSmoothPrior::Create(prior_para[2],prior_para[3]);

    problem.AddResidualBlock(cost_function,
                             NULL /* squared loss */,
                             &p[i*cnp],
                             &p[i*cnp+4],
                             &p[(i+1)*cnp],
                             &p[(i+1)*cnp+4]);

    // double cost = 0.0;
    // std::vector<double> residuals;
    // std::vector<double> gradient;
    // problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, &residuals, &gradient, NULL);
    // int testsize = residuals.size();
    // double test = 1.0;

  }

}

void add_persp_prior(ceres::Problem &problem,double *p, int cnp, int pnp, int m, int n,
                     double *prior_para, errorType &inputError, double *depth_mean, double *mprior_mask)
{

  ceres::CostFunction* cost_function;

  if(inputError != ePerspReprojectionErrorFull){

    if(prior_para[2]!=0 || prior_para[3]!=0){

      for(int i=0; i < m-1 ; ++i){

        if(0 == mprior_mask[i])
          continue;
        cost_function =
          PerspMotionSmoothPrior::Create(prior_para[2],prior_para[3]);

        problem.AddResidualBlock(cost_function,
                                 NULL /* squared loss */,
                                 &p[i*cnp],
                                 &p[i*cnp+4],
                                 &p[(i+1)*cnp],
                                 &p[(i+1)*cnp+4]);

      }
    }
  }

  if(prior_para[4] != 0){

    if(inputError == ePerspReprojectionErrorFull)
      {
        double depth;
        double *camera,*point;
        // initialize depth value
        for(int i=0; i<m; ++i){
          depth_mean[i] = 0;
          camera = &p[i*cnp];
          for(int j=0; j<n; ++j){
            point = &p[m*cnp+j*pnp];
            depth = camera[8]*point[0]+camera[9]*point[1]+camera[10]*point[2]+camera[11];
            depth_mean[i] += depth; // only cares about depth
          }
          depth_mean[i] /= n;
        }

        for(int i=0; i<m; ++i){
          for(int j=0; j<n; ++j){
            cost_function = ProjFullDepthReliefPriorFast::Create(prior_para[4]);
            problem.AddResidualBlock(cost_function,
                                     NULL,
                                     &depth_mean[i],
                                     &p[i*cnp],
                                     &p[m*cnp+j*pnp]);
          }
        }
      }
    else{
      double point[3];
      // initialize depth value
      for(int i=0; i<m; ++i){
        depth_mean[i] = 0;
        for(int j=0; j<n; ++j){
          ceres::QuaternionRotatePoint(&p[i*cnp],&p[m*cnp+j*pnp],point);
          depth_mean[i] += point[2]; // only cares about depth
        }
        depth_mean[i] /= n;
      }

      for(int i=0; i<m; ++i){
        for(int j=0; j<n; ++j){
          cost_function = DepthReliefPriorFast::Create(prior_para[4]);
          problem.AddResidualBlock(cost_function,
                                   NULL,
                                   &depth_mean[i],
                                   &p[i*cnp],
                                   &p[m*cnp+j*pnp]);

        }
      }
    }
  }
}

void set_const_camera(ceres::Problem &problem, double *p, int cnp, int i, int shift_trans, int shift_k)
{
  if(problem.HasParameterBlock(p+cnp*i))
    problem.SetParameterBlockConstant(p+cnp*i);
  if(shift_trans && problem.HasParameterBlock(p+cnp*i+shift_trans))
    problem.SetParameterBlockConstant(p+cnp*i+shift_trans);
  if(shift_k && problem.HasParameterBlock(p+cnp*i+shift_k))
    problem.SetParameterBlockConstant(p+cnp*i+shift_k);
}

void add_const_mask(ceres::Problem &problem,double *p,int cnp, int pnp, int mcon, int ncon, int m, int n,
                    int in_num, double *in_mask, errorType &inputError, baType ba, double *cconst_mask)
{
  int shift_trans,shift_k;
  shift_trans = 0;
  shift_k = 0;

  if(inputError == eOrthoReprojectionErrorWithQuaternions || inputError == eNormalizedPerspReprojectionErrorWithQuaternions || inputError == ePerspReprojectionErrorWithQuaternions)
    shift_trans=4;
  if(inputError == ePerspReprojectionErrorWithQuaternions )
    shift_k=7;

  // set cameras constant based on mcon
  for(int i=0; i < mcon; ++i)
    set_const_camera(problem, p, cnp, i, shift_trans, shift_k);

  // set cameras constant based on cconst_mask
  for(int i=0; i < m; ++i){
    if(cconst_mask[i])
      set_const_camera(problem, p, cnp, i, shift_trans, shift_k);
  }

  // set points fixed
  for(int j=0; j < ncon; ++j){
    if(problem.HasParameterBlock(p+cnp*m+pnp*j))
      problem.SetParameterBlockConstant(p+cnp*m+pnp*j);
  }

  // BA for motion or struture only, mot, str , motstr
  switch(ba){
  case BA_MOTSTR:
    break;
  case BA_MOT:
    {
      for(int j=0; j < n; ++j){
        if(problem.HasParameterBlock(p+cnp*m+pnp*j))
          problem.SetParameterBlockConstant(p+cnp*m+pnp*j);}
      break;
    }
  case BA_STR:
    {
      for(int i=0; i < m; ++i)
        set_const_camera(problem, p, cnp, i, shift_trans, shift_k);
      break;
    }
  }

  // Intrinsics mask
  ceres::SubsetParameterization *const_parameterization = NULL;
  std::vector<int> const_intrinsics_index;
  if(in_num > 0){
    for(int in_iter=0;in_iter<in_num;++in_iter)
      if(in_mask[in_iter])
        const_intrinsics_index.push_back(in_iter);
    if(in_num == const_intrinsics_index.size()){
      for(int i=0; i<m; ++i){
        if(problem.HasParameterBlock(p+cnp*i+cnp-in_num))
          problem.SetParameterBlockConstant(p+cnp*i+cnp-in_num);}
    }
    else{
      const_parameterization = new ceres::SubsetParameterization(in_num,const_intrinsics_index);
      for(int i=0; i<m; ++i){
        if(problem.HasParameterBlock(p+cnp*i+cnp-in_num))
          problem.SetParameterization(p+cnp*i+cnp-in_num,const_parameterization);}
    }
  }

}

void _cs_ba_python(double* pOut, int pOutn,
                   int n, int ncon, int m, int mcon,
                   double* vmask, int vmaskn,
                   double* p0, int p0n,
                   int cnp, int pnp,
                   double* x, int xn,
                   char* error_type,
                   char* sfmCase,
                   int verbose,
                   int knum,
                   double* kmask, int kmaskn,
                   int prior_model,
                   double* prior_para, int prior_paran,
                   double* mprior_mask, int mprior_maskn,
                   double* cconst_mask, int cconst_maskn)
{

  errorType inputError = mapError(std::string(error_type));
  baType ba = mapBA(std::string(sfmCase));

  ceres::Problem problem;
  ceres::CostFunction* cost_function;

  switch(inputError)
    {
    case eOrthoReprojectionErrorFull:
      {

        int observation_id = 0;

        for (int j = 0; j < n; ++j) {

          for (int i = 0; i < m; ++i) {

            if(vmask[j*m+i])
              {
                cost_function =
                  OrthoReprojectionErrorFull::Create(x[2 *observation_id + 0],
                                                     x[2 *observation_id + 1]);

                problem.AddResidualBlock(cost_function,
                                         NULL /* squared loss */,
                                         &pOut[i*cnp],
                                         &pOut[m*cnp+j*pnp]);

                observation_id++;

              }

          }

        }

        break;

      }

    case eOrthoReprojectionError:
      {

        int observation_id = 0;

        for (int j = 0; j < n; ++j) {

          for (int i = 0; i < m; ++i) {

            if(vmask[j*m+i])
              {

                cost_function =
                  OrthoReprojectionError::Create(x[2 *observation_id + 0],
                                                 x[2 *observation_id + 1]);

                problem.AddResidualBlock(cost_function,
                                         NULL /* squared loss */,
                                         &pOut[i*cnp],
                                         &pOut[i*cnp+3],
                                         &pOut[m*cnp+j*pnp]);

                observation_id++;

              }

          }

        }

        break;

      }

    case eOrthoReprojectionErrorWithQuaternions:
      {
        int observation_id = 0;

        ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization;

        for (int j = 0; j < n; ++j) {

          for (int i = 0; i < m; ++i) {

            if(vmask[j*m+i])
              {

                cost_function =
                  OrthoReprojectionErrorWithQuaternions::Create(x[2 *observation_id + 0],
                                                                x[2 *observation_id + 1]);

                problem.AddResidualBlock(cost_function,
                                         NULL /* squared loss */,
                                         &pOut[i*cnp],
                                         &pOut[i*cnp+4],
                                         &pOut[m*cnp+j*pnp]);

                problem.AddParameterBlock(&pOut[i*cnp], 4, local_parameterization);

                observation_id++;

              }
          }
        }

        break;

      }

    case ePerspReprojectionErrorFull:
      {
        int observation_id = 0;

        for (int j = 0; j < n; ++j) {

          for (int i = 0; i < m; ++i) {

            if(vmask[j*m+i])
              {

                // PerspReprojectionErrorFull
                cost_function =
                  PerspReprojectionErrorFull::Create(x[2 *observation_id + 0],
                                                     x[2 *observation_id + 1]);

                problem.AddResidualBlock(cost_function,
                                         NULL /* squared loss */,
                                         &pOut[i*cnp],
                                         &pOut[m*cnp+j*pnp]);

                // double cost = 0.0;
                // std::vector<double> residuals;
                // std::vector<double> gradient;
                // problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, &residuals, &gradient, NULL);
                // int testsize = residuals.size();
                // double test = 1.0;

                observation_id++;

              }
          }
        }
        break;

      }

    case eNormalizedPerspReprojectionError:
      {

        int observation_id = 0;

        for (int j = 0; j < n; ++j) {

          for (int i = 0; i < m; ++i) {

            if(vmask[j*m+i])
              {

                cost_function =
                  NormalizedPerspReprojectionError::Create(x[2 *observation_id + 0],
                                                           x[2 *observation_id + 1]);

                problem.AddResidualBlock(cost_function,
                                         NULL /* squared loss */,
                                         &pOut[i*cnp],
                                         &pOut[i*cnp+3],
                                         &pOut[m*cnp+j*pnp]);

                observation_id++;


              }
          }
        }

        break;

      }

    case eNormalizedPerspReprojectionErrorWithQuaternions:
      {

        int observation_id = 0;
        ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization;

        for (int j = 0; j < n; ++j) {

          for (int i = 0; i < m; ++i) {

            if(vmask[j*m+i])
              {

                cost_function =
                  NormalizedPerspReprojectionErrorWithQuaternions::Create(x[2 *observation_id + 0],
                                                                          x[2 *observation_id + 1]);

                problem.AddResidualBlock(cost_function,
                                         NULL /* squared loss */,
                                         &pOut[i*cnp],
                                         &pOut[i*cnp+4],
                                         &pOut[m*cnp+j*pnp]);

                problem.AddParameterBlock(&pOut[i*cnp], 4, local_parameterization);

                observation_id++;

              }
          }
        }

        break;

      }

    case ePerspReprojectionErrorFixedK:
      {
        int observation_id = 0;

        for (int j = 0; j < n; ++j) {

          for (int i = 0; i < m; ++i) {

            if(vmask[j*m+i])
              {

                cost_function =
                  PerspReprojectionErrorFixedK::Create(x[2 *observation_id + 0],
                                                       x[2 *observation_id + 1],
                                                       &pOut[i*cnp+6]);
                problem.AddResidualBlock(cost_function,
                                         NULL /* squared loss */,
                                         &pOut[i*cnp],
                                         &pOut[i*cnp+3],
                                         &pOut[m*cnp+j*pnp]);

                observation_id++;

              }
          }
        }

        break;

      }

    case ePerspReprojectionErrorFixedKWithQuaternions:
      {

        int observation_id = 0;

        ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization;

        for (int j = 0; j < n; ++j) {

          for (int i = 0; i < m; ++i) {

            if(vmask[j*m+i])
              {

                cost_function =
                  PerspReprojectionErrorFixedKWithQuaternions::Create(x[2 *observation_id + 0],
                                                                      x[2 *observation_id + 1],
                                                                      &pOut[i*cnp+7]);
                problem.AddResidualBlock(cost_function,
                                         NULL /* squared loss */,
                                         &pOut[i*cnp],
                                         &pOut[i*cnp+4],
                                         &pOut[m*cnp+j*pnp]);

                problem.AddParameterBlock(&pOut[i*cnp], 4, local_parameterization);

                observation_id++;

              }
          }
        }

        break;

      }

    case ePerspReprojectionError:
      {
        int observation_id = 0;

        for (int j = 0; j < n; ++j) {

          for (int i = 0; i < m; ++i) {

            if(vmask[j*m+i])
              {

                cost_function =
                  PerspReprojectionError::Create(x[2 *observation_id + 0],
                                                 x[2 *observation_id + 1]);
                problem.AddResidualBlock(cost_function,
                                         NULL /* squared loss */,
                                         &pOut[i*cnp+6],
                                         &pOut[i*cnp],
                                         &pOut[i*cnp+3],
                                         &pOut[m*cnp+j*pnp]);

                observation_id++;

              }
          }
        }

        break;

      }

    case ePerspReprojectionErrorWithQuaternions:
      {
        int observation_id = 0;

        ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization;

        for (int j = 0; j < n; ++j) {

          for (int i = 0; i < m; ++i) {

            if(vmask[j*m+i])
              {

                cost_function =
                  PerspReprojectionErrorWithQuaternions::Create(x[2 *observation_id + 0],
                                                                x[2 *observation_id + 1]);
                problem.AddResidualBlock(cost_function,
                                         NULL /* squared loss */,
                                         &pOut[i*cnp+7],
                                         &pOut[i*cnp],
                                         &pOut[i*cnp+4],
                                         &pOut[m*cnp+j*pnp]);

                problem.AddParameterBlock(&pOut[i*cnp], 4, local_parameterization);

                observation_id++;

              }
          }
        }

        break;
      }

    default:
      std::cout<<"wrong input errorType"<<std::endl;
    }

  // add orthographic prior
  if(prior_model > -1 && prior_model < 3)
    add_ortho_prior(problem, pOut, cnp, pnp, m, n,
                    prior_model, prior_para, prior_paran, mprior_mask);

  // add perspective prior
  double* depth_mean = new double[m];
  if(prior_model >= 3)
    add_persp_prior(problem, pOut, cnp, pnp, m, n,
                    prior_para, inputError, depth_mean, mprior_mask);

  // add mask
  add_const_mask(problem, pOut, cnp, pnp, mcon, ncon, m, n,
                 knum, kmask, inputError, ba, cconst_mask);

  // Make Ceres automatically detect the bundle structure. Note that the
  // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
  // for standard bundle adjustment problems.
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  if(verbose)
    options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  if(verbose)
    std::cout << summary.FullReport() << "\n";

  delete[] depth_mean;

}
