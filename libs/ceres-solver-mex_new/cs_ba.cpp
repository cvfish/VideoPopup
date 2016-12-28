// Matlab MEX file for ceres-solver

#include <cmath>
#include <cstdio>
#include <iostream>

#include "mex.h"
#include "rui_reprojection_error.h"

void add_ortho_prior(ceres::Problem &problem,double *p, int cnp, int pnp, int m, int n,
                     int prior_model, double *prior_para, const mxArray **prhs, double *mprior_mask)
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
    if(mxGetM(prhs[15]) == 3)
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
                     int prior_model, double *prior_para, errorType &inputError, double *depth_mean, double *mprior_mask)
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


        // double cost = 0.0;
        // std::vector<double> residuals;
        // std::vector<double> gradient;
        // problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, &residuals, &gradient, NULL);
        // int testsize = residuals.size();
        // double test = 1.0;

      }
    }
  }
  // // scale invariant relief prior, quaternion parameterization
  // // for each frame
  // for(int i=0; i<m; ++i){
  //   vector<double *> motshape;
  //   motshape.push_back(&p[i*cnp]);
  //   for(int j=0; j<n; ++j){
  // 	motshape.push_back(&p[m*cnp+j*pnp]);
  //   }
  // }
  // cost_function = DepthReliefPrior::Create(n,cnp,pnp,alpha_relief);
  // // See the public problem.h file for description of these methods.
  // problem.AddResidualBlock(cost_function,
  // 			     NULL,
  // 			     motshape);

  // scale invariant relief prior, set mean as a free parameter to decouple points


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

          // double cost = 0.0;
          // std::vector<double> residuals;
          // std::vector<double> gradient;
          // problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, &residuals, &gradient, NULL);
          // int testsize = residuals.size();
          // double test = 1.0;

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

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  int n, ncon, m, mcon, cnp, pnp, mnp, nvars, nprojs, minnvars, len, status;
  double *p0,*x,*p;
  double *vmask, *in_mask, *prior_para;
  int in_num,verbose, prior_model;

  // used for fast relief regularization
  double *depth_mean;

  // motion smoothness prior mask
  double *mprior_mask;

  // const camera mask
  double *cconst_mask;

  // default value, no output
  verbose = 0; in_num = 0; in_mask = NULL;
  prior_model = -1; prior_para = NULL;


  //parse input args, checking input parameters

  /** n **/
  /* the first argument must be a scalar */
  if(!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) || mxGetM(prhs[0])!=1 || mxGetN(prhs[0])!=1)
    mexErrMsgTxt("cs_ba: n must be a scalar.");
  n=(int)mxGetScalar(prhs[0]);

  /** ncon **/
  /* the second argument must be a scalar */
  if(!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) || mxGetM(prhs[1])!=1 || mxGetN(prhs[1])!=1)
    mexErrMsgTxt("cs_ba: ncon must be a scalar.");
  ncon=(int)mxGetScalar(prhs[1]);

  /** m **/
  /* the third argument must be a scalar */
  if(!mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]) || mxGetM(prhs[2])!=1 || mxGetN(prhs[2])!=1)
    mexErrMsgTxt("cs_ba: m must be a scalar.");
  m=(int)mxGetScalar(prhs[2]);

  // depth_mean initialization
  depth_mean = new double[m];

  /** mcon **/
  /* the fourth argument must be a scalar */
  if(!mxIsDouble(prhs[3]) || mxIsComplex(prhs[3]) || mxGetM(prhs[3])!=1 || mxGetN(prhs[3])!=1)
    mexErrMsgTxt("cs_ba: mcon must be a scalar.");
  mcon=(int)mxGetScalar(prhs[3]);

  /** mask **/
  /* the fifth argument must be a mxn matrix */
  //std::cout<<mxGetM(prhs[4])<<" "<<mxGetN(prhs[4])<<std::endl;
  if(!mxIsDouble(prhs[4]) || mxIsComplex(prhs[4]) || mxGetM(prhs[4])!=m || mxGetN(prhs[4])!=n)
    mexErrMsgTxt("cs_ba: mask must be a mxn matrix.");
  // if(mxIsSparse(prhs[4])) vmask=getVMaskSparse(prhs[4]);
  // else vmask=getVMaskDense(prhs[4]);
  vmask=mxGetPr(prhs[4]);

  /** p **/
  /* the sixth argument must be a vector */
  if(!mxIsDouble(prhs[5]) || mxIsComplex(prhs[5]) || !(mxGetM(prhs[5])==1 || mxGetN(prhs[5])==1))
    mexErrMsgTxt("cs_ba: p must be a real vector.");
  p0=mxGetPr(prhs[5]);
  p = new double[mxGetM(prhs[5])*mxGetN(prhs[5])];
  memcpy(p,p0,sizeof(double)*mxGetM(prhs[5])*mxGetN(prhs[5]));

  /** cnp **/
  /* the seventh argument must be a scalar */
  if(!mxIsDouble(prhs[6]) || mxIsComplex(prhs[6]) || mxGetM(prhs[6])!=1 || mxGetN(prhs[6])!=1)
    mexErrMsgTxt("cs_ba: cnp must be a scalar.");
  cnp=(int)mxGetScalar(prhs[6]);

  /** pnp **/
  /* the eighth argument must be a scalar */
  if(!mxIsDouble(prhs[7]) || mxIsComplex(prhs[7]) || mxGetM(prhs[7])!=1 || mxGetN(prhs[7])!=1)
    mexErrMsgTxt("cs_ba: pnp must be a scalar.");
  pnp=(int)mxGetScalar(prhs[7]);

  /** x **/
  /* the ninth argument must be a vector */
  if(!mxIsDouble(prhs[8]) || mxIsComplex(prhs[8]) || !(mxGetM(prhs[8])==1 || mxGetN(prhs[8])==1))
    mexErrMsgTxt("cs_ba: x must be a real vector.");
  x=mxGetPr(prhs[8]);

  char *refhowto;

  /* examine supplied name */
  len=mxGetN(prhs[9])+1;
  refhowto=(char*) mxCalloc(len, sizeof(char));
  status=mxGetString(prhs[9], refhowto, len);
  if(status!=0)
    mexErrMsgTxt("cs_ba: not enough space. String is truncated.");
  errorType inputError = mapError(std::string(refhowto));

  /* examine supplied name */
  len=mxGetN(prhs[10])+1;
  refhowto=(char*) mxCalloc(len, sizeof(char));
  status=mxGetString(prhs[10], refhowto, len);
  if(status!=0)
    mexErrMsgTxt("cs_ba: not enough space. String is truncated.");
  baType ba = mapBA(std::string(refhowto));


  if(nrhs > 11){
    if(!mxIsDouble(prhs[11]) || mxIsComplex(prhs[11]))
      mexErrMsgTxt("cs_ba: verbose level must be a scalar.");
    verbose = (int)mxGetScalar(prhs[11]);
  }

  /** intrinsic parameter number **/
  /* the ninth argument must be a vector */
  if(nrhs > 13){

    if(!mxIsDouble(prhs[12]) || mxIsComplex(prhs[12]) || mxGetM(prhs[12])!=1 || mxGetN(prhs[12])!=1)
      mexErrMsgTxt("cs_ba: intrinsic number must be a scalar.");
    in_num = (int)mxGetScalar(prhs[12]);

    if(!mxIsDouble(prhs[13]) || mxIsComplex(prhs[13]))
      mexErrMsgTxt("cs_ba: intrinsic mask must be a real vector.");
    in_mask = mxGetPr(prhs[13]);
  }

  if(nrhs > 15){

    if(!mxIsDouble(prhs[14]) || mxIsComplex(prhs[14]) || mxGetM(prhs[14])!=1 || mxGetN(prhs[14])!=1)
      mexErrMsgTxt("cs_ba: prior_model must be a scalar.");
    prior_model = (int)mxGetScalar(prhs[14]);

    if(!mxIsDouble(prhs[15]) || mxIsComplex(prhs[15]) || mxGetN(prhs[15])!=1)
      mexErrMsgTxt("cs_ba: prior_para must be a real vector.");
    prior_para = mxGetPr(prhs[15]);

  }

  // default value, use motion prior for every frame
  if(nrhs > 16){
    if(!mxIsDouble(prhs[16]) || mxIsComplex(prhs[16]) || mxGetN(prhs[16])!=1)
      mexErrMsgTxt("cs_ba: motion prior mask must be a real vector.");
    mprior_mask = mxGetPr(prhs[16]);}
  else{
    mprior_mask = new double[m-1];
    for(int i = 0; i < m-1; i++)
      mprior_mask[i] = 1;
  }

  // default value, all cameras will be optimized
  if(nrhs > 17){
    if(!mxIsDouble(prhs[17]) || mxIsComplex(prhs[17]) || mxGetN(prhs[17])!=1 || mxGetM(prhs[17]) != m)
      mexErrMsgTxt("cs_ba: camera const mask must be a real mx1 vector.");
    cconst_mask = mxGetPr(prhs[17]);}
  else{
    cconst_mask = new double[m];
    for(int i = 0; i < m; i++)
      cconst_mask[i] = 0;
  }

  // visibiltiy number
  int *pVisNum = new int[n];
  for (int j = 0; j < n; ++j) {
    pVisNum[j] = 0;
    for (int i = 0; i < m; ++i) {
      if(vmask[j*m+i])
        {
          ++pVisNum[j];
        }
    }
    //    pVisNum[j] = sqrt(pVisNum[j]);
  }

  // Create residuals for each observation in the bundle adjustment problem. The
  // parameters for cameras and points are added automatically.

  int observation_id = 0;

  ceres::Problem problem;
  ceres::CostFunction* cost_function;

  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {

      if(vmask[j*m+i])
        {
          switch(inputError){

          case eOrthoReprojectionErrorFull:
            {

              cost_function =
                OrthoReprojectionErrorFull::Create(x[2 *observation_id + 0],
                                                   x[2 *observation_id + 1]);

              problem.AddResidualBlock(cost_function,
                                       NULL /* squared loss */,
                                       &p[i*cnp],
                                       &p[m*cnp+j*pnp]);


              break;

            }

          case eOrthoReprojectionError:
            {
              cost_function =
                OrthoReprojectionError::Create(x[2 *observation_id + 0],
                                               x[2 *observation_id + 1]);

              problem.AddResidualBlock(cost_function,
                                       NULL /* squared loss */,
                                       &p[i*cnp],
                                       &p[i*cnp+3],
                                       &p[m*cnp+j*pnp]);

              break;

            }

          case eOrthoReprojectionErrorWithQuaternions:
            {

              ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization;

              cost_function =
                OrthoReprojectionErrorWithQuaternions::Create(x[2 *observation_id + 0],
                                                              x[2 *observation_id + 1]);

              problem.AddResidualBlock(cost_function,
                                       NULL /* squared loss */,
                                       &p[i*cnp],
                                       &p[i*cnp+4],
                                       &p[m*cnp+j*pnp]);

              problem.AddParameterBlock(&p[i*cnp], 4, local_parameterization);


              break;

            }

          case ePerspReprojectionErrorFull:
            {
              // PerspReprojectionErrorFull
              cost_function =
                PerspReprojectionErrorFull::Create(x[2 *observation_id + 0],
                                                   x[2 *observation_id + 1]);

              problem.AddResidualBlock(cost_function,
                                       NULL /* squared loss */,
                                       &p[i*cnp],
                                       &p[m*cnp+j*pnp]);

              // double cost = 0.0;
              // std::vector<double> residuals;
              // std::vector<double> gradient;
              // problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, &residuals, &gradient, NULL);
              // int testsize = residuals.size();
              // double test = 1.0;

              break;

            }

          case eNormalizedPerspReprojectionError:
            {
              cost_function =
                NormalizedPerspReprojectionError::Create(x[2 *observation_id + 0],
                                                         x[2 *observation_id + 1]);

              problem.AddResidualBlock(cost_function,
                                       NULL /* squared loss */,
                                       &p[i*cnp],
                                       &p[i*cnp+3],
                                       &p[m*cnp+j*pnp]);

              break;

            }

          case eNormalizedPerspReprojectionErrorWithQuaternions:
            {

              ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization;

              cost_function =
                NormalizedPerspReprojectionErrorWithQuaternions::Create(x[2 *observation_id + 0],
                                                                        x[2 *observation_id + 1]);

              problem.AddResidualBlock(cost_function,
                                       NULL /* squared loss */,
                                       &p[i*cnp],
                                       &p[i*cnp+4],
                                       &p[m*cnp+j*pnp]);

              problem.AddParameterBlock(&p[i*cnp], 4, local_parameterization);

              break;

            }

          case ePerspReprojectionErrorFixedK:
            {
              cost_function =
                PerspReprojectionErrorFixedK::Create(x[2 *observation_id + 0],
                                                     x[2 *observation_id + 1],
                                                     &p[i*cnp+6]);
              problem.AddResidualBlock(cost_function,
                                       NULL /* squared loss */,
                                       &p[i*cnp],
                                       &p[i*cnp+3],
                                       &p[m*cnp+j*pnp]);
              break;

            }

          case ePerspReprojectionErrorFixedKWithQuaternions:
            {

              ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization;

              cost_function =
                PerspReprojectionErrorFixedKWithQuaternions::Create(x[2 *observation_id + 0],
                                                                    x[2 *observation_id + 1],
                                                                    &p[i*cnp+7]);
              problem.AddResidualBlock(cost_function,
                                       NULL /* squared loss */,
                                       &p[i*cnp],
                                       &p[i*cnp+4],
                                       &p[m*cnp+j*pnp]);

              problem.AddParameterBlock(&p[i*cnp], 4, local_parameterization);

              break;

            }

          case ePerspReprojectionError:
            {
              cost_function =
                PerspReprojectionError::Create(x[2 *observation_id + 0],
                                               x[2 *observation_id + 1]);
              problem.AddResidualBlock(cost_function,
                                       NULL /* squared loss */,
                                       &p[i*cnp+6],
                                       &p[i*cnp],
                                       &p[i*cnp+3],
                                       &p[m*cnp+j*pnp]);

              break;

            }

          case ePerspReprojectionErrorWithQuaternions:
            {

              ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization;

              cost_function =
                PerspReprojectionErrorWithQuaternions::Create(x[2 *observation_id + 0],
                                                              x[2 *observation_id + 1]);
              problem.AddResidualBlock(cost_function,
                                       NULL /* squared loss */,
                                       &p[i*cnp+7],
                                       &p[i*cnp],
                                       &p[i*cnp+4],
                                       &p[m*cnp+j*pnp]);

              problem.AddParameterBlock(&p[i*cnp], 4, local_parameterization);

              break;
            }

          default:

            std::cout<<"wrong input errorType"<<std::endl;

          }

          observation_id++;

        }

    }

  }

  // add orthographic prior
  if(prior_model > -1 && prior_model < 3)
    add_ortho_prior(problem,p,cnp,pnp,m,n,prior_model,prior_para,prhs,mprior_mask);

  // add perspective prior
  if(prior_model >= 3)
    add_persp_prior(problem,p,cnp,pnp,m,n,prior_model,prior_para,inputError,depth_mean,mprior_mask);

  // add mask
  add_const_mask(problem,p,cnp,pnp,mcon,ncon,m,n,in_num,in_mask,inputError,ba,cconst_mask);

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

  plhs[0] = mxCreateDoubleMatrix(mxGetM(prhs[5]),mxGetN(prhs[5]), mxREAL);
  //plhs[0] = mxCreateDoubleMatrix(mxGetM(p),mxGetN(p), mxREAL);
  memcpy(mxGetPr(plhs[0]),p,sizeof(double)*mxGetM(prhs[5])*mxGetN(prhs[5]));
  delete[] p;

  delete[] pVisNum;
  delete[] depth_mean;
  if(nrhs <= 16)
    delete[] mprior_mask;
  if(nrhs <= 17)
    delete[] cconst_mask;

}
