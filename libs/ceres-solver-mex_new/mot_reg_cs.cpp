#include <cmath>
#include <cstdio>
#include <iostream>

#include "mex.h"
#include "rui_reprojection_error.h"


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  int nlab,nf,nb,nref,verbose;
  double *p_rigmots_init,*vvt,*vvdt,*u,*ud,*para;
  double *p_rigmots;

  //default value
  verbose = 0;
  nref = 1;

  //parse input args, checking input parameters

  /** nlab **/
  /* the first argument must be a scalar */
  if(!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) || mxGetM(prhs[0])!=1 || mxGetN(prhs[0])!=1)
    mexErrMsgTxt("mot_reg_cs: the first argument nlab must be a scalar.");
  nlab=(int)mxGetScalar(prhs[0]);

  /** nf **/
  /* the second argument must be a scalar */
  if(!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) || mxGetM(prhs[1])!=1 || mxGetN(prhs[1])!=1)
    mexErrMsgTxt("mot_reg_cs: the second argument nf must be a scalar.");
  nf=(int)mxGetScalar(prhs[1]);

  /** nb **/
  /* the third argument must be a scalar */
  if(!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) || mxGetM(prhs[1])!=1 || mxGetN(prhs[1])!=1)
    mexErrMsgTxt("mot_reg_cs: the third argument nb must be a scalar.");
  nb=(int)mxGetScalar(prhs[2]);

  /** p_rigmots_init **/
  /* the fourth argument must be a 6xnf*nlab matrix */
  if(!mxIsDouble(prhs[3]) || mxIsComplex(prhs[3]) || mxGetM(prhs[3])!=6 || mxGetN(prhs[3])!=nf*nlab)
    mexErrMsgTxt("mot_reg_cs: the fourth argument p_rigmots_init must be a 6xnf*nlab matrix.");
  p_rigmots_init=mxGetPr(prhs[3]);

  p_rigmots = new double[6*nf*nlab];
  memcpy(p_rigmots,p_rigmots_init,sizeof(double)*6*nf*nlab);

  /** vvt **/
  /* the fifth argument must be a 16*nlab matrix */
  if(!mxIsDouble(prhs[4]) || mxIsComplex(prhs[4]) || mxGetM(prhs[4])!=16 || mxGetN(prhs[4])!=nlab)
    mexErrMsgTxt("mot_reg_cs: the fifth argument ud must be a 16*nlab matrix.");
  vvt=mxGetPr(prhs[4]);

  /** vvdt **/
  /* the sixth argument must be a 18*nb matrix */
  if(!mxIsDouble(prhs[5]) || mxIsComplex(prhs[5]) || mxGetM(prhs[5])!=18 || mxGetN(prhs[5])!=nb)
    mexErrMsgTxt("mot_reg_cs: the sixth argument ud must be a 18*nb matrix.");
  vvdt=mxGetPr(prhs[5]);

  /** u **/
  /* the seventh argument must be a 16*nlab matrix */
  if(!mxIsDouble(prhs[6]) || mxIsComplex(prhs[6]) || mxGetM(prhs[6])!=16 || mxGetN(prhs[6])!=nlab)
    mexErrMsgTxt("mot_reg_cs: the seventh argument ud must be a 16*nlab matrix.");
  u=mxGetPr(prhs[6]);

  /** ud **/
  /* the eighth argument must be a 18*nb matrix */
  if(!mxIsDouble(prhs[7]) || mxIsComplex(prhs[7]) || mxGetM(prhs[7])!=18 || mxGetN(prhs[7])!=nb)
    mexErrMsgTxt("mot_reg_cs: the eighth argument ud must be a 18*nb matrix.");
  ud=mxGetPr(prhs[7]);

  /** para **/
  /* the ninth argument must be a 3*1 matrix */
  if(!mxIsDouble(prhs[8]) || mxIsComplex(prhs[8]) || mxGetM(prhs[8])!=3 || mxGetN(prhs[8])!=1)
    mexErrMsgTxt("mot_reg_cs: the ninth argument ud must be a 3*1 matrix.");
  para=mxGetPr(prhs[8]);

  if(nrhs > 9){
    if(!mxIsDouble(prhs[9]) || mxIsComplex(prhs[9]))
      mexErrMsgTxt("mot_reg_cs: nref must be a scalar.");
    nref = (int)mxGetScalar(prhs[9]);
  }

  if(nrhs > 10){
    if(!mxIsDouble(prhs[10]) || mxIsComplex(prhs[10]))
      mexErrMsgTxt("mot_reg_cs: verbose level must be a scalar.");
    verbose = (int)mxGetScalar(prhs[10]);
  }

  ceres::Problem problem;
  ceres::CostFunction* cost_function;

  for(int i=0; i<nlab; ++i)
    {
      for(int j=0; j<nf; ++j)
        {

	  if(nref-1!=j)
	    {
	      cost_function =
		MotionDiffWithPointv2::Create(para[0],vvt+16*i,u+16*i);
	      problem.AddResidualBlock(cost_function,
				       NULL /* squared loss */,
				       &p_rigmots[6*(nf*i+j)],
				       &p_rigmots[6*(nf*i+j)+3],
				       &p_rigmots_init[6*(nf*i+j)],
				       &p_rigmots_init[6*(nf*i+j)+3]);

	      // Set the block of initial value as constant
	      problem.SetParameterBlockConstant(&p_rigmots_init[6*(nf*i+j)]);
	      problem.SetParameterBlockConstant(&p_rigmots_init[6*(nf*i+j)+3]);
	    }
	  // double cost = 0.0;
	  // std::vector<double> residuals;
	  // std::vector<double> gradient;
	  // problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, &residuals, &gradient, NULL);
	  // int testsize = residuals.size();
	  // double test = 1.0;

          if(j>0){
            cost_function =
              MotionDiffWithPointv2::Create(para[1],vvt+16*i,u+16*i);
            problem.AddResidualBlock(cost_function,
                                     NULL /* squared loss */,
                                     &p_rigmots[6*(nf*i+j)],
                                     &p_rigmots[6*(nf*i+j)+3],
                                     &p_rigmots[6*(nf*i+j-1)],
                                     &p_rigmots[6*(nf*i+j-1)+3]);

	  // double cost = 0.0;
	  // std::vector<double> residuals;
	  // std::vector<double> gradient;
	  // problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, &residuals, &gradient, NULL);
	  // int testsize = residuals.size();
	  // double test = 1.0;

          }
        }
    }

  for(int k=0; k<nb; ++k)
    {
      int i = vvdt[k*18];
      int i2 = vvdt[k*18+1];

      for(int j=0; j<nf; ++j)
        {
          cost_function =
            MotionDiffWithPointv2::Create(para[2],vvdt+18*k+2,ud+18*k+2);
          problem.AddResidualBlock(cost_function,
                                   NULL /* squared loss */,
                                   &p_rigmots[6*(nf*(i-1)+j)],
                                   &p_rigmots[6*(nf*(i-1)+j)+3],
                                   &p_rigmots[6*(nf*(i2-1)+j)],
  				   &p_rigmots[6*(nf*(i2-1)+j)+3]);

        }
    }

  //Set parameters corresponding to reference frame as constant
  for(int i=0; i<nlab; ++i)
    {
      problem.SetParameterBlockConstant(&p_rigmots[6*(nf*i+nref-1)]);
      problem.SetParameterBlockConstant(&p_rigmots[6*(nf*i+nref-1)+3]);
    }

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

  plhs[0] = mxCreateDoubleMatrix(6,nf*nlab, mxREAL);
  memcpy(mxGetPr(plhs[0]),p_rigmots,sizeof(double)*6*nf*nlab);
  delete[] p_rigmots;

}

