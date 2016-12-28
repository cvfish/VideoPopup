#include"class.cpp"
#include "mex.h"
#include<cstdio>
void mexFunction (int nlhs, mxArray *out[],
                  int nrhs, const mxArray *in[]) {
  if (nrhs != 6)
    mexErrMsgTxt("Incorrect number of input arguments.\n Correct form is (1:unary cost, 2:neigbourhood, 3:neighbour costs, 4:current labels, 5:cost of model appearence, 6:outlier rejection).");
  if (nlhs != 2)
    mexErrMsgTxt("Incorrect number of output arguments should be of the form [Binary matrix, labels].");

  // mexPrintf("Are we dead yet?\n");
  int points = mxGetM(in[0]);
  int hyp   = mxGetN(in[0]);
  int p2 = mxGetN(in[1]);
  int maxn   = mxGetM(in[1]);
  double thresh=mxGetPr(in[5])[0];

  double* pweight=mxGetPr(in[2]);
  double* hweight=mxGetPr(in[4]);

  //  mexPrintf("Are we dead now?\n");
  ////clog<<"hyp "<<hyp<<endl;
  ////clog<<"points "<<points<<endl;
  ////clog<<"maxn "<<maxn<<endl;
  if (points!=p2){
    //cout<<"points "<<points<<endl;
    //cout<<"p2 "<<p2<<endl;
    mexErrMsgTxt("Matrixs 1 and 2 must be of same length.");
  }

  if ((mxGetN(in[2])!=p2)||mxGetM(in[2])!=maxn){
    mexErrMsgTxt("second and third arguements must be matrices of same dimension");
  }

  if(points !=mxGetN(in[3]))
    mexErrMsgTxt("Matrix 1, and vector 4 must be of same length.");

  if(1 !=mxGetM(in[3]))
    mexErrMsgTxt("Matrix 4 must be a vector.");
  for (unsigned int i = 0; i !=hyp; ++i)
    if (hweight[i]<0){
      //clog<<"i: "<<i<<"hweight[i]: "<<hweight[i]<<endl;
      mexErrMsgTxt("hweight[i] must be >= 0");
    }
  if (hyp != mxGetM(in[4])){
    //clog<<"hyp "<<hyp<<endl;
    //clog<<"mxGetM(in[4]) "<<mxGetM(in[4])<<endl;
    mexErrMsgTxt("Matrix 1^T, and vector 5 must be of same length.");
  }
  printf("Done tests\n");
  hypothesis H(points,0,hyp,0,hweight,maxn);

  H.un= mxGetPr(in[0]);
  for (unsigned int i = 0; i !=points*hyp; ++i)
    if(H.un[i]<0){
      //clog<<"i: "<<i<<"H.un[i]: "<<H.un[i]<<endl;
      mexErrMsgTxt("hweight[i] must be >= 0");
    }
  H._ncost=pweight;

  for (unsigned int i = 0; i !=points;++i)
    H.label[i]=mxGetPr(in[3])[i];
  for(unsigned int i=0;i!=points;++i){
    if (H.label[i]>=hyp)
      mexErrMsgTxt("All labels must be less than number of hypothesis");
    if (H.label[i]<0)
      mexErrMsgTxt("All labels must be greater than number 0");
  }

  //H.neigh= mxGetPr(in[1]);
  for (unsigned int i = 0; i !=points;++i)
    for (unsigned int j  = 0; j !=maxn; ++j){
      H.neigh_p[i*maxn+j]=mxGetPr(in[1])[i*maxn+j]-1;
      if (H.neigh_p[i*maxn+j]>=points)
        mexErrMsgTxt("neighbours must be less  or equal to than number of points");
      if (H.neigh_p[i*maxn+j]<-1)
        mexErrMsgTxt("neighbours must be greater than 0");
      if(H.neigh_p[i*maxn+j]==i){
        mexPrintf("%d has a self neighbour in position %d\n",i+1,j+1);
        mexErrMsgTxt("No self neighbours");
      }
      // //clog<<H.neigh[i*maxn+j]<<endl;
    }
  H.solve();

  out[0]=mxCreateDoubleMatrix(points,hyp,mxREAL);
  H.annotate((double*)mxGetPr(out[0]));
  for (int i = 0; i !=hyp*points; ++i)
    if (H.un[i]==thresh)
      mxGetPr(out[0])[i]=0;


  out[1]=mxCreateDoubleMatrix(points,1,mxREAL);
  for (int i = 0; i !=points; ++i)
    mxGetPr(out[1])[i]=H.label[i];


  return ;
}
