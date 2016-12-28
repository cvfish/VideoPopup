#include"class.cpp"
#include "mex.h"
//include<cstdio>

void mexFunction (int nlhs, mxArray *out[],
                  int nrhs, const mxArray *in[]) {
  //        clog<<"Code begins"<<endl;
  // Check to see if we have the correct number of input and output
  // arguments.
  if (nrhs != 10)
    mexErrMsgTxt("Correct form is (cost, overlapping neigbourhood, pairwise neighbourhood, pairwise costs, interior labels, weight of exterior points, cost of models,threshold,ppthreshold,overlap_penalty]).");
  if (nlhs != 3)
    mexErrMsgTxt("Incorrect number of output arguments should be of the form [Binary matrix,interior labels, outliers].");
  int points = mxGetM(in[0]);
  int hyp   = mxGetN(in[0]);
  int p2 = mxGetN(in[1]);
  int maxn   = mxGetM(in[1]);
  int p2p = mxGetN(in[2]);
  int maxnp   = mxGetM(in[2]);
  int cp2p = mxGetN(in[3]);
  int cmaxnp   = mxGetM(in[3]);
  double lambda=mxGetPr(in[5])[0];
  double* hweight=mxGetPr(in[6]);

  double thresh_num = mxGetN(in[7]);
  double* thresh = mxGetPr(in[7]);

  double *ppthresh = NULL;
  //  if(nrhs == 9)
  ppthresh = mxGetPr(in[8]);

  double overlap_penalty= mxGetPr(in[9])[0];
  cout<<"overlap_penalty is "<<overlap_penalty<<endl;
  double* pweight=mxGetPr(in[3]);

  mexPrintf("What do I think is going on? \n points %d, p2 %d, p2p %d cp2p %d\n", points,p2,p2p,cp2p);
  mexPrintf("Neighbourhoods: maxn %d, maxnp %d, cmaxnp %d\n",maxn,maxnp,cmaxnp);
  if((lambda>1)||!(lambda>=0)){
    //clog<<"Lambda: "<<lambda<<endl;
    mexErrMsgTxt("Lambda must be in [0,1]");
  }
  ////clog<<"hyp "<<hyp<<endl;
  ////clog<<"points "<<points<<endl;
  ////clog<<"maxn "<<maxn<<endl;
  if (points!=p2){
    //cout<<"points "<<points<<endl;
    //cout<<"p2 "<<p2<<endl;
    mexErrMsgTxt("Matrixs 1 and 2 must be of same length.");
  }
  if (points!=p2p){
    //cout<<"points "<<points<<endl;
    //cout<<"p2 "<<p2<<endl;
    mexErrMsgTxt("Matrixs 1 and 3 must be of same length.");
  }
  if (cp2p!=p2p){
    mexErrMsgTxt("Matrixs 3 and 4 must be the same size.");
  }
  if (maxnp!=cmaxnp){
    mexErrMsgTxt("Matrixs 3 and 4 must be the same size.");
  }

  if(maxn>=points)
    mexErrMsgTxt("Matrix 2 has too many neighbours (no neigh must always be larger than points)");

  if(maxnp>=points)
    mexErrMsgTxt("Matrix 3 has too many neighbours (no neigh must always be larger than points)");

  if (points != static_cast<int> (mxGetM(in[4]))){
    //cout<<"points "<<points<<endl;
    //cout<<"mxGetM(in[2]) "<<mxGetM(in[2])<<endl;
    mexErrMsgTxt("Matrix 1, and vector 4 must be of same length.");
  }
  if (mxGetN(in[4])!=1)
    mexErrMsgTxt("Third element must be a vector.");

  if ((mxGetN(in[5])!=1)||(mxGetM(in[5])!=1))
    mexErrMsgTxt("Sixth element must be a scalar.");

  if (mxGetN(in[6])!=1)
    mexErrMsgTxt("7th element must be a vector.");
  if (static_cast<int>(mxGetM(in[6]))!=hyp)
    mexErrMsgTxt("7th element must be a vector of length hyp.");

  for (int i = 0; i !=hyp; ++i)
    if (!(hweight[i]>=0)){
      mexErrMsgTxt("matrix 7 must be elementwise >= 0");
    }
  for (int i = 0; i !=hyp*maxnp; ++i)
    if (!(pweight[i]>=0)){
      mexErrMsgTxt("Matrix 3 must be elementwise >= 0");
    }

  // hypothesis H(points,maxn,hyp,lambda,hweight,maxnp);
  hypothesis H(points,maxn,hyp,lambda,hweight,maxnp,ppthresh);
  if (ppthresh && (mxGetM(in[8])>1) && (mxGetN(in[8])>1)){
    mexPrintf("New code path!\n");
    if(mxGetM(in[8])!=mxGetM(in[1]))
      mexErrMsgTxt("Matrixs 2 and 9 must be the same size.");
    if (mxGetN(in[8])!=mxGetN(in[1]))
      mexErrMsgTxt("Matrixs 2 and 9 must be the same size.");
    H.thresh=0;
    H._mcost=ppthresh;
  }
  H.overlap_penalty=overlap_penalty;
  if (ppthresh)
    for (int i = 0; i !=mxGetM(in[8])*mxGetN(in[8]); ++i)
      if (!(ppthresh[i]>=0))
        mexErrMsgTxt("matrix 9 must be elementwise >= 0");


  // H.un= mxGetPr(in[0]);
  // for (int i = 0; i !=points*hyp; ++i)
  //   if(H.un[i]<0){
  //     mexErrMsgTxt("matrix 1 must be elementwise >= 0");
  //   }
  //   else if(H.un[i]>thresh)
  //     H.un[i]=thresh;

  H.un= mxGetPr(in[0]);
  for(int i = 0; i< points;++i)
  {
     for(int j=0; j< hyp; ++j)
     {
       if(!(H.un[points*j+i]>=0))
        {
          mexErrMsgTxt("matrix 1 must be elementwise >= 0");
        }
        else if(thresh_num == points && H.un[points*j+i]> thresh[i])
        {
          H.un[points*j+i] = thresh[i];
        }
        else if(thresh_num == 1 && H.un[points*j+i]> thresh[0])
          H.un[points*j+i] = thresh[0];

     }
  }

  for (int i = 0; i !=points;++i){
    H.label[i]=mxGetPr(in[4])[i];
    if (H.label[i]>=hyp){
      mexErrMsgTxt("All labels must be less than number of hypothesis");
    }
    if (!(H.label[i]>=0))
      mexErrMsgTxt("All labels must be greater than or equal to number 0");
  }

  for (int i = 0; i !=points;++i)
    for (int j  = 0; j !=maxn; ++j){
      H.neigh_m[i*maxn+j]=mxGetPr(in[1])[i*maxn+j]-1;
      if (H.neigh_m[i*maxn+j]>=points)
        mexErrMsgTxt("overlap neighbours (matrix 2) must be less  or equal to than number of points");
      if (H.neigh_m[i*maxn+j]<-1)
        mexErrMsgTxt("overlap neighbours (matrix 2) must be greater than 0");
      if(H.neigh_m[i*maxn+j]==i){
          mexPrintf("Matrix 2 %d has a self neighbour in position %d\n",i+1,j+1);
          mexErrMsgTxt("No self neighbours");
      }
    }

  for (int i = 0; i !=points;++i)
    for (int j  = 0; j !=maxnp; ++j){
      H.neigh_p[i*maxnp+j]=mxGetPr(in[2])[i*maxnp+j]-1;
      if (H.neigh_p[i*maxnp+j]>=points)
        mexErrMsgTxt("pairwise neighbours (matrix 3) must be less  or equal to than number of points");
      if (H.neigh_p[i*maxnp+j]<-1){
        char a[80];
        sprintf(a,"points: %d H.neigh_p[%d]=%d i:%d maxnp:%d j:%d",points,i*maxnp+j,H.neigh_p[i*maxnp+j],i,maxnp,j);
        mexErrMsgTxt(a);
        mexErrMsgTxt("pairwise neighbours (matrix 3) must be greater than or equal to 0");
        if(H.neigh_p[i*maxn+j]==i){
          mexPrintf("Matrix 3 %d has a self neighbour in position %d\n",i+1,j+1);
          mexErrMsgTxt("No self neighbours");
        }
      }
    }

  H._ncost=pweight;
  //  mexPrintf("Calling debug solve \n");
  H.solve();//

  out[0]=mxCreateDoubleMatrix(points,hyp,mxREAL);
  out[2]=mxCreateDoubleMatrix(points,hyp,mxREAL);
  H.annotate((double*)mxGetPr(out[0]));

  for(int i = 0;i<points;++i)
  {
      for(int j = 0;j<hyp;j++)
      {
        if((thresh_num == points && H.un[j*points+i] == thresh[i]) || (thresh_num == 1 && H.un[j*points+i] == thresh[0]))
           mxGetPr(out[0])[j*points+i]=0;
      }
  }

//   for (int i = 0; i !=hyp*points; ++i)
//     if (H.un[i]==thresh)
//       if(mxGetPr(out[0])[i]){
//      mxGetPr(out[0])[i]=0;
//
//      mxGetPr(out[2])[i]=1;
//       }
//

  out[1]=mxCreateDoubleMatrix(points,1,mxREAL);
  for (int i = 0; i !=points; ++i)
    mxGetPr(out[1])[i]=H.label[i];

  return ;
}

