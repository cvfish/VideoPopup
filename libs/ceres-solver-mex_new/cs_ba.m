function popt = cs_ba(n,ncon,m,mcon,vmask,p0,cnp,pnp,x,error_typt,sfmCase,verbose,knum,kmask,prior_model,prior_para,mprior_mask,cconst_mask)

% Ceres-solver matlab MEX interface

% In the following, the word "vector" is meant to imply either a row or a column vector.
%
% required input arguments:
% - n: number of 3D points
%
% - ncon: number of 3D points whose parameters should not be modified
%
% - m: number of images
%
% - mcon: number of images whose parameters should not be modified
%
% - vmask: nxm matrix specifying the points visible in each camera. It can be either a dense or a sparse matrix;
%      in both cases, a nonzero element at position (i, j) indicates that point i is visible in image j.
%
% - p0: vector of doubles holding the initial parameter estimates laid out as (a1, ..., am, b1, ..., bn),
%      aj, bi being the j-th image and i-th point parameters, respectively
%
% - cnp: number of parameters for each camera
%
% - pnp: number of parameters for each 3D point
%
% - x: vector of doubles holding the measurements (i.e., image projections) laid out as
%      (x_11, .. x_1m, ..., x_n1, .. x_nm), where x_ij is the projection of the i-th point on the j-th image.
%      If point i is not visible in image j, x_ij is missing; see vmask(i, j).
%
% - error_type: cost function model

%         OrthoReprojectionErrorFull,
%         OrthoReprojectionError,
%         OrthoReprojectionErrorWithQuaternions,
%         PerspReprojectionErrorFull,
%         NormalizedPerspReprojectionError,
%         NormalizedPerspReprojectionErrorWithQuaternions,
%         PerspReprojectionErrorFixedK,
%         PerspReprojectionErrorFixedKWithQuaternions,
%         PerspReprojectionError,
%         PerspReprojectionErrorWithQuaternions
%
% - sfmCase: String defining the type of refinement to be carried out. It should be one of the following:
%      'motstr' refinement of motion & structure, default
%      'mot'    refinement of motion only (ncon is redundant in this case)
%      'str'    refinement of structure only (mcon is redundant in this case)
%      If omitted, a default of 'motstr' is assumed. Depending on the minimization type, the MEX
%      interface will invoke one of sba_motstr_levmar(), sba_mot_levmar() or sba_str_levmar()
%
% - verbose: verbosity level
%
% additional optional input parameters

% - knum: number of intrinsic parameters(only used for intrinsic mask)

% - kmask: intrinsic mask

% - prior_model: what prior to use

  % prior_model : flag of used model, -1 for nothing, 0 for shape, 1 for depth mean and 2 for depth , 3 for perspective prior(motion smoothness prior and depth relief prior)

% - prior_para: prior parameters([alpha_shape/depth_mean/depth,alpha_depth_diff,alpha_motion_smoothness,alpha_motion_smoothness,alpha_depth_relief])

% - mprior_mask: motion prior mask( m-1 dimension vector, default values is all 1, i.e. use motion prior )

% - cconst_mask: camera const mask ( m dimension vector, default value is all 0, i.e. no camera mask )

% output arguments

% - popt: estimated minimizer, i.e. minimized parameters vector.

error('cs_ba.m is used only for providing documentation to cs_ba; make sure that cs_ba.c has been compiled using mex');