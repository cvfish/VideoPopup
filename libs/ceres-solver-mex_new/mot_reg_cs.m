function p_rigmots = mot_reg_cs(nlab,nf,nb,p_rigmots,vvt,vvdt,u,ud,para,ind_ref,verbose)

%% Ceres-solver matlab MEX interface

% required input arguments
% - nlab: number of parts
% - nf: number of frames
% - nb: number of boundaries between segments
% - p_rigmots: 6*nf*nlab matrix initial value of motion parameters, axis angle representation
% - vvt: 16*nlab matrix,precomputed VV' of points in each part
% - vvdt: 18*nb matrix,precomputed VV' of points on the boundary
% - u: 16*nlab matrix, precomputed UD^(1/2)(svd decomposition of VV' of points in each part),
%       first 16 rows correspondes to UD^(1/2)
% - ud: 18*nb matrix, precomputed UD^(1/2)(svd decomposition of VV' of points in boundary)
%       first 2 rows correspondes to the id of the adjacent parts
%       next 16 rows correspondes to UD^(1/2)
% - para: 3*1 matrix, weighting parameters of three terms
% - ind_ref: reference frame which is assumed to be fixed, starting
% from one, default value is 1.
% - verbose: printing the optimization details or not, 0 for no, 1 for yes,
%   no printing by default

error('mot_reg_cs.m is used only for providing documentation to mot_reg_cs; make sure that mot_reg_cs.c has been compiled using mex');
