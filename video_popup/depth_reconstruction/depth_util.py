import numpy as np
import cvxpy as cp

from scipy.interpolate import griddata
from video_popup.visualization.vispy_viewer import app_call

def get_kitti_depth_gt(K, Tr, nH, nW, gt_file, depth_min=5, depth_max=20):

    try:
        with open(gt_file, "rb") as f:
            # struct.unpack('f',f.read(4))
            A = np.fromfile(f, dtype=np.float32)
    except:
        print "reading error"

    X = A.reshape((-1,4))
    X[:,3] = 1

    # the first half of the ground data is corrupted, we should delete these
    numPnts = X.shape[0]
    X = X[0:numPnts/2,:]

    X1 = np.dot(Tr, X.T)
    X1 = X1[:, X1[2,:] > 0  ]

    x = np.dot(K, X1)
    depths = x[2,:]
    x = x / depths

    grid_v, grid_u = np.mgrid[0:nH, 0:nW]
    # depth_map_gt_interp = griddata(x[1::-1,:].T, depths, (grid_v, grid_u), method='nearest')
    depth_map_gt_interp = griddata(x[1::-1,:].T, depths, (grid_v, grid_u), method='linear')

    depth_mask = np.logical_and(depths < depth_max, depths > depth_min)

    uvd = (x[0,depth_mask], x[1,depth_mask], depths[depth_mask])

    return  depth_map_gt_interp, uvd

def evaluate_dense(depth_map, depth_map_gt, mask,
                   step=10, K=0, check_3d=0, ref_img=0):

    global_scale_op = cp.Variable()

    mask_sample = mask[0:-1:step, 0:-1:step]
    depth_map_sample = depth_map[0:-1:step, 0:-1:step]
    depth_map_gt_sample = depth_map_gt[0:-1:step, 0:-1:step]

    numPnts_sample = np.sum(mask_sample)

    ## set up the optimization problem to get optimal scale
    # if(method == 'MRE'):
    objective_gs = cp.Minimize( 1.0 / numPnts_sample * cp.sum_entries( cp.mul_elemwise(
        mask_sample * 1.0 / depth_map_gt_sample,
        cp.abs( depth_map_sample * global_scale_op - depth_map_gt_sample ) ) ) )
    # elif(method == 'RMSE'):
    #     objective_gs = cp.Minimize( cp.sqrt( 1.0 / numPnts_sample * cp.sum_squares( cp.mul_elemwise(
    #         mask_sample,
    #         depth_map_sample * global_scale_op - depth_map_gt_sample ) ) ) )
    # elif(method == 'LOG10'): # in the unit of meters ?
    #     objective_gs = cp.Minimize( step ** 2.0 * cp.sum_entries( cp.mul_elemwise(
    #         mask_sample,
    #         cp.log( depth_map_sample * global_scale_op ) - cp.log( depth_map_gt_sample ) ) ) )

    prob_gs = cp.Problem(objective_gs)
    prob_gs.solve(verbose=True, solver=cp.SCS)

    global_scale = global_scale_op.value

    # if we we want to check if the alignment to ground truth has been done properly
    if(check_3d and isinstance(K, np.ndarray) and isinstance(ref_img, np.ndarray)):

        nH, nW = depth_map.shape
        grid_v, grid_u = np.mgrid[0:nH, 0:nW]
        grid_v_sample = grid_v[0:-1:step, 0:-1:step].reshape((1,-1))
        grid_u_sample = grid_u[0:-1:step, 0:-1:step].reshape((1,-1))

        grid_v_sample = grid_v_sample[mask_sample.reshape((1,-1))]
        grid_u_sample = grid_u_sample[mask_sample.reshape((1,-1))]

        vertices = np.dot(
            np.linalg.inv(K),
            global_scale * depth_map_sample[mask_sample].reshape((1,-1)) *
            np.vstack((grid_u_sample,
                       grid_v_sample,
                       np.ones((1, numPnts_sample))))
        ).astype(np.float32).T

        vertices_gt =  np.dot(
            np.linalg.inv(K),
            depth_map_gt_sample[mask_sample].reshape((1,-1)) *
            np.vstack((grid_u_sample,
                       grid_v_sample,
                       np.ones((1, numPnts_sample))))
        ).astype(np.float32).T

        colors = ref_img[grid_v_sample, grid_u_sample, :]

        vertices = np.vstack((vertices, vertices_gt))
        colors = np.vstack((colors, colors))

        labels = np.vstack((np.zeros((numPnts_sample,1)),
                            np.ones((numPnts_sample,1)))).astype(np.uint32)

        point_cloud_plot(vertices, colors, K, nH, nW, labels)

    depth_map = depth_map * global_scale

    depth_map_mask = depth_map[mask]
    depth_map_gt_mask = depth_map_gt[mask]
    numPnts = np.sum(mask)

    # if(method == 'MRE'):
    error_mre = np.sum(np.abs(depth_map_gt_mask - depth_map_mask).astype(np.float32) / depth_map_gt_mask) / numPnts
    #     return error_mre, depth_map
    # elif(method == 'RMSE'):
    error_rmse = np.sqrt( np.sum( (depth_map_mask - depth_map_gt_mask)** 2.0 ) / numPnts )
    #     return  error_rmse, depth_map
    # elif(method == 'LOG10'):
    error_log10 = np.sum( np.abs(np.log10(depth_map_mask) - np.log10(depth_map_gt_mask)) )

    outlier_mask = (np.abs(depth_map_gt_mask - depth_map_mask).astype(np.float32) / depth_map_gt_mask) > 2 * error_mre

    # return  error_mre, error_rmse, error_log10, depth_map.astype(np.uint16), outlier_mask, global_scale
    return  error_mre, error_rmse, error_log10, depth_map, outlier_mask, global_scale

def evaluate_sparse(depth_map, u_gt, v_gt, d_gt,
                    step = 30, K=0, check_3d=0, ref_img=0):

    # d = depth_map[v_gt, u_gt]

    nH, nW = depth_map.shape
    grid_v, grid_u = np.mgrid[0:nH, 0:nW]

    d = griddata(np.hstack((grid_v.reshape(-1,1), grid_u.reshape(-1,1))),
                 depth_map.reshape(-1,), (v_gt, u_gt), method='nearest')

    keep_mask = np.logical_not( np.logical_or(
        d == np.inf, d == -np.inf, np.isnan(d) ) )

    d_sample = d[keep_mask][0:-1:step]
    d_gt_sample = d_gt[keep_mask][0:-1:step]

    numPnts_sample = len(d_sample)

    global_scale_op = cp.Variable()

    objective_gs = cp.Minimize( 1.0 / numPnts_sample *
                                cp.sum_entries( 1.0 / d_gt_sample * cp.abs( d_sample * global_scale_op - d_gt_sample ) ) )

    prob_gs = cp.Problem(objective_gs)

    prob_gs.solve(verbose=True, solver=cp.SCS)

    global_scale = global_scale_op.value

    if(check_3d and isinstance(K, np.ndarray) and isinstance(ref_img, np.ndarray)):

        vertices = np.dot(
            np.linalg.inv(K),
            global_scale * d_sample.reshape((1,-1)) *
            np.vstack((u_gt[0:-1:step],
                       v_gt[0:-1:step],
                       np.ones((1, numPnts_sample))))
        ).astype(np.float32).T

        vertices_gt =  np.dot(
            np.linalg.inv(K),
            d_gt_sample.reshape((1,-1)) *
            np.vstack((u_gt[0:-1:step],
                       v_gt[0:-1:step],
                       np.ones((1, numPnts_sample))))
        ).astype(np.float32).T

        u_int = np.minimum(nW-1,
                                np.maximum(0, u_gt[0:-1:step].astype(np.uint32)))
        v_int = np.minimum(nH-1,
                                np.maximum(0, v_gt[0:-1:step].astype(np.uint32)))

        colors = ref_img[v_int, u_int, :]

        vertices = np.vstack((vertices, vertices_gt))
        colors = np.vstack((colors, colors))

        labels = np.vstack((np.zeros((numPnts_sample,1)),
                            np.ones((numPnts_sample,1)))).astype(np.uint32)

        point_cloud_plot(vertices, colors, K, nH, nW, labels)

    depth_map = depth_map * global_scale

    # d = depth_map[v_gt, u_gt]
    d = griddata(np.hstack((grid_v.reshape(-1,1), grid_u.reshape(-1,1))),
                 depth_map.reshape(-1,), (v_gt, u_gt), method='nearest')

    d = d[keep_mask]; d_gt = d_gt[keep_mask]

    numPnts = len(d)

    error_mre = np.sum( np.abs( d - d_gt ) / d_gt ) / numPnts

    error_rmse = np.sqrt( np.sum( ( d - d_gt ) ** 2 ) / numPnts )

    error_log10 = np.sum( np.abs( np.log10( d_gt) - np.log10( d ) ) )

    outlier_mask = np.abs( d - d_gt ) / d_gt > 2*error_mre

    return error_mre, error_rmse, error_log10, depth_map, outlier_mask, global_scale

def depth_map_plot(depth_map, ref_image, K, labels = 0, edge_thresh = 1000):

    nH, nW, nC = ref_image.shape
    grid_x, grid_y = np.mgrid[0:nH, 0:nW]

    grid_x = grid_x.reshape((1,-1))
    grid_y = grid_y.reshape((1,-1))

    colors = ref_image.reshape((-1,3))

    vertices = np.dot(np.linalg.inv(K),
                      depth_map.reshape((1,-1)) * np.vstack( ( grid_y.reshape((1,-1)),
                                                               grid_x.reshape((1,-1)),
                                                               np.ones((1, nH*nW)) ) ) ).T.astype(np.float32)
    vertices[:,1] = -vertices[:,1]
    vertices[:,2] = -vertices[:,2]

    app_call(vertices, colors, K, nH, nW, image_grid = 1, labels = labels, edge_thresh =  edge_thresh)

def depth_maps_plot(depth_maps, ref_image, K, labels = 0, edge_thresh = 1000):

    if(isinstance(ref_image, dict)):
        nH, nW, nC = ref_image[0].shape
        colors = ref_image[0].reshape((-1,3))
    else:
        nH, nW, nC = ref_image.shape
        colors = ref_image.reshape((-1,3))

    grid_x, grid_y = np.mgrid[0:nH, 0:nW]

    grid_x = grid_x.reshape((1,-1))
    grid_y = grid_y.reshape((1,-1))

    vertices = np.dot(np.linalg.inv(K),
                      depth_maps[0].reshape((1,-1)) * np.vstack( ( grid_y.reshape((1,-1)),
                                                                   grid_x.reshape((1,-1)),
                                                                   np.ones((1, nH*nW)) ) ) ).T.astype(np.float32)
    frames = len(depth_maps)

    for i in range(1, frames):
        vertices_i = np.dot(np.linalg.inv(K),
                            depth_maps[i].reshape((1,-1)) * np.vstack( ( grid_y.reshape((1,-1)),
                                                                         grid_x.reshape((1,-1)),
                                                                         np.ones((1, nH*nW)) )) ).T.astype(np.float32)
        vertices = np.vstack((vertices, vertices_i))

        colors_i = ref_image[i].reshape((-1,3))
        colors = np.vstack((colors, colors_i))

    vertices[:,1] = -vertices[:,1]
    vertices[:,2] = -vertices[:,2]

    app_call(vertices, colors, K, nH, nW, image_grid = 1, labels = labels, nframes = frames, edge_thresh = edge_thresh)

def point_cloud_plot(vertices, colors, K, nH, nW, labels=0, nframes=1):

    vertices[:,1] = -vertices[:,1]
    vertices[:,2] = -vertices[:,2]

    app_call(vertices, colors, K, nH, nW, labels=labels, nframes=nframes)

# mask = depth_map_gt != np.max(depth_map_gt)
#
# # global scale optimization
# step = 10
# global_scale_op = cp.Variable()
# objective_gs = cp.Minimize( cp.sum_entries( cp.mul_elemwise(
#     1.0 / depth_map_gt[0:-1:step, 0:-1:step] * mask[0:-1:step, 0:-1:step],
#     cp.abs( depth_map[0:-1:step, 0:-1:step] * global_scale_op -
#             depth_map_gt[0:-1:step, 0:-1:step] ) ) ) * step ** 2  / (nH * nW) )
#
# prob_gs = cp.Problem(objective_gs)
# # result_gs = prob_gs.solve(verbose=True, solver=cp.CVXOPT)
# result_gs = prob_gs.solve(verbose=True, solver=cp.SCS)
# global_scale = global_scale_op.value
#
# depth_map = depth_map * global_scale
#
# fig = plt.figure("depth evaluation")
# ax1 = fig.add_subplot(2, 2, 1)
# ax1.imshow(depth_map)
# ax2 = fig.add_subplot(2, 2, 2)
# ax2.imshow(depth_map_gt)
# ax3 = fig.add_subplot(2, 2, 3)
# ax3.imshow(depth_map_gt - depth_map)
# ax3 = fig.add_subplot(2, 2, 4)
# ax3.imshow(mask)
#
# plt.axis("off")
# plt.show()
#
# plt.waitforbuttonpress()
#
# # compute errors
# error_mre = np.sum( np.abs(depth_map_gt - depth_map).astype(np.float32) * mask / depth_map_gt ) / (nH * nW)
#
#
# print error_mre
#
# depth_results = {'depth_map': depth_map, 'depth_map_gt': depth_map_gt, 'mask': mask,
#                  'error_mre': error_mre, }





