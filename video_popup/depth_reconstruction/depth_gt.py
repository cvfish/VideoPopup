# check the ground truth depth image

import cv2
import numpy as np

import matplotlib.pyplot as plt

import depth_util

expr = 'kitti_test'
expr = 'kitti_rigid'
#expr = 'kitti_05'

if(expr == 'kitti_test'):

    folder = '../../data/Kitti/05/broxmalik_Size4/'
    gt_file = folder + '002491.bin'
    image_file = folder + '002491.ppm'

    img = cv2.imread(image_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    nH, nW, nC = img.shape

    K = np.array([[707.0912, 0, 601.8873],
                  [0, 707.0912, 183.1104],
                  [0, 0, 1.0000 ]])

    Tr = np.array([[-0.001857739385241,  -0.999965951351000,  -0.008039975204516,  -0.004784029760483],
                   [-0.006481465826011,   0.008051860151134,  -0.999946608177400,  -0.073374294642310],
                   [0.999977309828700,  -0.001805528627661,  -0.006496203536139,  -0.333996806443300]])

    depth_map_gt_interp, uvd = depth_util.get_kitti_depth_gt(K, Tr, nH, nW, gt_file, depth_min = 0, depth_max=10000)

    plt.imshow(depth_map_gt_interp)

    plt.waitforbuttonpress()

elif(expr == 'kitti_rigid'):

    folder = '../../data/Kitti/05_rigid/broxmalik_Size2/'
    gt_file = folder + '002637.bin'
    image_file = folder + '002637.ppm'
    img = cv2.imread(image_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    nH, nW, nC = img.shape

    K = np.array([[707.0912, 0, 601.8873],
                  [0, 707.0912, 183.1104],
                  [0, 0, 1.0000 ]])

    Tr = np.array([[-0.001857739385241,  -0.999965951351000,  -0.008039975204516,  -0.004784029760483],
                   [-0.006481465826011,   0.008051860151134,  -0.999946608177400,  -0.073374294642310],
                   [0.999977309828700,  -0.001805528627661,  -0.006496203536139,  -0.333996806443300]])

    depth_map_gt_interp, uvd = depth_util.get_kitti_depth_gt(K, Tr, nH, nW, gt_file, depth_min = 0, depth_max=10000)

    plt.imshow(depth_map_gt_interp)

    plt.waitforbuttonpress()

elif(expr == 'kitti_05'):

    root_folder = '/media/cvfish/Seagate/Work/Datasets/Kitti/Kitti2012/'
    gt_folder = '/dataset/sequences/05/velodyne/'
    image_folder = 'dataset_color/sequences/05/image_2/'
    save_folder = '/dataset_gt_imgs/sequences/05/image_2/'

    first_img = 2491
    last_img = 2506

    K = np.array([[707.0912, 0, 601.8873],
                  [0, 707.0912, 183.1104],
                  [0, 0, 1.0000 ]])

    Tr = np.array([[-0.001857739385241,  -0.999965951351000,  -0.008039975204516,  -0.004784029760483],
                   [-0.006481465826011,   0.008051860151134,  -0.999946608177400,  -0.073374294642310],
                   [0.999977309828700,  -0.001805528627661,  -0.006496203536139,  -0.333996806443300]])

    fig = plt.figure()

    fig, (ax1, ax2) = plt.subplots(1, 2)

    is_first = 1

    depth_maps = {}
    ref_images = {}

    for i in range(first_img, last_img):

        gt_file = root_folder + gt_folder + '{:06d}.bin'.format(i)
        image_file = root_folder + image_folder + '{:06d}.png'.format(i)
        results_folder = root_folder + save_folder

        img = mpimg.imread(image_file)
        # img = cv2.imread(image_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        nH, nW, nC = img.shape

        #test = util.image_inpainting_dense( img[:,:,0].reshape((nH, nW)) )

        # depth_map_gt_interp, uvd = depth_util.get_kitti_depth_gt(K, Tr, nH, nW, gt_file, depth_min = 0, depth_max=10000)
        depth_map_gt_interp, uvd = depth_util.get_kitti_depth_gt(K, Tr, nH, nW, gt_file, depth_min = 5, depth_max=20)

        vertices = np.dot(np.linalg.inv(K),
                          uvd[2].reshape((1,-1))
                          * np.vstack((uvd[0].reshape((1,-1)),
                                       uvd[1].reshape((1,-1)),
                                       np.ones((1, len(uvd[0]) ) ) ) )
                          ).T
        u_int = np.minimum(nW-1,
                                np.maximum(0, uvd[0].reshape((1,-1)).astype(np.uint32)))
        v_int = np.minimum(nH-1,
                                np.maximum(0, uvd[1].reshape((1,-1)).astype(np.uint32)))

        colors = img[v_int, u_int, :].reshape((-1,3))

        labels = np.zeros((colors.shape[0])).astype(np.uint32)

        #depth_util.point_cloud_plot(vertices, colors, K, nH, nW, labels)

        ref_images[i - first_img] = img
        depth_maps[i - first_img] = depth_map_gt_interp

        # depth_util.depth_map_plot(depth_map_gt_interp, img, K)
        #
        # fig.suptitle("depth_checking: image " + '{:06d}'.format(i) )
        #
        # if(i == first_img):
        #     im1 = ax1.imshow(img)
        #     plt.axis("off")
        #     im2 = ax2.imshow(depth_map_gt_interp)
        #     plt.axis("off")
        # else:
        #     im1.set_data(img)
        #     im2.set_data(depth_map_gt_interp)
        #     fig.canvas.draw()
        #
        # plt.pause(.1)
        # plt.draw()
        #
        # util.ensure_dir(results_folder)
        # fig.savefig(results_folder + '{:06d}.png'.format(i), bbox_inches='tight')
        #
        # # ax1 = fig.add_subplot(1, 2, 1)
        # # ax1.imshow(img)
        # # plt.axis("off")
        # #
        # # ax2 = fig.add_subplot(1, 2, 2)
        # # ax2.imshow(depth_map_gt_interp)
        # # plt.axis("off")
        # #
        # # plt.pause(.1)
        # # plt.draw()

    depth_util.depth_maps_plot(depth_maps, ref_images, K)