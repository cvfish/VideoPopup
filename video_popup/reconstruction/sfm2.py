"""
Two frame reconstruction
"""
import  cv2
import  numpy as np
import matplotlib.pyplot as plt

def reconstruction(W, K, img1 = None, img2 = None, checking = 0):

    p1 = W[0:2,:]
    p2 = W[2:4,:]

    #F, inliersF = cv2.findFundamentalMat(p1.T.astype(np.float32), p2.T.astype(np.float32), cv2.FM_LMEDS)
    F, inliersF = cv2.findFundamentalMat(p1.T.astype(np.float32), p2.T.astype(np.float32), cv2.RANSAC, 0.5)

    #H, inliersH = cv2.findHomography(p1.T.astype(np.float32), p2.T.astype(np.float32), cv2.LMEDS)
    H, inliersH = cv2.findHomography(p1.T.astype(np.float32), p2.T.astype(np.float32), cv2.RANSAC, 0.5)

    if(checking and img1 != None and img2 != None):
        check_f(img1, img2, F, W, inliersF)

    if(checking and img1 != None and img2 != None):
        check_h(img1, img2, H, W, inliersH)

    if( np.sum(inliersH) / np.sum(inliersF)  < 0.9 ):
        R,T = decompose_f(K, F, W)
        cost = fitting_error_f(p1, p2, F)
        inliers = inliersF
    else:
        R,T = decompose_h(K, H, W)
        cost = fitting_error_h(p1, p2, H)
        inliers = inliersH

    # do triangluation
    P1 = np.dot( K, np.hstack((np.identity(3), np.zeros( (3,1) ) ) ) )
    P2 = np.dot( K, np.hstack((R, T.reshape((3,1)))) )

    X = np.zeros( (4, W.shape[1]) )
    for i in range( W.shape[1] ):
        X[:, i] = triangulate(P1, P2, W[0:2, i], W[2:4, i])

    return R, T, X, cost, inliers.astype(bool)

def decompose_f(K, F, W):

    E = np.dot( np.dot( K.T, F ), K)
    U, s, V = np.linalg.svd( E, full_matrices=True )
    V = V.T
    E = E / (s[0] + s[1] / 2)

    W1_normalized = np.dot( np.linalg.inv(K),
                           np.vstack( ( W[0:2,:], np.ones( (1, W.shape[1]) ) ) ) )

    W2_normalized = np.dot( np.linalg.inv(K),
                           np.vstack( ( W[2:4,:], np.ones( (1, W.shape[1]) ) ) ) )

    W_normalized = np.vstack( ( W1_normalized[0:2,:], W2_normalized[0:2,:] ) )

    if(np.linalg.det(U) < 0):
        U = -U

    if(np.linalg.det(V) < 0):
        V = -V

    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])

    P1 = np.array([[1,0,0,0],
                   [0,1,0,0],
                   [0,0,1,0]])

    max_front_num = 0
    bestP = []

    A1 = np.dot( np.dot( U, W ), V.T )
    A2 = np.dot( np.dot( U, W.T ), V.T )
    B1 = U[:,2].reshape((3,1))
    B2 = -U[:,2].reshape((3,1))

    # np.hstack(
    #     (
    #         np.cross( B1[:,0], A1[:,0] ).reshape((3,1)),
    #         np.cross( B1[:,0], A1[:,1] ).reshape((3,1)),
    #         np.cross( B1[:,0], A1[:,2] ).reshape((3,1))
    #     )
    # ) - E

    for i in range(4):

        if(i == 0):
              P2 = np.hstack( ( A1 , B1) )
        if(i == 1):
              P2 = np.hstack( ( A1 , B2) )
        if(i == 2):
              P2 = np.hstack( ( A2 , B1) )
        if(i == 3):
              P2 = np.hstack( ( A2 , B2) )

        front_pnts_num = 0

        # check which one has most points in front
        if(W_normalized.shape[1] > 1000):
            sample = (W_normalized.shape[1] / 1000)
        else:
            sample = 1

        for k in range( 0, W_normalized.shape[1], sample ):

            X = triangulate(P1, P2, W_normalized[0:2, k], W_normalized[2:4, k])
            depth1 = compute_depth(P1, X)
            depth2 = compute_depth(P2, X)

            if(depth1 > 0 and depth2 > 0):
                front_pnts_num += 1

        if(front_pnts_num > max_front_num):
            max_front_num = front_pnts_num
            bestP = P2

    bestR = bestP[:,0:3]
    bestT = bestP[:,3]

    return bestR, bestT

def decompose_h(K, H, W):
    # reference "http://vision.ucla.edu//MASKS/MASKS-ch5.pdf", p136

    H_norm = np.dot( np.linalg.inv(K), np.dot( H, K ) )
    U, s, V = np.linalg.svd( H_norm, full_matrices=True )
    H_norm = H_norm / s[1]

    U, s, V = np.linalg.svd(H_norm, full_matrices=True)
    V = V.T
    """ now U*S*V.T = H_norm """

    W1_normalized = np.dot( np.linalg.inv(K),
                           np.vstack( ( W[0:2,:], np.ones( (1, W.shape[1]) ) ) ) )

    W2_normalized = np.dot( np.linalg.inv(K),
                           np.vstack( ( W[2:4,:], np.ones( (1, W.shape[1]) ) ) ) )

    W_normalized = np.vstack( ( W1_normalized[0:2,:], W2_normalized[0:2,:] ) )

    v1 = V[:,0].reshape((3,1))
    v2 = V[:,1].reshape((3,1))
    v3 = V[:,2].reshape((3,1))

    u1 = (v1 * np.sqrt(1 - s[2] ** 2) + v3 * np.sqrt( s[0]**2 - 1 ) ) / np.sqrt( s[0]**2 - s[2]**2 )
    u2 = (v1 * np.sqrt(1 - s[2] ** 2) - v3 * np.sqrt( s[0]**2 - 1 ) ) / np.sqrt( s[0]**2 - s[2]**2 )

    U1 = np.hstack( ( v2, u1, np.cross(v2.reshape(-1), u1.reshape(-1)).reshape((3,1)) ) )
    U2 = np.hstack( ( v2, u2, np.cross(v2.reshape(-1), u2.reshape(-1)).reshape((3,1)) ) )
    Hv2 = np.dot( H_norm, v2 )
    Hu1 = np.dot( H_norm, u1 )
    Hu2 = np.dot( H_norm, u2 )

    W1 = np.hstack( ( Hv2, Hu1, np.cross( Hv2.reshape(-1), Hu1.reshape(-1) ).reshape((3,1)) ) )
    W2 = np.hstack( ( Hv2, Hu2, np.cross( Hv2.reshape(-1), Hu2.reshape(-1) ).reshape((3,1)) ) )

    # we have 4 possible solutions
    R = {}
    N = {}
    T = {}

    R[0] = np.dot( W1, U1.T )
    N[0] = np.cross(v2.reshape(-1), u1.reshape(-1)).reshape((3,1))
    T[0] = np.dot( H_norm - R[0], N[0] )

    R[1] = np.dot( W2, U2.T )
    N[1] = np.cross(v2.reshape(-1), u2.reshape(-1)).reshape((3,1))
    T[1] = np.dot( H_norm - R[1], N[1] )

    R[2] = R[0]
    N[2] = -N[0]
    T[2] = -T[0]

    R[3] = R[1]
    N[3] = -N[1]
    T[3] = -T[1]

    # test which one has most in front points
    P1 = np.array([[1,0,0,0],
                   [0,1,0,0],
                   [0,0,1,0]])

    max_front_num = 0
    bestInd = -1

    for i in range(4):

        front_pnts_num = 0

        # check which one has most points in front
        if(W_normalized.shape[1] > 1000):
            sample = (W_normalized.shape[1] / 1000)
        else:
            sample = 1

        for k in range( 0, W_normalized.shape[1], sample ):

            P2 = np.hstack( ( R[i], T[i] ) )
            X = triangulate(P1, P2, W_normalized[0:2, k], W_normalized[2:4, k])
            depth1 = compute_depth(P1, X)
            depth2 = compute_depth(P2, X)

            if(depth1 > 0 and depth2 > 0):
                front_pnts_num += 1

        if(front_pnts_num > max_front_num):
            max_front_num = front_pnts_num
            bestInd = i

    return R[bestInd], T[bestInd].reshape(-1)

def triangulate(P1, P2, x1, x2):
    # P_1 * X = x1, P_2 * X = x2
    #
    # ( P_1(1,:) - P_1(3,:)*u1 ) * X = 0
    # ( P_1(2,:) - P_1(3,:)*v1 ) * X = 0
    #
    A = np.zeros((4,4))

    A[0,:] = P1[0,:] - P1[2,:] * x1[0]
    A[1,:] = P1[1,:] - P1[2,:] * x1[1]

    A[2,:] = P2[0,:] - P2[2,:] * x2[0]
    A[3,:] = P2[1,:] - P2[2,:] * x2[1]

    U, s, V = np.linalg.svd( A, full_matrices=True )
    V = V.T

    X = V[:,3]

    X = X / X[3]

    return X

def compute_depth(P, X):
    #check HZ2 p162
    depth = np.sign( np.linalg.det( P[:, 0:3] ) ) * np.dot( P[2,:], X ) / (X[3] * np.linalg.norm(P[2,0:3]) )

    return depth

def fitting_error_f(p1, p2, F, method = 'geometric'):

    P = p1.shape[1]
    x1 = np.vstack(( p1, np.ones((1, P)) ))
    x2 = np.vstack(( p2, np.ones((1, P)) ))

    Fx1 = np.dot(F, x1)
    Ftx2 = np.dot(F.T, x2)

    Fx1 = Fx1 / Fx1[2,:]
    Ftx2 = Ftx2 / Ftx2[2,:]

    if(method == 'geometric'):

        dx2 = np.abs( np.sum(x2 * Fx1, axis=0) / np.sqrt( Fx1[0, :] ** 2 + Fx1[1,:] ** 2 ) )
        dx1 = np.abs( np.sum(x1 * Ftx2, axis=0) / np.sqrt( Ftx2[0,:] ** 2 + Ftx2[1,:] ** 2 ) )
        cost = dx1 + dx2

    elif(method == 'sampson'):

        x2tFx1 = np.sum(x2 * Fx1, axis=0)
        cost = x2tFx1 ** 2 / (Fx1[0, :] ** 2 + Fx1[1, :] ** 2 + Ftx2[0, :] ** 2 + Ftx2[1, :] ** 2)

    return cost

def fitting_error_h(p1, p2, H, method = 'geometric'):

    P = p1.shape[1]
    x1 = np.vstack(( p1, np.ones((1, P)) ))
    x2 = np.vstack(( p2, np.ones((1, P)) ))

    Hx1 = np.dot(H, x1)
    invHx2 = np.dot(np.linalg.inv(H), x2)

    Hx1 = Hx1 / Hx1[2,:]
    invHx2 = invHx2 / invHx2[2,:]

    if(method == 'geometric'):

        dx2 = np.sqrt( np.sum( (x2 - Hx1)**2, axis=0 ) )
        dx1 = np.sqrt( np.sum( (x1 - invHx2)**2, axis=0 ) )
        cost = dx2 + dx1

    elif(method == 'sampson'):

        J = np.zeros((2,4))
        J11 = -H[1,0] + x2[1,:] * H[2,0]
        J12 = -H[1,1] + x2[1,:] * H[2,1]
        J13 = 0
        J14 = x1[0,:] * H[2,0] + x1[1,:] * H[2,1] + H[2,2]
        J21 = H[0,0] - x2[0,:] * H[2,0]
        J22 = H[0,1] - x2[0,:] * H[2,1]
        J23 = -J14
        J24 = 0

        err1 = - np.dot(H[1,:], x1) + x2[1,:] * np.dot(H[2,:], x1)
        err2 = np.dot(H[0,:], x1) - x2[0,:] * np.dot(H[2,:], x1)

        err1 = err1 * err1 / (J11 ** 2 + J12 ** 2 + J13 ** 2 + J14 ** 2)
        err2 = err2 * err2 / (J21 ** 2 + J22 ** 2 + J23 ** 2 + J24 ** 2)

        cost = err1 + err2

    return cost

def check_f(img1, img2, F, W, inliersF = None, sampling = 500):

    pts1 = W[0:2,:].T
    pts2 = W[2:4,:].T

    if(inliersF != None):
        pts1 = pts1[:, inliersF]
        pts2 = pts2[:, inliersF]

    pts1 = np.ascontiguousarray(pts1[0:-1:sampling, :], dtype=np.float32)
    pts2 = np.ascontiguousarray(pts2[0:-1:sampling, :], dtype=np.float32)

    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)

    pts1 = np.asarray(pts1, 'int32')
    pts2 = np.asarray(pts2, 'int32')

    r, c, _ = img1.shape

    numPnts = pts1.shape[0]

    color = np.random.randint(0, 255, 3 * numPnts).reshape((numPnts, 3))

    plt.subplot(121)

    plt.imshow(img1)
    plt.scatter(pts1[:, 0], pts1[:, 1], c=color / 255.0)

    p11 = np.ones_like(pts1)
    p12 = np.ones_like(pts1)

    ind = 0

    for r in lines1:

        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])

        p11[ind, 0] = x0
        p11[ind, 1] = y0
        p12[ind, 0] = x1
        p12[ind, 1] = y1

        ind += 1
        plt.plot([x0, x1], [y0, y1])

    """ the second image """

    plt.subplot(122)

    plt.imshow(img2)
    plt.scatter(pts2[:, 0], pts2[:, 1], c=color / 255.0)

    p21 = np.ones_like(pts1)
    p22 = np.ones_like(pts1)

    ind = 0

    for r in lines2:

        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])

        p21[ind, 0] = x0
        p21[ind, 1] = y0
        p22[ind, 0] = x1
        p22[ind, 1] = y1

        ind += 1
        plt.plot([x0, x1], [y0, y1])

    plt.show()

def check_h(img1, img2, H, W, inliersH = None, sampling = 500):

    pts1 = W[0:2, :].T
    pts2 = W[2:4, :].T

    if (inliersH != None):
        pts1 = pts1[:, inliersH]
        pts2 = pts2[:, inliersH]

    pts1 = np.ascontiguousarray(pts1[0:-1:sampling, :], dtype=np.float32)
    pts2 = np.ascontiguousarray(pts2[0:-1:sampling, :], dtype=np.float32)

    numPnts = pts1.shape[0]

    pts1_t = np.dot(H, np.vstack( ( pts1.T, np.ones(1, numPnts) ) ) )
    pts1_t = pts1_t / pts1_t[2,:]
    pts1_t = pts1_t.T

    color = np.random.randint(0, 255, 3 * numPnts).reshape((numPnts, 3))

    plt.subplot(121)

    plt.imshow(img1)
    plt.scatter(pts1[:, 0], pts1[:, 1], c = color / 255.0)

    """ the second image """

    plt.subplot(122)

    plt.imshow(img2)
    # plt.scatter(pts2[:, 0], pts2[:, 1], c = color / 255.0, marker='o')
    # plt.scatter(pts1_t[:, 0], pts1_t[:, 1], c = color / 255.0, marker='+')
    plt.scatter(pts2[:, 0], pts2[:, 1], c='g', marker='o')
    plt.scatter(pts1_t[:, 0], pts1_t[:, 1], c='r', marker='+')

    plt.show()