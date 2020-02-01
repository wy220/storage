# 主函数从 513 行开始
# 最小二乘使用雅各比矩阵运行出现异常
# 没得到内参结果 外参结果 和畸变系数的值 
# 在进一步优化的时候，同样使用最小二乘法进行极大似然估计，还是得不到结果

import numpy as np 
from scipy import optimize as opt
import math
import glob
import cv2 

# 初始化输入数据的归一化矩阵
def normalizing_Matrix(coor):

    x_avg = np.mean(coor[:, 0])
    y_avg = np.mean(coor[:, 1])
    
    X = 2**0.5 / (np.std(coor[:, 0])) 
    Y = 2**0.5 / (np.std(coor[:, 1]))
    
    normalMatrix = np.array(
        [
            [X, 0, -X*x_avg],
            [0, Y, -Y*y_avg],
            [0, 0, 1]
        ]
    )

    return normalMatrix

# 由初始化的归一化矩阵，初始化单应性矩阵
def get_initial_H(corners, objp):
    img_normal_mat = normalizing_Matrix(corners)
    obj_normal_mat = normalizing_Matrix(objp)
    
    M = []
    
    for i in range(len(corners)):
        # 转为齐次坐标
        single_coor_img = np.array(
            [corners[i][0], corners[i][1], 1]
        )
        single_coor_obj = np.array(
            [objp[i][0], objp[i][0], 1]
        )
        # print(single_coor_img)

        # 坐标归一化
        img_normal = np.dot(img_normal_mat, single_coor_img)
        obj_normal = np.dot(obj_normal_mat, single_coor_obj)
        
        # print(img_normal, obj_normal)
        
        # 构造 M 矩阵
        M.append(
            [
                    -obj_normal[0], -obj_normal[1], -1,
                      0, 0, 0,
                      img_normal[0] * obj_normal[0], img_normal[0] * obj_normal[1], img_normal[0]
                ]
        )
        M.append(
            [
                    0, 0, 0,
                      -obj_normal[0], -obj_normal[0], -1,
                      img_normal[1] * obj_normal[0], img_normal[1] * obj_normal[1], img_normal[1]
                ]
        )
    M = np.array(M)    # (108, 9)
    # print(M)
    
    # 利用 SVD 求解 V * b = 0 中 b 的解
    U, S, VT = np.linalg.svd(
        (np.array(M, dtype='float32')).reshape((-1, 9))
    )

    # 最小的奇异值对应的奇异向量, S 求出来按大小排列
    H = VT[-1].reshape((3, 3))
    H = np.dot(
        np.dot(np.linalg.inv(img_normal_mat), H), obj_normal_mat
    )
    H /= (H[-1, -1])
    
    return H
    
# 计算 3D 坐标 * H 的值与真实 2D 坐标的偏差
def bias(H, corners, objp):
    H = np.array(H)
    labels = []
    for i in range(len(objp)):
        single_coor_obj = np.array(
            [objp[i][0], objp[i][1], 1]
        )
        U = np.dot(H.reshape(3, 3), single_coor_obj)
        U /= U[-1]
        labels.append(U)

    labels = np.array(labels)
    
    New_2D_labels = []
    
    for i in range(len(labels)):
        New_2D_labels.append(labels[i][:2])

    New_2D_labels = np.array(New_2D_labels)
    
    biasMatrix = corners - New_2D_labels

    return biasMatrix
    
# 建立雅各比矩阵
def jacobian(H, corners, objp):
    J = []
    H = H.reshape(1, 9)
    H = H[0]
    for i in range(len(objp)):
        X = H[0]*objp[i][0] + H[1]*objp[i][1] + H[2]
        Y = H[3]*objp[i][0] + H[4]*objp[i][1] + H[5]
        W = H[6]*objp[i][0] + H[7]*objp[i][1] + H[8]
        W_2 = W*W

        temp = (
            np.array(
                [objp[i][0] / W, objp[i][1] / W, 1/W, 0, 0, 0, -X*objp[i][0] / W_2, -X*objp[i][1] / W_2, -X / W_2]
            )
            + 
            np.array(
                [0, 0, 0, objp[i][0] / W, objp[i][1] / W, 1/W, -Y*objp[i][0] / W_2, -Y*objp[i][1] / W_2, -Y / W_2]
            )
        ) / 2   
        J.append(temp)
    
    J = np.array(J)    # (54, 9)

    return J
    
# LM 算法 进行极大似然估计，但 leastsq 函数和我得到的雅各比矩阵一起用报错，我也没找到合适的解决办法
def refine_H(corners, objp, initial_H):
    # initial_H = initial_H.reshape(1, 9)
    final_H = opt.leastsq(
        bias, 
        initial_H, 
        args=(corners, objp),
        Dfun = jacobian    # 应该是这里出现问题
    )[0]
    # 报错结果：
    # object too deep for desired array
    # Result from function call is not a proper array of floats.

    final_H /= np.array(final_H[-1])

    return final_H


def get_homography(corners, objp):
    refined_homographies =[]
    error = []
    for i in range(len(corners)):
        initial_H = get_initial_H(corners[i], objp[i])
        final_H = refine_H(corners[i], objp[i], initial_H)
        refined_homographies.append(final_H)
    
    refined_homographies = np.array(refined_homographies)
 
    return refined_homographies


# 自己思考下的解方程方法，但结果是奇异矩阵
# 认为是两个坐标系下的点不对应导致的
'''
def linalgSolve(inputList, outputList):
    inputMatrix = np.zeros((9, 9), np.float32)
    for i in range(9):
        if i < 3:
            inputMatrix[i, 3*(i%3):3+3*(i%3)] = inputList[:3]
        elif i < 6:
            inputMatrix[i, 3*(i%3):3+3*(i%3)] = inputList[3:6]
        else:
            inputMatrix[i, 3*(i%3):3+3*(i%3)] = inputList[6:9]

    H = np.linalg.solve(inputMatrix, outputList)
    H = np.array(H)

    return H 

def avg_H(corners_3D, objp):
    X = []
    Y = []
    for i in range(0, len(objp), 18):
        X.append(objp[i])
        Y.append(corners_3D[i])
    X = np.array(X)
    Y = np.array(Y)
    X = X.reshape(1, 9)[0]
    Y = Y.reshape(1, 9)[0]
    H = linalgSolve(X, Y)
'''

# 接下来求解内参矩阵

# 得到 i，j 位置对应的 v 向量
def create_v(i, j, H):
    H = H.reshape(3, 3)
    v = np.array(
        [
            H[0, i] * H[0, j],
        H[0, i] * H[1, j] + H[1, i] * H[0, j],
        H[1, i] * H[1, j],
        H[2, i] * H[0, j] + H[0, i] * H[2, j],
        H[2, i] * H[1, j] + H[1, i] * H[2, j],
        H[2, i] * H[2, j]
        ]
    )

    return v

# 求解相机内参矩阵 A
def get_intrinsics_param(H):
    # 创建 V 矩阵
    V = np.array([])
    for i in range(len(H)):
        V = np.append(
            V, np.array(
                [create_v(0, 1, H[i]), create_v(0, 0 , H[i]) - create_v(1, 1 , H[i])]
            )
        )
 
    # 求解 V*b = 0 中的 b
    U, S, VT = np.linalg.svd(
        (np.array(V, dtype='float32')).reshape((-1, 6))
    )
    # 最小的奇异值对应的奇异向量,S求出来按从大到小排列的
    b = VT[-1]
 
    # 求相机内参
    w = b[0] * b[2] * b[5] - b[1] * b[1] * b[5] - b[0] * b[4] * b[4] + 2 * b[1] * b[3] * b[4] - b[2] * b[3] * b[3]

    d = b[0] * b[2] - b[1] * b[1]
 
    alpha = np.sqrt(w / (d * b[0]))
    beta = np.sqrt(w / d**2 * b[0])
    gamma = np.sqrt(w / (d**2 * b[0])) * b[1]
    uc = (b[1] * b[4] - b[2] * b[3]) / d
    vc = (b[1] * b[3] - b[0] * b[4]) / d
    
    A = np.array([
        [alpha, gamma, uc],
        [0,     beta,  vc],
        [0,     0,      1]
    ])

    return A

# 求解外参矩阵 [R | T]
def get_extrinsics_param(H, intrinsics_param):
    extrinsics_param = []

    inv_intrinsics_param = np.linalg.inv(intrinsics_param)
    for i in range(len(H)):
        h0 = (H[i].reshape(3, 3))[:, 0]
        h1 = (H[i].reshape(3, 3))[:, 1]
        h2 = (H[i].reshape(3, 3))[:, 2]
    
    scale_factor = 1 / np.linalg.norm(
        np.dot(inv_intrinsics_param, h0)
    )

    r0 = scale_factor * np.dot(inv_intrinsics_param, h0)
    r1 = scale_factor * np.dot(inv_intrinsics_param, h1)
    t = scale_factor * np.dot(inv_intrinsics_param, h2)

    r2 = np.cross(r0, r1)

    RT = np.array([r0, r1, r2, t]).transpose()
    extrinsics_param.append(RT)

    return extrinsics_param

# 求解畸变矫正系数 k1, k2
def get_distortion(intrinsics_param, extrinsics_param, corners, objp):
    D = []
    d = []
    for i in range(len(corners)):
        for j in range(len(corners[i])):
            #转换为齐次坐标
            single_coor = np.array([(objp[i])[j, 0], (objp[i])[j, 1], 0, 1])

            #利用现有内参及外参求出估计图像坐标
            u = np.dot(
                np.dot(intrinsic_param, extrinsic_param[i]), single_coor
            )
            [u_estim, v_estim] = [u[0]/u[2], u[1]/u[2]]

            # 归一化坐标
            normal_coor = np.dot(extrinsic_param[i], single_coor)
            normal_coor /= normal_coor[-1]

            r = np.linalg.norm(normal_coor)

            D.append(np.array([(u_estim - intrinsic_param[0, 2]) * r ** 2, (u_estim - intrinsic_param[0, 2]) * r ** 4]))
            D.append(np.array([(v_estim - intrinsic_param[1, 2]) * r ** 2, (v_estim - intrinsic_param[1, 2]) * r ** 4]))

            #求出估计坐标与真实坐标的残差
            d.append(corners[i][j, 0] - u_estim)
            d.append(corners[i][j, 1] - v_estim)

    D = np.array(D)
    temp = np.dot(np.linalg.inv(np.dot(D.T, D)), D.T)

    k = np.dot(temp, d)

    return k

# 使用极大似然估计提升内参外参和畸变系数的精度
def refinall_all_param(intrinsics, k, extrinsics, objp, corners):
    P_init = compose_paramter_vector(intrinsics, k, extrinsics)

    X_double = np.zeros((2 * len(objp) * len(objp[0]), 3))
    Y = np.zeros((2 * len(objp) * len(objp[0])))
 
    M = len(objp)
    N = len(objp[0])
    for i in range(M):
        for j in range(N):
            X_double[(i * N + j) * 2] = (objp[i])[j]
            X_double[(i * N + j) * 2 + 1] = (objp[i])[j]
            Y[(i * N + j) * 2] = (corners[i])[j, 0]
            Y[(i * N + j) * 2 + 1] = (corners[i])[j, 1]

    P = opt.leastsq(
        value, P_init, args=(extrinsics, objp, corners), Dfun=Jacobian
    )[0]

    error = value(P, extrinsics, objp, corners)
    raial_error = [np.sqrt(error[2 * i]**2 + error[2 * i + 1]**2) \
        for i in range(len(error) // 2)]
    
    print("total max error:\t", np.max(raial_error))
 
    # 返回拆解后参数，分别为内参矩阵，畸变矫正系数，每幅图对应的外参矩阵
    return decompose_paramter_vector(P)


def compose_paramter_vector(intrinsics, k, extrinsics):
    alpha = np.array([intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 1], intrinsics[0, 2], intrinsics[1, 2], k[0], k[1]])

    P = alpha
    for i in range(len(extrinsics)):
        R, t = (extrinsics[i])[:, :3], (extrinsics[i])[:, 3]
 
        # 旋转矩阵转换为一维向量形式
        zrou = to_rodrigues_vector(R)
 
        w = np.append(zrou, t)
        P = np.append(P, w)

    return P

def decompose_paramter_vector(P):
    [alpha, beta, gamma, uc, vc, k0, k1] = P[0:7]
    A = np.array([[alpha, gamma, uc],
                  [0, beta, vc],
                  [0, 0, 1]])
    k = np.array([k0, k1])
    W = []
    M = (len(P) - 7) // 6
 
    for i in range(M):
        m = 7 + 6 * i
        zrou = P[m:m+3]
        t = (P[m+3:m+6]).reshape(3, -1)
 
        #将旋转矩阵一维向量形式还原为矩阵形式
        R = to_rotation_matrix(zrou)
 
        #依次拼接每幅图的外参
        w = np.concatenate((R, t), axis=1)
        W.append(w)
 
    W = np.array(W)

    return A, k, W

#把旋转矩阵的一维向量形式还原为旋转矩阵并返回
def to_rotation_matrix(zrou):
    theta = np.linalg.norm(zrou)
    zrou_prime = zrou / theta
 
    W = np.array(
        [
            [0, -zrou_prime[2], zrou_prime[1]],
            [zrou_prime[2], 0, -zrou_prime[0]],
            [-zrou_prime[1], zrou_prime[0], 0]
        ]
    )
    R = np.eye(3, dtype='float') + W * math.sin(theta) + np.dot(W, W) * (1 - math.cos(theta))
 
    return R

# 求解所有世界坐标映射到的图像坐标与真实图像坐标的残差
def value(P, org_W, X, Y_real):
    M = (len(P) - 7) // 6
    N = len(X[0])
    A = np.array([
        [P[0], P[2], P[3]],
        [0, P[1], P[4]],
        [0, 0, 1]
    ])
    Y = np.array([])
 
    for i in range(M):
        m = 7 + 6 * i
        # 取出当前图像对应的外参
        w = P[m:m + 6]

        W = org_W[i]
        # 计算每幅图的坐标残差
        for j in range(N):
            Y = np.append(
                Y, get_single_project_coor(A, W, np.array([P[5], P[6]]), (X[i])[j])
            )
 
    error_Y  =  np.array(Y_real).reshape(-1) - Y
 
    return error_Y

def get_single_project_coor(A, W, k, coor):
    single_coor = np.array([coor[0], coor[1], coor[2], 1])
    normal_coor = np.dot(W, single_coor)
    normal_coor /= normal_coor[-1]
    r = np.linalg.norm(normal_coor)
    uv = np.dot(np.dot(A, W), single_coor)
    uv /= uv[-1]
    # 畸变
    u0 = uv[0]
    v0 = uv[1]
    uc = A[0, 2]
    vc = A[1, 2]
    u = u0 + (u0 - uc) * r**2 * k[0] + (u0 - uc) * r**4 * k[1]
    v = v0 + (v0 - vc) * r**2 * k[0] + (v0 - vc) * r**4 * k[1]

    return np.array([u, v])


def to_rodrigues_vector(R):
    p = 0.5 * np.array(
        [
            [R[2, 1] - R[1, 2]],
            [R[0, 2] - R[2, 0]],
            [R[1, 0] - R[0, 1]]
        ]
    )
    c = 0.5 * (np.trace(R) - 1)
 
    if np.linalg.norm(p) == 0:
        if c == 1:
            zrou = np.array([0, 0, 0])
        elif c == -1:
            R_plus = R + np.eye(3, dtype='float')
 
            norm_array = np.array(
                [
                    np.linalg.norm(R_plus[:, 0]),
                    np.linalg.norm(R_plus[:, 1]),
                    np.linalg.norm(R_plus[:, 2])
                ]
            )
            v = R_plus[:, np.where(norm_array == max(norm_array))]
            u = v / np.linalg.norm(v)
            if u[0] < 0 or (u[0] == 0 and u[1] < 0) or (u[0] == u[1] and u[0] == 0 and u[2] < 0):
                u = -u
            zrou = math.pi * u
        else:
            zrou = []
    else:
        u = p / np.linalg.norm(p)
        theata = math.atan2(np.linalg.norm(p), c)
        zrou = theata * u
 
    return zrou

# 计算对应 Jacobian 矩阵
def Jacobian(P, X, Y_real):
    M = (len(P) - 7) // 6
    N = len(X[0])
    K = len(P)
    A = np.array([
        [P[0], P[2], P[3]],
        [0, P[1], P[4]],
        [0, 0, 1]
    ])
 
    res = np.array([])
 
    for i in range(M):
        m = 7 + 6 * i
 
        w = P[m:m + 6]
        R = to_rotation_matrix(w[:3])
        t = w[3:].reshape(3, 1)
        W = np.concatenate((R, t), axis=1)
 
        for j in range(N):
            res = np.append(res, get_single_project_coor(A, W, np.array([P[5], P[6]]), (X[i])[j]))
 
    #求得x, y方向对P[k]的偏导
    J = np.zeros((K, 2 * M * N))
    for k in range(K):
        J[k] = np.gradient(res, P[k])
 
    return J.T


# 主函数
if __name__ == "__main__":

    # 先定义三维空间坐标点，所有图像的空间点的相同
    objp = np.ones((9*6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)  # (9*6, 3)

    # 初始化三维点集和二维点集
    obj_set = []
    obj_x_y_set = []
    img_set = []

    # 得到所有图片的地址列表
    images = glob.glob("D:\Camera Calibration\Project_Stereo_left\left\*.jpg")

    # 利用 OpenCV 库函数找到图像的角点二维坐标，并添加到点集里面
    for fname in images:
        img = cv2.imread(fname)
        ret, corners = cv2.findChessboardCorners(img, (9, 6), None)
        
        if ret:
            corners = corners.reshape(-1, 2)
            img_set.append(corners)
            obj_set.append(objp)
            obj_x_y_set.append(objp[:, :2])
    '''
    # 角点转为 3D 坐标（用于补充维数）
    corners_3D = []
    for i in range(len(corners)):
        corners_3D.append(
            [corners[i][0], corners[i][1], 1]
        )
    corners_3D = np.array(corners_3D)

    # 后续工作应该会用到
    # 将得到的点集格式由(13, 9*6, 3) 或 (13, 9*6, 2) 
    # 改成 (13*9*6, 3) 或 (13*9*6, 2)
    array_img_set = []
    for i in range(len(img_set)):
        for j in range(len(img_set[0])):
            array_img_set.append(img_set[i][j])
    array_img_set = np.array(array_img_set)

    array_obj_set = []
    for i in range(len(obj_set)):
        for j in range(len(obj_set[0])):
            array_obj_set.append(obj_set[i][j])
    array_obj_set = np.array(array_obj_set)

    array_obj_x_y_set = []  
    for i in range(len(array_obj_set)):
        array_obj_x_y_set.append(array_obj_set[i,: 2])
    array_obj_x_y_set = np.array(array_obj_x_y_set)
    '''

    # 注释掉的函数是我自己想的但是同样出现了问题···
    # refine_H(corners, objp, H)
    # avg_H(corners_3D, objp)

    
    # 求单应性矩阵
    H = get_homography(corners, objp)
 
    # 求内参
    intrinsics_param = get_intrinsics_param(H)
 
    # 求对应每幅图外参
    extrinsics_param = get_extrinsics_param(H, intrinsics_param)
 
    # 畸变矫正
    k = get_distortion(intrinsics_param, extrinsics_param, corners, objp)
 
    # 微调所有参数
    [new_intrinsics_param, new_k, new_extrinsics_param]  = refinall_all_param(
        intrinsics_param, k, extrinsics_param, obj_set, img_set
    )
 
    print("intrinsics_parm:\t", new_intrinsics_param)
    print("distortionk:\t", new_k)
    print("extrinsics_parm:\t", new_extrinsics_param)
