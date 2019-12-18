# 主函数从 252 行开始
# 只做到修正畸变之前，因为最小二乘使用雅各比矩阵运行出现异常，没得到内参结果

import numpy as np 
from scipy import optimize as opt
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
    
# LM 算法 进行极大似然估计，但 leastsq 函数和我得到的雅各比矩阵一起使用报错，我也没找到合适的解决办法
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

# 得到相机内参矩阵 A
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


# 主函数


# 先定义三维空间坐标点，所有图像的空间点的相同
objp = np.ones((9*6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)  # (9*6, 3)

# 初始化三维点集和二维点集
obj_set = []
obj_x_y_set = []
img_set = []

# 得到所有图片的地址列表
images = glob.glob("D:\Project_Camera Calibration\Project_Stereo_left\left\*.jpg")

# 利用 OpenCV 库函数找到图像的角点二维坐标，并添加到点集里面
for fname in images:
    img = cv2.imread(fname)
    ret, corners = cv2.findChessboardCorners(img, (9, 6), None)

    if ret:
        corners = corners.reshape(-1, 2)
        img_set.append(corners)

        obj_set.append(objp)
        obj_x_y_set.append(objp[:, :2])

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

# 注释掉的函数是我自己想的但是同样出现了问题···
# refine_H(corners, objp, H)
# avg_H(corners_3D, objp)


refined_H = get_homography(corners, objp)
intrinsicsMatrix = get_intrinsics_param(refined_H)
