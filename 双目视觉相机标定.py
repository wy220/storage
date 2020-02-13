import cv2
import numpy as np
import glob

# 初始化标定板角点的空间坐标
objp = np.zeros((9*6, 3), np.float32)
# 将世界坐标系建在标定板上，张氏标定，所有点的 z = 0，所以只需要赋值 x 和 y
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)  

# 设置亚像素角点的参数，采用的停止准则是最大循环次数 30 或最大误差容限 0.001
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
criteria_cal = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

obj_points = [] # 存储物体 3D 坐标
img_points_l = [] # 存储左边相机 2D 坐标
img_points_r = [] # 存储右边相机 2D 坐标

img_path = 'D:\Camera Calibration'
images_left  = glob.glob(img_path + '\Project_Stereo_left\left\*.jpg')
images_right = glob.glob(img_path + '\Project_Stereo_right\\right\*.jpg')

for i in range(len(images_left)):
    img_l = cv2.imread(images_left[i])
    img_r = cv2.imread(images_right[i])
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, (9, 6), None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, (9, 6), None)

    obj_points.append(objp)

    if ret_l:
        corners2_l = cv2.cornerSubPix(
            gray_l, corners_l, (5, 5), (-1, -1), criteria
        )
        if [corners2_l]:
            img_points_l.append(corners2_l)
        else:
            img_points_l.append(corners_l)
    
    if ret_r:
        corners2_r = cv2.cornerSubPix(
            gray_r, corners_r, (5, 5), (-1, -1), criteria
        )
        if [corners2_r]:
            img_points_r.append(corners2_r)
        else:
            img_points_r.append(corners_r)

size = gray_l.shape[::-1]

rt_l, intrinsicsM_l, distortCoef_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
    obj_points, img_points_l, size, None, None
)
rt_r, intrinsicsM_r, distortCoef_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
    obj_points, img_points_r, size, None, None
)

stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
cv2.TERM_CRITERIA_EPS, 100, 1e-5)

ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
    obj_points, img_points_l, img_points_r, intrinsicsM_l, distortCoef_l, intrinsicsM_r, distortCoef_r, size, criteria=stereocalib_criteria, flags=0
)

'''
print('IntrinsicM_l\n', M1)
print('dist_l\n', d1)
print('IntrinsicM_r\n', M2)
print('dist_r\n', d2)
print('R\n', R)
print('T\n', T)
print('E\n', E)
print('F\n', F)
'''

# 第 14 题
# 使用 Bouguet 标定算法
R1, R2, P1, P2, Q ,validPixROI1, validPixROI2 = cv2.stereoRectify(M1, d1, M2, d2, size, R, T)


# 校正映射

# 对左边相机图像
map1_l, map2_l = cv2.initUndistortRectifyMap(M1, d1, R1, P1, size, cv2.CV_16SC2)

# 对右边相机图像
map1_r, map2_r = cv2.initUndistortRectifyMap(M2, d2, R2, P2, size, cv2.CV_16SC2)

left01 = cv2.imread('left01.jpg')
right01 = cv2.imread('right01.jpg')
# 重映射
left01_re = cv2.remap(left01, map1_l, map2_l, cv2.INTER_LINEAR)
right01_re = cv2.remap(right01, map1_r, map2_r, cv2.INTER_LINEAR)
# 写入图片
cv2.imwrite("left01_re.jpg", left01_re)
cv2.imwrite("right01_re.jpg", right01_re)


# 第 17 题
# 计算视差图
for i in [4, 8, 16]:
    for j in [1, 3, 5]:
        blockSize = i
        img_channels = j
        stereo = cv2.StereoSGBM_create(
            minDisparity = 1,
            numDisparities = 64,
            blockSize = blockSize,
            P1 = 8 * img_channels * blockSize * blockSize,
            P2 = 32 * img_channels * blockSize * blockSize,
            disp12MaxDiff = -1,
            preFilterCap = 1,
            uniquenessRatio = 10,
            speckleWindowSize = 100,
            speckleRange = 100,
            mode = cv2.STEREO_SGBM_MODE_HH
        )

        disp = stereo.compute(left01_re, right01_re)
        # 除以 16 得到真实视差图
        disp = np.divide(disp.astype(np.float32), 16.)    
        cv2.imwrite("disp" + str(i) + str(j) + ".jpg", disp)
