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

print('IntrinsicM_l\n', M1)
print('dist_l\n', d1)
print('IntrinsicM_r\n', M2)
print('dist_r\n', d2)
print('R\n', R)
print('T\n', T)
print('E\n', E)
print('F\n', F)
