import numpy as np
import cv2
import glob


# 初始化标定板角点的空间坐标
objp = np.zeros((9*6, 3), np.float32)
# 将世界坐标系建在标定板上，张氏标定，所有点的 z = 0，所以只需要赋值 x 和 y
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)  

obj_points = [] # 存储 3D 坐标
img_points = [] # 存储 2D 坐标

# 得到所有图片的地址数组
images = glob.glob("D:\Project_Stereo\Project_Stereo_left\left\*.jpg")

# 初始化生成图片标号
i = 0

# 设置亚像素角点的参数，采用的停止准则是最大循环次数 30 或最大误差容限 0.001
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1] # 输出列和行
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None) # ret 为真假标识符，corners 为角点坐标

    # 向 3D 和 2D 存储数组添加坐标
    if ret:
        obj_points.append(objp)
        corners2 = cv2.cornerSubPix(
            gray, corners, (5, 5), (-1, -1), criteria
            )  # 在原角点的基础上计算亚像素角点坐标
        if [corners2]:
            img_points.append(corners2)
        else:
            img_points.append(corners)
        
        cv2.drawChessboardCorners(img, (9, 6), corners, ret)  
        i += 1
        cv2.imwrite('conimg'+str(i)+'.jpg', img)
        cv2.waitKey(150)
        
cv2.destroyAllWindows()

# 相机标定
ret,  intrinsicsM, distortCoef, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
'''
print("ret:", ret)
print("intrinsicsMatrix = ")
print(intrinsicsM)
print("distortCoef = ", distortCoef)
print("rvecs = ")
print(rvecs)
print("tvecs = ")
print(tvecs)
'''
# 修正畸变
img = cv2.imread(images[0])
width, lenth = img.shape[: 2]
newcameraMatrix, roi = cv2.getOptimalNewCameraMatrix(
    intrinsicsM, distortCoef, (width, lenth), 1, (width, lenth)
)
distort = cv2.undistort(img, intrinsicsM, distortCoef, None, newcameraMatrix)
cv2.imwrite("testing.jpg", distort)
x, y, width, lenth = roi
dst = distort[y: y+lenth, x: x+width]
cv2.imwrite("calibresult0.jpg", dst)
