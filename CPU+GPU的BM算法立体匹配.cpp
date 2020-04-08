#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2\core\cuda.hpp>
#include <opencv2\cudastereo.hpp>
#include <opencv2\cudaimgproc.hpp>
#include <time.h>
#include <iostream>
#include <fstream>
#include <string.h> 
#include <string>

using namespace cv;
using namespace cv::cuda;
using namespace std;

//cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(64, 11);

Ptr<cv::StereoBM> GPU_bm = cuda::StereoBM::create(16, 9);

void match();

int main(int argc, char** argv) {
	clock_t start, finish;
	start = clock();
	match();
	finish = clock();
	cout << "运行时间:" << (double)(finish - start) / CLOCKS_PER_SEC << endl;
	
	
	getchar();

	return 0;
}

void match() {
	ifstream fin_L("D:/Junior to senior related/Camera Calibration/标定左图片.txt");
	ifstream fin_R("D:/Junior to senior related/Camera Calibration/标定右图片.txt");

	int image_count = 0;  // 图像数量 
	Size image_size;  // 图像的尺寸 
	Size board_size = Size(9, 6);      // 标定板上每行、列的角点数 
	// allocate storage
	vector<Point2f> image_points_buf_l;    // 缓存每幅图像上检测到的角点 
	vector<Point2f> image_points_buf_r;

	vector<vector<Point2f>> image_points_seq_l;
	vector<vector<Point2f>> image_points_seq_r;

	string filename_l, filename_r;

	while (getline(fin_L, filename_l)) {
		if (filename_l == "") break;
		//if (image_count > 4) break;
		getline(fin_R, filename_r);
		image_count++;    // 用于观察检验输出
		Mat imageInput_l = imread(filename_l);
		Mat imageInput_r = imread(filename_r);

		if (image_count == 1) {        // 读入第一张图片时获取图像宽高信息
			image_size.width = imageInput_l.cols;
			image_size.height = imageInput_r.rows;
		}

		// 提取角点 （findChessboardCorners 不支持用 cuda 加速）
		findChessboardCorners(imageInput_l, board_size, image_points_buf_l);
		findChessboardCorners(imageInput_r, board_size, image_points_buf_r);

		Mat view_gray_l, view_gray_r;
		// 对左右相机分别提取压像素角点 （也不能用 cuda 加速）
		// 对左
		cv::cvtColor(imageInput_l, view_gray_l, COLOR_BGR2GRAY);
		find4QuadCornerSubpix(view_gray_l, image_points_buf_l, Size(5, 5));   
		image_points_seq_l.push_back(image_points_buf_l); 
		// 对右
		cv::cvtColor(imageInput_r, view_gray_r, COLOR_BGR2GRAY);
		find4QuadCornerSubpix(view_gray_r, image_points_buf_r, Size(5, 5));
		image_points_seq_r.push_back(image_points_buf_r);
		
	}
	// 相机标定的参数准备
	vector<vector<Point3f>> object_points;    // 保存标定板上角点的三维坐标

	// 初始化左右视图内外参数
	Mat cameraMatrix_l = Mat(3, 3, CV_32FC1, Scalar::all(0));    // 内参矩阵
	Mat cameraMatrix_r = Mat(3, 3, CV_32FC1, Scalar::all(0));

	Mat distCoeffs_l = Mat(1, 5, CV_32FC1, Scalar::all(0));    // 5个畸变系数：k1,k2,p1,p2,k3 
	Mat distCoeffs_r = Mat(1, 5, CV_32FC1, Scalar::all(0));

	vector<Mat> tvecsMat_l;              // 每幅图像的旋转向量
	vector<Mat> tvecsMat_r;

	vector<Mat> rvecsMat_l;              // 每幅图像的平移向量
	vector<Mat> rvecsMat_r;

	// 初始化空间点的三维坐标
	int i, j, t;
	for (t = 0; t < image_count; t++) {
		vector<Point3f> tempPointSet;
		for (i = 0; i < board_size.height; i++) {
			for (j = 0; j < board_size.width; j++) {
				Point3f realPoint;
				realPoint.x = i;
				realPoint.y = j;
				realPoint.z = 0;
				tempPointSet.push_back(realPoint);
			}
		}
		object_points.push_back(tempPointSet);
	}

	// 对左右分别标定 （不支持 cuda 加速）
	calibrateCamera(
		object_points, image_points_seq_l,
		image_size, cameraMatrix_l,
		distCoeffs_l, rvecsMat_l, tvecsMat_l, 0);
	calibrateCamera(
		object_points, image_points_seq_r,
		image_size, cameraMatrix_r,
		distCoeffs_r, rvecsMat_r, tvecsMat_r, 0);

	// 立体标定 （不支持 cuda 加速）
	Mat R, T, E, F;
	stereoCalibrate(
		object_points, image_points_seq_l, image_points_seq_r,
		cameraMatrix_l, distCoeffs_l, cameraMatrix_r, distCoeffs_r,
		image_size, R, T, E, F, CALIB_FIX_INTRINSIC,
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-5)
	);
	
	// 立体校正（不支持 cuda 加速）
	Mat R1, R2, P1, P2, Q;
	stereoRectify(
		cameraMatrix_l, distCoeffs_l, cameraMatrix_r, distCoeffs_r,
		image_size, R, T, R1, R2, P1, P2, Q,
		0, 0);

	
	// 校正映射（不支持 cuda 加速）
	Mat mapL_x, mapL_y, mapR_x, mapR_y;

	initUndistortRectifyMap(
		cameraMatrix_l, distCoeffs_l, R1, P1,
		image_size, CV_32FC1, mapL_x, mapL_y);
	initUndistortRectifyMap(
		cameraMatrix_r, distCoeffs_r, R2, P2,
		image_size, CV_32FC1, mapR_x, mapR_y);
    
  // 重映射（CPU版 支持 cuda 加速）
	/*
	String Filename_l = "D:/left/left01.jpg";
	String Filename_r = "D:/right/right01.jpg";
	Mat left01 = imread(Filename_l);
	Mat right01 = imread(Filename_r);
	Mat re_left = left01.clone();
	Mat re_right = right01.clone();

	cv::remap(left01, re_left, mapL_x, mapL_y, INTER_LINEAR);
	cv::remap(right01, re_right, mapR_x, mapR_y, INTER_LINEAR);
	
  // 立体匹配 BM
	
	Mat disp;
	bm->setPreFilterType(cv::StereoBM::PREFILTER_NORMALIZED_RESPONSE);
	bm->setPreFilterSize(9);
	bm->setPreFilterCap(31);
	bm->setBlockSize(21);
	bm->setMinDisparity(0);
	bm->setNumDisparities(64);
	bm->setTextureThreshold(10);
	bm->setUniquenessRatio(5);
	bm->setSpeckleWindowSize(100);
	bm->setSpeckleRange(32);
	// 去黑边
	//cv::copyMakeBorder(re_left, re_left, 0, 0, 80, 0, BORDER_REPLICATE);
	//cv::copyMakeBorder(re_right, re_right, 0, 0, 80, 0, BORDER_REPLICATE);
	cv::cvtColor(re_left, re_left, COLOR_BGR2GRAY);
	cv::cvtColor(re_right, re_right, COLOR_BGR2GRAY);
	bm->compute(re_left, re_right, disp);
	
	disp = disp.colRange(80, re_left.cols);
	//disp.convertTo(disp, CV_32F, 1.0 / 16);
	imshow("disp", disp);
	waitKey(0);
  */
  
  // 重映射 cuda 版
	
	String Filename_l = "D:/left/left01.jpg";
	String Filename_r = "D:/right/right01.jpg";

	Mat left01 = imread(Filename_l);
	Mat right01 = imread(Filename_r);
	GpuMat GPU_left01(left01);
	GpuMat GPU_right01(right01);
	GpuMat GPU_mapL_x(mapL_x);
	GpuMat GPU_mapL_y(mapL_y);
	GpuMat GPU_mapR_x(mapR_x);
	GpuMat GPU_mapR_y(mapR_y);

	Mat re_left = left01.clone();
	Mat re_right = right01.clone();
	GpuMat GPU_re_left(re_left);
	GpuMat GPU_re_right(re_right);

	cuda::remap(GPU_left01, GPU_re_left, GPU_mapL_x, GPU_mapL_y, INTER_LINEAR);
	cuda::remap(GPU_right01, GPU_re_right, GPU_mapR_x, GPU_mapR_y, INTER_LINEAR);
	
  // 立体匹配 cuda 版 BM
	
	Mat disp;
	GpuMat GPU_disp;
  
	GPU_bm->setPreFilterType(cuda::StereoBM::PREFILTER_NORMALIZED_RESPONSE);
	GPU_bm->setPreFilterSize(9);
	GPU_bm->setPreFilterCap(31);
	GPU_bm->setBlockSize(21);
	GPU_bm->setMinDisparity(0);
	GPU_bm->setNumDisparities(64);
	GPU_bm->setTextureThreshold(10);
	GPU_bm->setUniquenessRatio(5);
	GPU_bm->setSpeckleWindowSize(100);
	GPU_bm->setSpeckleRange(32);

	cuda::cvtColor(GPU_re_left, GPU_re_left, COLOR_BGR2GRAY);
	cuda::cvtColor(GPU_re_right, GPU_re_right, COLOR_BGR2GRAY);
	GPU_re_left.download(re_left);
	GPU_re_right.download(re_right);
	GPU_bm->compute(re_left, re_right, disp);

	GPU_disp.download(disp);
	disp = disp.colRange(80, re_left.cols);
  imshow("GPU_disp", disp);
	waitKey(0);
}
