#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace cv::cuda;
using namespace std;

void single_calibration();
void SGBMUpdate(int pos, void* data);
void stereo_match();


int main(int argc, char** argv) {
	//single_calibration();
	stereo_match();


	getchar();

	return 0;
}


void SGBMUpdate(int pos, void* data) {
	int SGBMNum = 2;
	int blockSize = cv::getTrackbarPos("blockSize", "SGBM_disparity");
	if (blockSize % 2 == 0) {
		blockSize += 1;
	}
	if (blockSize < 5) {
		blockSize = 5;
	}
	SGBM->setBlockSize(blockSize);
	SGBM->setNumDisparities(cv::getTrackbarPos("numDisparities", "SGBM_disparity"));
	SGBM->setSpeckleWindowSize(cv::getTrackbarPos("speckleWindowSize", "SGBM_disparity"));
	SGBM->setSpeckleRange(cv::getTrackbarPos("speckleRange", "SGBM_disparity"));
	SGBM->setUniquenessRatio(cv::getTrackbarPos("uniquenessRatio", "SGBM_disparity"));
	SGBM->setDisp12MaxDiff(cv::getTrackbarPos("disp12MaxDiff", "SGBM_disparity"));
	SGBM->setP1(600);
	SGBM->setP2(2400);
	SGBM->setMode(cv::StereoSGBM::MODE_SGBM);
}

// 单目标定
void single_calibration() {
	ifstream fin("D:/标定图片地址.txt");    // TXT文档存储了图片的地址
	int image_count = 0;  // 图像数量 
	Size image_size;  // 图像的尺寸 
	Size board_size = Size(9, 6);      // 标定板上每行、列的角点数 
  // allocate storage
	vector<Point2f> image_points_buf;    // 缓存每幅图像上检测到的角点 
	vector<vector<Point2f>> image_points_seq;   // 保存检测到的所有角点 
	string filename;

	while (getline(fin, filename)) {
		if (filename == "") break;
		image_count++;    
		Mat imageInput = imread(filename);

		if (image_count == 1) {        // 读入第一张图片时获取图像宽高信息
			image_size.width = imageInput.cols;
			image_size.height = imageInput.rows;
			//cout << "image_size.width = " << image_size.width << endl;
			//cout << "image_size.height = " << image_size.height << endl;
		}
		// 提取角点 
		if (0 == findChessboardCorners(imageInput, board_size, image_points_buf)) {
			cout << "can not find chessboard corners!\n";    // 找不到角点
			exit(1);
		}
		else {
			Mat view_gray;
			cv::cvtColor(imageInput, view_gray, COLOR_BGR2GRAY);
			find4QuadCornerSubpix(view_gray, image_points_buf, Size(5, 5));    // 亚像素精确化
			image_points_seq.push_back(image_points_buf);    // 把精确后的点添加到像点矩阵里
			//drawChessboardCorners(view_gray, board_size, image_points_buf, false);    // 在图片中标记角点
			//imshow("Camera Calibration", view_gray);
			//waitKey(0);
		}
	}

	// 相机标定的参数准备
	vector<vector<Point3f>> object_points;    // 保存标定板上角点的三维坐标

	// 初始化内外参数
	Mat cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));    // 内参矩阵
	vector<int> point_counts;                                // 每幅图像中角点的数量
	Mat distCoeffs = Mat(1, 5, CV_32FC1, Scalar::all(0));    // 摄像机的5个畸变系数：k1,k2,p1,p2,k3 */
	vector<Mat> tvecsMat;                                    // 每幅图像的旋转向量
	vector<Mat> rvecsMat;                                    // 每幅图像的平移向量
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
  
	// 标定
	calibrateCamera(
		object_points, image_points_seq, 
		image_size, cameraMatrix, 
		distCoeffs, rvecsMat, tvecsMat, 0);
  
  // 初始化每幅图像中的角点数量
	for (i = 0; i < image_count; i++) 
		point_counts.push_back(board_size.width * board_size.height);
    
  // 评价标定结果
	
	double total_err = 0.0;    // 所有图像的平均误差的总和 
	double err = 0.0;          // 每幅图像的平均误差
	vector<Point2f> image_points2;    // 保存重新计算得到的投影点
	cout << "\t每幅图像的标定误差：\n";
	fout << "每幅图像的标定误差：\n";
	for (i = 0; i < image_count; i++) {
		vector<Point3f> tempPointSet = object_points[i];
		// 通过得到的摄像机内外参数，对空间的三维点进行重新投影计算，得到新的投影点 
		projectPoints(
			tempPointSet, rvecsMat[i], tvecsMat[i], 
			cameraMatrix, distCoeffs, image_points2);
		// 计算新的投影点和旧的投影点之间的误差,z这个标定结果反应的是标定算法的好坏
		vector<Point2f> tempImagePoint = image_points_seq[i];
		Mat tempImagePointMat = Mat(1, tempImagePoint.size(), CV_32FC2);
		Mat image_points2Mat = Mat(1, image_points2.size(), CV_32FC2);
		for (int j = 0; j < tempImagePoint.size(); j++) {
			image_points2Mat.at<Vec2f>(0, j) = Vec2f(image_points2[j].x, image_points2[j].y);
			tempImagePointMat.at<Vec2f>(0, j) = Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
		}
		err = cv::norm(image_points2Mat, tempImagePointMat, NORM_L2);
		total_err += err /= point_counts[i];
		cout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
		fout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
	}
	cout << "总体平均误差：" << total_err / image_count << "像素" << endl;
	fout << "总体平均误差：" << total_err / image_count << "像素" << endl << endl;
  
  // 显示标定结果
	
	Mat mapx = Mat(image_size, CV_32FC1);
	Mat mapy = Mat(image_size, CV_32FC1);
	Mat R = Mat::eye(3, 3, CV_32F);     // eye 函数表示的是单位矩阵

	cout << "保存矫正图像" << endl;
	String imageFileName;
	stringstream StrStm;
	i = -1;
	fin.close();
	fin.open("D:/标定图片地址.txt");
	while (getline(fin, filename)) {
		if (filename == "") break;
		i++;
		initUndistortRectifyMap(
			cameraMatrix, distCoeffs, R, 
			cameraMatrix, image_size, CV_32FC1, 
			mapx, mapy);
		StrStm.clear();
		Mat imageSource = imread(filename);
		Mat newimage = imageSource.clone();
		try {
			cv::remap(imageSource, newimage, mapx, mapy, INTER_LINEAR);
		}
		catch (Exception e) {
			cout << e.what() << endl;
		}
		StrStm.clear();
		char* fullname = (char*)filename.data();
		const char* b = ".";
		imageFileName = strtok(fullname, b);
		imageFileName += "_d.jpg";
		cout << imageFileName << endl;
		imwrite(imageFileName, newimage);    // 写入图片
		//imshow("resultImage", newimage);
		//waitKey(0);
	}
    
}

// 双目标定+匹配
void stereo_match() {
	ifstream fin_L("D:/标定左图片.txt");
	ifstream fin_R("D:/标定右图片.txt");

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
		if (image_count > 4) break;
		getline(fin_R, filename_r);
		image_count++;    // 用于观察检验输出
		Mat imageInput_l = imread(filename_l);
		Mat imageInput_r = imread(filename_r);

		if (image_count == 1) {        // 读入第一张图片时获取图像宽高信息
			image_size.width = imageInput_l.cols;
			image_size.height = imageInput_r.rows;
		}

		// 提取角点 
		findChessboardCorners(imageInput_l, board_size, image_points_buf_l);
		findChessboardCorners(imageInput_r, board_size, image_points_buf_r);

		Mat view_gray_l, view_gray_r;
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

	// 对左右分别标定
	calibrateCamera(
		object_points, image_points_seq_l,
		image_size, cameraMatrix_l,
		distCoeffs_l, rvecsMat_l, tvecsMat_l, 0);
	calibrateCamera(
		object_points, image_points_seq_r,
		image_size, cameraMatrix_r,
		distCoeffs_r, rvecsMat_r, tvecsMat_r, 0);

	// 立体标定
	Mat R, T, E, F;
	stereoCalibrate(
		object_points, image_points_seq_l, image_points_seq_r,
		cameraMatrix_l, distCoeffs_l, cameraMatrix_r, distCoeffs_r,
		image_size, R, T, E, F, CALIB_FIX_INTRINSIC,
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-5)
	);

	// 立体校正
	Mat R1, R2, P1, P2, Q;
	stereoRectify(
		cameraMatrix_l, distCoeffs_l, cameraMatrix_r, distCoeffs_r,
		image_size, R, T, R1, R2, P1, P2, Q,
		0, 0);

	// 校正映射
	Mat mapL_x, mapL_y, mapR_x, mapR_y;
	initUndistortRectifyMap(
		cameraMatrix_l, distCoeffs_l, R1, P1,
		image_size, CV_32FC1, mapL_x, mapL_y);
	initUndistortRectifyMap(
		cameraMatrix_r, distCoeffs_r, R2, P2,
		image_size, CV_32FC1, mapR_x, mapR_y);

	// 重映射
	String Filename_l = "D:/left/left01.jpg";
	String Filename_r = "D:/right/right01.jpg";

	Mat left01 = imread(Filename_l);
	Mat right01 = imread(Filename_r);

	Mat re_left = left01.clone();
	Mat re_right = right01.clone();

	cv::remap(left01, re_left, mapL_x, mapL_y, INTER_LINEAR);
	cv::remap(right01, re_right, mapR_x, mapR_y, INTER_LINEAR);
  
  // 立体匹配 SGBM
	
	int minDisparity = 1;    // 最小视差值
	int SGBMNum = 2;
	int numDisparities = SGBMNum * 16;    
	int blockSize = 5;                 // 匹配块大小，大于1的奇数

	int p1 = 600;    
	int p2 = 2400;
	int disp12MaxDiff = -1;
	int preFilterCap = 1;
	int uniquenessRatio = 10;    	
	int speckleWindowSize = 100;    	
	int speckleRange = 2;    

	cv::createTrackbar("blockSize", "SGBM_disparity", &blockSize, 21, SGBMUpdate);
	cv::createTrackbar("numDisparities", "SGBM_disparity", &numDisparities, 20, SGBMUpdate);
	cv::createTrackbar("speckleWindowSize", "SGBM_disparity", &speckleWindowSize, 200, SGBMUpdate);
	cv::createTrackbar("speckleRange", "SGBM_disparity", &speckleRange, 50, SGBMUpdate);
	cv::createTrackbar("uniquenessRatio", "SGBM_disparity", &uniquenessRatio, 50, SGBMUpdate);
	cv::createTrackbar("disp12MaxDiff", "SGBM_disparity", &disp12MaxDiff, 21, SGBMUpdate);

	Mat disp;
	SGBM->compute(re_left, re_right, disp);
	Mat disp8U = Mat(disp.rows, disp.cols, CV_8UC1);
	cv::normalize(disp, disp8U, 0, 255, NORM_MINMAX, CV_8UC1);
	imwrite("D:/disp.jpg", disp);
	imshow("SGBM_disparity", disp8U);

	waitKey(0);
}
