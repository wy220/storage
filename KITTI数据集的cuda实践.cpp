// 运行会报错，应该是我对数据集的使用有问题，在remap()之后，新的图像已经无法显示，自然就不能stereoBM::compute()


#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core\cuda.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <time.h>
#include <iostream>
#include <fstream>
#include <string.h> 
#include <string>

using namespace cv;
using namespace cv::cuda;
using namespace std;

cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(64, 11);

//Ptr<cv::StereoBM> GPU_bm = cuda::StereoBM::create(16, 9);

void KITTI_test() {
	float Size_rect[1][2];
	float K_left[3][3];
	float distCoeffs_left[1][5];
	float R1[3][3];
	float P1[3][4];

	float K_right[3][3];
	float distCoeffs_right[1][5];
	float R2[3][3];
	float P2[3][4];

	FILE *fin;
	fin = fopen("D:/data_scene_flow_calib/training/calib_cam_to_cam/000000.txt", "r");    // 使用的数据参数在我的Github主页，已上传
	double n = 0.0;
	char words[200];
	while (!feof(fin)) {
		fscanf(fin, "%s", words);
		if (strcmp(words, "S_rect_00:") == 0) {
			for (int i = 0; i < 2; i++) {
				fscanf(fin, "%s", words);
				Size_rect[0][i] = (double)atof(words);
			}
		}
		// 对左边彩色相机
		else if (strcmp(words, "K_02:") == 0) {
			for (int i = 0; i < 3; i++) 
				for (int j = 0; j < 3; j++) {
					fscanf(fin, "%s", words);
					K_left[i][j] = (double)atof(words);
				}	
		}

		else if (strcmp(words, "D_02:") == 0) {
			for (int i = 0; i < 5; i++) {
				fscanf(fin, "%s", words);
				distCoeffs_left[0][i] = (double)atof(words);
			}
		}
		else if (strcmp(words, "R_rect_02:") == 0) {
			for (int i = 0; i < 3; i++)
				for (int j = 0; j < 3; j++) {
					fscanf(fin, "%s", words);
					R1[i][j] = (double)atof(words);
				}
		}
		else if (strcmp(words, "P_rect_02:") == 0) {
			for (int i = 0; i < 3; i++)
				for (int j = 0; j < 4; j++) {
					fscanf(fin, "%s", words);
					P1[i][j] = (double)atof(words);
				}
		}
		// 对右边彩色相机
		else if (strcmp(words, "K_03:") == 0) {
			for (int i = 0; i < 3; i++)
				for (int j = 0; j < 3; j++) {
					fscanf(fin, "%s", words);
					K_right[i][j] = (double)atof(words);
				}
		}
		else if (strcmp(words, "D_03:") == 0) {
			for (int i = 0; i < 5; i++) {
				fscanf(fin, "%s", words);
				distCoeffs_right[0][i] = (double)atof(words);
			}
		}
		else if (strcmp(words, "R_rect_03:") == 0) {
			for (int i = 0; i < 3; i++)
				for (int j = 0; j < 3; j++) {
					fscanf(fin, "%s", words);
					R2[i][j] = (double)atof(words);
				}
		}
		else if (strcmp(words, "P_rect_03:") == 0) {
			for (int i = 0; i < 3; i++)
				for (int j = 0; j < 4; j++) {
					fscanf(fin, "%s", words);
					P2[i][j] = (double)atof(words);
				}
		}
	}

	Mat image_size = Mat(Size(2, 1), CV_32FC1, Size_rect);    // Size(1, 2) is .t()
	Size imageSize;
	imageSize.width = image_size.cols;
	imageSize.height = image_size.rows;

	Mat cameraMatrix_left = Mat(Size(3, 3), CV_32FC1, K_left);
	Mat distort_left = Mat(Size(5, 1), CV_32FC1, distCoeffs_left);
	Mat R_left = Mat(Size(3, 3), CV_32FC1, R1);
	Mat P_left = Mat(Size(4, 3), CV_32FC1, P1);

	Mat cameraMatrix_right = Mat(Size(3, 3), CV_32FC1, K_right);
	Mat distort_right = Mat(Size(5, 1), CV_32FC1, distCoeffs_right);
	Mat R_right = Mat(Size(3, 3), CV_32FC1, R2);
	Mat P_right = Mat(Size(4, 3), CV_32FC1, P2);

	
	Mat mapL_x, mapL_y, mapR_x, mapR_y;

	initUndistortRectifyMap(
		cameraMatrix_left, distort_left, R_left, P_left,
		imageSize, CV_32FC1, mapL_x, mapL_y);
	initUndistortRectifyMap(
		cameraMatrix_right, distort_right, R_right, P_right,
		imageSize, CV_32FC1, mapR_x, mapR_y);
	
	String Filename_l = "D:/data_scene_flow/training/image_2/000000_10.png";
	String Filename_r = "D:/data_scene_flow/training/image_2/000000_11.png";

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
	GPU_bm->compute(re_left, re_right, disp);    // 出错 OpenCV(4.1.0) Error: One of arguments' values is out of range (SADWindowSize must be odd, be within 5..255 and be not larger than image width or height) in cv::StereoBMImpl::compute
 
	GPU_disp.download(disp);
	disp = disp.colRange(80, re_left.cols);
	disp.convertTo(disp, CV_32F, 1.0 / 16);
	imshow("GPU_disp", disp);
	waitKey(0);

}


int main(int argc, char** argv) {
	clock_t start, finish;
	start = clock();
	KITTI_test();
	finish = clock();
	cout << "运行时间:" << (double)(finish - start) / CLOCKS_PER_SEC << endl;
	
	getchar();
	return 0;
}
