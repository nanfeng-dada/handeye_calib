/*
eye to hand 手眼标定
*/

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

//opencv
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>

#include<vector>
#include<cmath>
#include<fstream>
#include<sstream>
#include <iostream>
#include <fstream>
#include <functional>   // std::minus
#include <numeric>      // std::accumulate

//our files
#include "handeye.h"
#include "quaternion.h"




using namespace std;
using namespace Eigen;
using namespace cv;




//棋盘格参数
#define dGRID_WIDTH_OF_NUM 11
#define dGRID_HEIGHT_OF_NUM 8
#define dGRID_MM 7

#define dPI 3.1415926


class CRAngleInfo {
public:
	double rx;
	double ry;
	double rz;
};

enum HandEyeCalibrationMethod
{
	CALIB_HAND_EYE_TSAI = 0, //!< A New Technique for Fully Autonomous and Efficient 3D Robotics Hand/Eye Calibration @cite Tsai89
	CALIB_HAND_EYE_PARK = 1, //!< Robot Sensor Calibration: Solving AX = XB on the Euclidean Group @cite Park94
	CALIB_HAND_EYE_HORAUD = 2, //!< Hand-eye Calibration @cite Horaud95
	CALIB_HAND_EYE_ANDREFF = 3, //!< On-line Hand-Eye Calibration @cite Andreff99
	CALIB_HAND_EYE_DANIILIDIS = 4  //!< Hand-Eye Calibration Using Dual Quaternions @cite Daniilidis98
};


static std::vector<Point2f> CRAlgoSortCornerPoints(std::vector<Point2f> corners)
{
	int i, num;

	std::vector<Point2f> output;

	num = corners.size();

	if ((corners[num - 1].x < corners[0].x)
		&& (corners[num - 1].y < corners[0].y))
	{
		for (i = 0; i < num; i++)
		{
			output.push_back(corners[(num - 1) - i]);
		}
	}
	else
	{
		for (i = 0; i < num; i++)
		{
			output.push_back(corners[i]);
		}
	}

	return output;
}

/*

找棋盘格点
*/
int CRAlgoTestChessBoard(Mat             		srcImg,
	std::vector<Point2f>*	ImageCorners)
{
	int exitCode = dFALSE,
		i;

	std::vector<Point2f> corners,
		sortingCorners;

	bool found = false;

	Size grids(dGRID_WIDTH_OF_NUM, dGRID_HEIGHT_OF_NUM); //number of centers

	try
	{
		found = findChessboardCorners(srcImg, grids, corners, /*CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + */CALIB_CB_FAST_CHECK);
		drawChessboardCorners(srcImg, grids, corners, found);
		namedWindow("found", 0);//创建窗口
		resizeWindow("found", 800, 800); //创建一个500*500大小的窗口
		imshow("found", srcImg);
		while (1)
		{
			if (waitKey() != -1)break;
		}
		if (found)
		{
			Mat     gray;

			cvtColor(srcImg, gray, COLOR_BGR2GRAY);

			cornerSubPix(gray, corners, Size(3, 3), Size(-1, -1),
				TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 330, 0.000001));//0.1 is accuracy

			gray.release();

			sortingCorners = CRAlgoSortCornerPoints(corners);

			for (i = 0; i < (int)sortingCorners.size(); i++)
			{
				ImageCorners->push_back(sortingCorners[i]);
			}

			std::vector<Point2f>().swap(corners);

		}
	}
	catch (cv::Exception& e)
	{
		const char* err_msg = e.what();
		std::cout << "exception caught: " << err_msg << std::endl;
		exitCode = dTRUE;
	}

	return exitCode;
}


static std::vector<std::vector<Point3f> > CRAlgoCalcBoardCornerPositionsList(int gridW, int gridH, double squareSize, int imagesCount)
{
	std::vector<std::vector<Point3f> > objectPointsList(imagesCount);
	for (int k = 0; k <imagesCount; k++) {
		objectPointsList[k] = std::vector<Point3f>(0);
		for (int i = 0; i < gridH; i++)
			for (int j = 0; j < gridW; j++)
				objectPointsList[k].push_back(Point3f(double(j*squareSize), double(i*squareSize), 0));
	}
	return objectPointsList;
}


int CRAlgoCalcCameraMat(int                               gridSize,
	std::vector<std::vector<Point2f>> pointList,
	Mat&                              cameraMatrix,
	Mat&                              distCoeffs,
	std::vector<Mat>*                 rmatCamObjectList,
	std::vector<Mat>*                 tvecsCamObjectList
	)
{
	// The exit code of the sample application.
	int exitCode = dFALSE;

	double  rms = 0.0;

	//number of centers
	Size grids(dGRID_WIDTH_OF_NUM, dGRID_HEIGHT_OF_NUM);

	try
	{
		std::vector<std::vector<Point3f>> objectList;

		cv::Mat rvecsCamList;
		cv::Mat tvecsCamList;

		ofstream ofilee("HandEyeCalibrate\\camera_T.txt");

		objectList = CRAlgoCalcBoardCornerPositionsList(grids.width, grids.height, gridSize, pointList.size());
		for (int i = 0; i < (int)pointList.size(); i++)
		{
		  cv::solvePnP(objectList[i], pointList[i], cameraMatrix, distCoeffs, rvecsCamList, tvecsCamList, false, SOLVEPNP_ITERATIVE);
		  
			Mat R(3, 3, CV_64FC1);

			Rodrigues(rvecsCamList, R);

			rmatCamObjectList->push_back(R);

			ofilee << R.at<double>(0, 0) << " " << R.at<double>(0, 1) << " " << R.at<double>(0, 2) << " " << tvecsCamList.at<double>(0, 0) / 1000.0f << endl;
			ofilee << R.at<double>(1, 0) << " " << R.at<double>(1, 1) << " " << R.at<double>(1, 2) << " " << tvecsCamList.at<double>(1, 0) / 1000.0f << endl;
			ofilee << R.at<double>(2, 0) << " " << R.at<double>(2, 1) << " " << R.at<double>(2, 2) << " " << tvecsCamList.at<double>(2, 0) / 1000.0f << endl;
			ofilee << 0 << " " << 0 << " " << 0 << " " << 1 << endl;
			R.release();
			tvecsCamObjectList->push_back(tvecsCamList.clone()/1000.0f);
		}

		std::vector<std::vector<Point3f>>().swap(objectList);

		if (rms > 1.0)
		{
			std::cout << "rms is too large:" << rms << std::endl;
		}
	}
	catch (cv::Exception& e)
	{
		const char* err_msg = e.what();
		std::cout << "exception caught: " << err_msg << std::endl;
		exitCode = dTRUE;
	}
	cout << tvecsCamObjectList->size() << endl;
	return exitCode;
}



int CRAlgoCalcAngleToRmat(CRAngleInfo 	angle,
	Mat&        	R)
{
	int exitCode = dFALSE;

	double tempXTheta = 0.0,
		tempYTheta = 0.0,
		tempZTheta = 0.0;
	
	try
	{
		Mat tempRx(3, 3, CV_64FC1),
			tempRy(3, 3, CV_64FC1),
			tempRz(3, 3, CV_64FC1),
			tempR;

	//	tempXTheta = (angle.rx  * dPI) / 180;
	//	tempYTheta = (angle.ry  * dPI) / 180;
	//	tempZTheta = (angle.rz  * dPI) / 180;   //eye in hand


		tempXTheta = angle.rx ;
		tempYTheta = angle.ry ;
		tempZTheta = angle.rz ;                 //eye to hand


		tempRx.at<double>(0, 0) = 1;
		tempRx.at<double>(0, 1) = 0;
		tempRx.at<double>(0, 2) = 0;
		tempRx.at<double>(1, 0) = 0;
		tempRx.at<double>(1, 1) = cos(tempXTheta);
		tempRx.at<double>(1, 2) = -sin(tempXTheta);
		tempRx.at<double>(2, 0) = 0;
		tempRx.at<double>(2, 1) = sin(tempXTheta);
		tempRx.at<double>(2, 2) = cos(tempXTheta);

		tempRy.at<double>(0, 0) = cos(tempYTheta);
		tempRy.at<double>(0, 1) = 0;
		tempRy.at<double>(0, 2) = sin(tempYTheta);
		tempRy.at<double>(1, 0) = 0;
		tempRy.at<double>(1, 1) = 1;
		tempRy.at<double>(1, 2) = 0;
		tempRy.at<double>(2, 0) = -sin(tempYTheta);
		tempRy.at<double>(2, 1) = 0;
		tempRy.at<double>(2, 2) = cos(tempYTheta);

		tempRz.at<double>(0, 0) = cos(tempZTheta);
		tempRz.at<double>(0, 1) = -sin(tempZTheta);
		tempRz.at<double>(0, 2) = 0;
		tempRz.at<double>(1, 0) = sin(tempZTheta);
		tempRz.at<double>(1, 1) = cos(tempZTheta);
		tempRz.at<double>(1, 2) = 0;
		tempRz.at<double>(2, 0) = 0;
		tempRz.at<double>(2, 1) = 0;
		tempRz.at<double>(2, 2) = 1;

		tempR = tempRz * tempRy * tempRx;
		tempR.copyTo(R);

		tempRz.release();
		tempRy.release();
		tempRx.release();
		tempR.release();
	}
	catch (cv::Exception& e)
	{
		const char* err_msg = e.what();
		std::cout << "exception caught: " << err_msg << std::endl;
		exitCode = dTRUE;
	}

	return exitCode;
}


/*读入机器人末端数据*/
int CRAlgoCalcWorldEndEffectorMat(
	std::vector<Mat>*         tvecsWorldEndEffectorList,
	std::vector<Mat>*         rmatWorldEndEffectorList)
{
	int exitCode = dFALSE;
	//ifstream ifile("HandEyeCalibrate\\ss\\robot.txt", ios::in);
	//ofstream ofile("HandEyeCalibrate\\ss\\robot_T.txt");
	ifstream ifile("HandEyeCalibrate\\3c\\1112\\robot.txt", ios::in);
	ofstream ofile("HandEyeCalibrate\\3c\\1112\\robot_T.txt");
	string strr;
	istringstream ostr;
	string lefted;
	int i = 0;
	try
	{
		Mat rmatWorldEndEffector(3, 3, CV_64F),
			tvecsWorldEndEffector(3, 1, CV_64F);
		while (getline(ifile, strr))
		{
			i++;
			ostr.clear();
			ostr.str(strr);
			ostr >> tvecsWorldEndEffector.at<double>(0, 0);
			ostr >> tvecsWorldEndEffector.at<double>(1, 0);
			ostr >> tvecsWorldEndEffector.at<double>(2, 0);

			CRAngleInfo angle;
			ostr >> angle.rx;
			ostr >> angle.ry;
			ostr >> angle.rz;

			CRAlgoCalcAngleToRmat(angle,
				rmatWorldEndEffector);
			//cout << rmatWorldEndEffector << endl;
		//	Mat homogeneousMatt(4, 4, CV_64F);
		//	CRAlgoCalcHomogeneousMat(tvecsWorldEndEffector, rmatWorldEndEffector, homogeneousMatt);
			ofile << rmatWorldEndEffector.at<double>(0, 0) << " "<<rmatWorldEndEffector.at<double>(0, 1) <<" "<< rmatWorldEndEffector.at<double>(0, 2) <<" "<< tvecsWorldEndEffector.at<double>(0, 0)/1000.0f<<endl;
			ofile << rmatWorldEndEffector.at<double>(1, 0)<<" " << rmatWorldEndEffector.at<double>(1, 1)<<" " << rmatWorldEndEffector.at<double>(1, 2) <<" "<< tvecsWorldEndEffector.at<double>(1, 0)/1000.0f <<endl;
			ofile << rmatWorldEndEffector.at<double>(2, 0)<<" " << rmatWorldEndEffector.at<double>(2, 1)<<" " << rmatWorldEndEffector.at<double>(2, 2)<<" " << tvecsWorldEndEffector.at<double>(2, 0)/1000.0f <<endl;
			ofile <<0 <<" "<< 0 <<" "<< 0<<" " << 1<<endl;

		/*	if(i%5==1)
				ostr >> rmatWorldEndEffector.at<double>(0, 0), ostr >> rmatWorldEndEffector.at<double>(0, 1), ostr >> rmatWorldEndEffector.at<double>(0, 2), ostr >> tvecsWorldEndEffector.at<double>(0, 0);
			else if (i % 5 == 2)
			    ostr >> rmatWorldEndEffector.at<double>(1, 0), ostr >> rmatWorldEndEffector.at<double>(1, 1), ostr >> rmatWorldEndEffector.at<double>(1, 2), ostr >> tvecsWorldEndEffector.at<double>(1, 0);
			else if (i % 5 == 3)
			    ostr >> rmatWorldEndEffector.at<double>(2, 0), ostr >> rmatWorldEndEffector.at<double>(2, 1), ostr >> rmatWorldEndEffector.at<double>(2, 2), ostr >> tvecsWorldEndEffector.at<double>(2, 0);
			else if (i % 5 == 4)
				ostr >> lefted;
			else if (i % 5 == 0)
			{
				ostr >> lefted;
				rmatWorldEndEffectorList->push_back(rmatWorldEndEffector.clone());
				tvecsWorldEndEffectorList->push_back(tvecsWorldEndEffector.clone()/1000.0f);
			}*/
			rmatWorldEndEffectorList->push_back(rmatWorldEndEffector.clone());
			tvecsWorldEndEffectorList->push_back(tvecsWorldEndEffector.clone() / 1000.0f);
		}
	}
	catch (cv::Exception& e)
	{
		const char* err_msg = e.what();
		std::cout << "exception caught: " << err_msg << std::endl;
		exitCode = dTRUE;
	}

	return exitCode;
}




/*
手眼标函数
*/
int CRAlgoCalcEndEffectorCamMat(std::vector<Mat> 	tvecsWorldEndEffectorList,
	std::vector<Mat> 	rmatWorldEndEffectorList,
	std::vector<Mat> 	tvecsCamObjectList,
	std::vector<Mat> 	rmatCamObjectList,
	std::vector<Mat> 	rmatWorldCam,
	std::vector<Mat> 	tmatWorldCam
	
	)
{
	int exitCode = dFALSE,
		i = 0;
	cout << tvecsWorldEndEffectorList.size() << "  " << tvecsCamObjectList.size() << endl;
	CV_Assert(tvecsWorldEndEffectorList.size() == tvecsCamObjectList.size());
	CV_Assert(rmatWorldEndEffectorList.size() == rmatCamObjectList.size());
	Mat R_cam2gripper; Mat t_cam2gripper;


	try
	{
		std::vector<Mat>  wMeList,
			cMoList,
			refeMeList,
			refcMcList;

		Mat eMc(4, 4, CV_64FC1);

		for (i = 0; i < (int)rmatCamObjectList.size(); i++)
		{
			Mat wMe(4, 4, CV_64FC1),
				cMo(4, 4, CV_64FC1);

			exitCode = CRAlgoCalcHomogeneousMat(tvecsWorldEndEffectorList[i],
				rmatWorldEndEffectorList[i],
				wMe);

			if (exitCode)
			{
				return dTRUE;
			}

			exitCode = CRAlgoCalcHomogeneousMat(tvecsCamObjectList[i],
				rmatCamObjectList[i],
				cMo);

			if (exitCode)
			{
				return dTRUE;
			}

			wMeList.push_back(wMe);
			cMoList.push_back(cMo);

			cout << "wMe" << wMe << endl;
			cout << "cMo" << cMo << endl;

			wMe.release();
			cMo.release();
		}

		//CALIB_HAND_EYE_TSAI = 0, //!< A New Technique for Fully Autonomous and Efficient 3D Robotics Hand/Eye Calibration @cite Tsai89
		//	CALIB_HAND_EYE_PARK = 1, //!< Robot Sensor Calibration: Solving AX = XB on the Euclidean Group @cite Park94
		//	CALIB_HAND_EYE_HORAUD = 2, //!< Hand-eye Calibration @cite Horaud95
		//	CALIB_HAND_EYE_ANDREFF = 3, //!< On-line Hand-Eye Calibration @cite Andreff99
		//	CALIB_HAND_EYE_DANIILIDIS = 4  //!< Hand-Eye Calibration Using Dual Quaternions @cite Daniilidis98

		calibrateHandEyeTsai(wMeList, cMoList, R_cam2gripper, t_cam2gripper);
		cout << "TSAI" << endl;
		cout << "R_cam2gripper" << R_cam2gripper << endl;
		cout << "t_cam2gripper" << t_cam2gripper << endl;
		cout <<  endl;
		rmatWorldCam.push_back(R_cam2gripper);
		tmatWorldCam.push_back(t_cam2gripper);

		calibrateHandEyePark(wMeList, cMoList, R_cam2gripper, t_cam2gripper);
		cout << "Park" << endl;
		cout << "R_cam2gripper" << R_cam2gripper << endl;
		cout << "t_cam2gripper" << t_cam2gripper << endl;
		cout << endl;
		rmatWorldCam.push_back(R_cam2gripper);
		tmatWorldCam.push_back(t_cam2gripper);

		calibrateHandEyeAndreff(wMeList, cMoList, R_cam2gripper, t_cam2gripper);
		cout << "Andreff" << endl;
		cout << "R_cam2gripper" << R_cam2gripper << endl;
		cout << "t_cam2gripper" << t_cam2gripper << endl;
		cout  << endl;
		rmatWorldCam.push_back(R_cam2gripper);
		tmatWorldCam.push_back(t_cam2gripper);


		calibrateHandEyeDaniilidis(wMeList, cMoList, R_cam2gripper, t_cam2gripper);
		cout << "Daniilidis" << endl;
		cout << "R_cam2gripper" << R_cam2gripper << endl;
		cout << "t_cam2gripper" << t_cam2gripper << endl;
		cout <<  endl;
		rmatWorldCam.push_back(R_cam2gripper);
		tmatWorldCam.push_back(t_cam2gripper);

		//double  aa[] = { 5.3966594447318217e-01, 2.2085724062102480e-01,- 2.6895722773773745e-01 };     //eye in hand 数据
		//Mat R(3, 3, CV_64FC1);
		//cv::Mat aa_r(Size(3, 1), CV_64F, aa);
		//Rodrigues(aa_r, R);
		//cout << R << endl << endl << endl;
	}
	catch (cv::Exception& e)
	{
		const char* err_msg = e.what();
		std::cout << "exception caught: " << err_msg << std::endl;
		exitCode = dTRUE;
	}
	return exitCode;
}



int main()
{
	std::vector<std::vector<Point2f>> image_list;
	std::vector<Mat>rmatCamObjectList;
	std::vector<Mat>tvecsCamObjectList;
	std::vector<Mat> tvecsWorldEndEffectorList, rmatWorldEndEffectorList;

	//相机内参
	double  instri_param_matrix[3][3] = { { 2.4606028655140117e+03, 0., 1.2433252307808543e+03 },
	{ 0., 2.4559333644176145e+03, 1.0420874394843283e+03 },
	{ 0.0, 0.0, 1.0 } };
	//畸变数据
	double  dis_param_matrix[] = { -9.2639914701504775e-02, 1.0334743618841541e-01,
		1.8694078615969704e-04, 1.2592752600556767e-03,
		1.2571171650945409e-01 };  //eye to hand 数据

	cv::Mat  instri_param(Size(3, 3), CV_64F, instri_param_matrix);
	cv::Mat dis_param(Size(5, 1), CV_64F, dis_param_matrix);


	string testStr = "HandEyeCalibrate\\3c\\1112\\";
	//set<int> no_use{ 3,29 };
	set<int> no_use{  };
	for (int i = 0; i <30 ; i++)
	{
		set<int>::iterator iter;
		if ((iter = no_use.find(i)) != no_use.end())
			{
				continue;
			}

		std::vector<Point2f> imgPointList;

		Mat frame;

		cout << testStr + "left"+cv::format("%d", i) + ".bmp" << endl;
		frame = imread(testStr + "left" + cv::format("%d", i) + ".bmp", 1);


		CRAlgoTestChessBoard(frame, &imgPointList);

		image_list.push_back(imgPointList);
	}
	//计算棋盘位姿
	CRAlgoCalcCameraMat(dGRID_MM, image_list, instri_param, dis_param, &rmatCamObjectList, &tvecsCamObjectList);


	//读取机器人末端数据
	CRAlgoCalcWorldEndEffectorMat(&tvecsWorldEndEffectorList, &rmatWorldEndEffectorList);

	std::vector<Mat> 	rmatWorldCam, tmatWorldCam;
	CRAlgoCalcEndEffectorCamMat(tvecsWorldEndEffectorList, rmatWorldEndEffectorList, tvecsCamObjectList, rmatCamObjectList, rmatWorldCam, tmatWorldCam);



	//CALIB_HAND_EYE_TSAI = 0, //!< A New Technique for Fully Autonomous and Efficient 3D Robotics Hand/Eye Calibration @cite Tsai89
	//	CALIB_HAND_EYE_PARK = 1, //!< Robot Sensor Calibration: Solving AX = XB on the Euclidean Group @cite Park94
	//	CALIB_HAND_EYE_HORAUD = 2, //!< Hand-eye Calibration @cite Horaud95
	//	CALIB_HAND_EYE_ANDREFF = 3, //!< On-line Hand-Eye Calibration @cite Andreff99
	//	CALIB_HAND_EYE_DANIILIDIS

	Mat R_cam2gripper = Mat(3, 3, CV_64FC1);				//相机与机械臂末端坐标系的旋转矩阵与平移矩阵
	Mat T_cam2gripper = Mat(3, 1, CV_64FC1);




	//矩阵求逆，验证opencv用于eye to hand 手眼标使用方式
	std::vector<Mat> inv_rmatWorldEndEffectorList;
	std::vector<Mat>inv_tvecsCamObjectList;
	for (int i = 0; i < (int)rmatWorldEndEffectorList.size(); i++)
	{
		Mat inv_rmat(3, 3, CV_64FC1), inv_tmat(3, 1, CV_64FC1);
		inv_rmat = rmatWorldEndEffectorList[i].inv();
		inv_tmat = -inv_rmat*tvecsWorldEndEffectorList[i];
		inv_rmatWorldEndEffectorList.push_back(inv_rmat);

		inv_tvecsCamObjectList.push_back(inv_tmat);

		
	}


	//cv::calibrateHandEye(rmatWorldEndEffectorList, tvecsWorldEndEffectorList, rmatCamObjectList, tvecsCamObjectList, R_cam2gripper, T_cam2gripper, 
	//	cv::CALIB_HAND_EYE_TSAI);
	cv::calibrateHandEye(inv_rmatWorldEndEffectorList, inv_tvecsCamObjectList, rmatCamObjectList, tvecsCamObjectList,  R_cam2gripper, T_cam2gripper,
		cv::CALIB_HAND_EYE_TSAI);
	cout << "TSAI--opencv:\n" << "R_cam2gripper\n" << R_cam2gripper << "\nT_cam2gripper\n" << T_cam2gripper << endl;


}