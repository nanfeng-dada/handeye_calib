/*!
* \file handeye.h
* Hand-Eye Calibration by Different Methods.
*
* \author NUDT_WG
* \date March 2016
*
*
*/

#pragma once

#include <opencv2\opencv.hpp>
#include <vector>
#include "quaternion.h"

using namespace std;
using namespace cv;

#define dTRUE   1
#define dFALSE  0


Mat skew(Mat A)
{
	CV_Assert(A.cols == 1 && A.rows == 3);
	Mat B(3, 3, CV_64FC1);

	B.at<double>(0, 0) = 0.0;
	B.at<double>(0, 1) = -A.at<double>(2, 0);
	B.at<double>(0, 2) = A.at<double>(1, 0);

	B.at<double>(1, 0) = A.at<double>(2, 0);
	B.at<double>(1, 1) = 0.0;
	B.at<double>(1, 2) = -A.at<double>(0, 0);

	B.at<double>(2, 0) = -A.at<double>(1, 0);
	B.at<double>(2, 1) = A.at<double>(0, 0);
	B.at<double>(2, 2) = 0.0;

	return B;
}


/**
* Creates a dual quaternion from a rotation matrix and a translation vector.
*
* @Returns  void
* @param q [out] q
* @param qprime [out] q'
* @param R [in] Rotation
* @param t [in] Translation
*/
void getDualQ(Mat q, Mat qprime, Mat R, Mat t)
{
	Mat r(3, 1, CV_64FC1);
	Mat l(3, 1, CV_64FC1);
	double theta;
	Mat tempd(1, 1, CV_64FC1);
	double d;
	Mat c(3, 1, CV_64FC1);
	Mat m(3, 1, CV_64FC1);
	Mat templ(3, 1, CV_64FC1);
	Mat tempqt(1, 1, CV_64FC1);
	double qt;
	Mat tempml(3, 1, CV_64FC1);

	Rodrigues(R, r);
	theta = norm(r);
	l = r / theta;
	tempd = l.t()*t;
	d = tempd.at<double>(0, 0);

	c = 0.5*(t - d*l) + cos(theta / 2) / sin(theta / 2)*l.cross(t);
	m = c.cross(l);

	q.at<double>(0, 0) = cos(theta / 2);
	templ = sin(theta / 2)*l;
	templ.copyTo(q(Rect(0, 1, 1, 3)));

	tempqt = -0.5*templ.t()*t;
	qt = tempqt.at<double>(0, 0);
	tempml = 0.5*(q.at<double>(0, 0)*t + t.cross(templ));

	qprime.at<double>(0, 0) = qt;
	tempml.copyTo(qprime(Rect(0, 1, 1, 3)));

}

/**
* Compute the Kronecker tensor product of matrix A and B.
*
* @Returns  cv::Mat (MP)x(NQ) matrix
* @param A [in] MxN matrix
* @param B [in] PxQ matrix
*/
Mat kron(Mat A, Mat B)
{
	Mat C(A.rows*B.rows, A.cols*B.cols, CV_64FC1, Scalar(0));

	for (int i = 0; i < A.rows; i++)
		for (int j = 0; j < A.cols; j++)
			C(Rect(B.cols * j, B.rows * i, B.cols, B.rows)) = A.at<double>(i, j)*B;

	return C;
}

/**
* Signum function.
* For each element of X, SIGN(X) returns 1 if the element is greater than zero,
* return 0 if it equals zero and -1 if it is less than zero.
*
* @Returns  double
* @param a [in]
*/
double sign(double a)
{
	if (a > 0)
		return 1;
	else if (a < 0)
		return -1;
	else
		return 0;
}


static int sign_double(double val)
{
	return (0 < val) - (val < 0);
}

int CRAlgoCalcHomogeneousMat(Mat     translateVec,
	Mat     rotationMat,
	Mat&    homogeneousMat)
{
	int exitCode = dFALSE,
		i = 0,
		j = 0;

	try
	{
		for (j = 0; j < 3; j++)
		{
			for (i = 0; i < 3; i++)
			{
				homogeneousMat.at<double>(i, j) = (double)rotationMat.at<double>(i, j);
			}

			homogeneousMat.at<double>(j, 3) = (double)translateVec.at<double>(j, 0);

			homogeneousMat.at<double>(3, j) = 0.0;
		}

		homogeneousMat.at<double>(3, 3) = 1;
	}
	catch (cv::Exception& e)
	{
		const char* err_msg = e.what();
		std::cout << "exception caught: " << err_msg << std::endl;
		exitCode = dTRUE;
	}

	return exitCode;
}


static Mat homogeneousInverse(const Mat& T)
{
	CV_Assert(T.rows == 4 && T.cols == 4);

	Mat R = T(Rect(0, 0, 3, 3));
	Mat t = T(Rect(3, 0, 1, 3));
	Mat Rt = R.t();
	Mat tinv = -Rt * t;
	Mat Tinv = Mat::eye(4, 4, T.type());
	Rt.copyTo(Tinv(Rect(0, 0, 3, 3)));
	tinv.copyTo(Tinv(Rect(3, 0, 1, 3)));

	return Tinv;
}


static Mat rot2quat(const Mat& R)
{
	CV_Assert(R.type() == CV_64FC1 && R.rows >= 3 && R.cols >= 3);

	double m00 = R.at<double>(0, 0), m01 = R.at<double>(0, 1), m02 = R.at<double>(0, 2);
	double m10 = R.at<double>(1, 0), m11 = R.at<double>(1, 1), m12 = R.at<double>(1, 2);
	double m20 = R.at<double>(2, 0), m21 = R.at<double>(2, 1), m22 = R.at<double>(2, 2);
	double trace = m00 + m11 + m22;

	double qw, qx, qy, qz;
	if (trace > 0) {
		double S = sqrt(trace + 1.0) * 2; // S=4*qw
		qw = 0.25 * S;
		qx = (m21 - m12) / S;
		qy = (m02 - m20) / S;
		qz = (m10 - m01) / S;
	}
	else if ((m00 > m11)&(m00 > m22)) {
		double S = sqrt(1.0 + m00 - m11 - m22) * 2; // S=4*qx
		qw = (m21 - m12) / S;
		qx = 0.25 * S;
		qy = (m01 + m10) / S;
		qz = (m02 + m20) / S;
	}
	else if (m11 > m22) {
		double S = sqrt(1.0 + m11 - m00 - m22) * 2; // S=4*qy
		qw = (m02 - m20) / S;
		qx = (m01 + m10) / S;
		qy = 0.25 * S;
		qz = (m12 + m21) / S;
	}
	else {
		double S = sqrt(1.0 + m22 - m00 - m11) * 2; // S=4*qz
		qw = (m10 - m01) / S;
		qx = (m02 + m20) / S;
		qy = (m12 + m21) / S;
		qz = 0.25 * S;
	}

	return (Mat_<double>(4, 1) << qw, qx, qy, qz);
}


static Mat rot2quatMinimal(const Mat& R)
{
	CV_Assert(R.type() == CV_64FC1 && R.rows >= 3 && R.cols >= 3);

	double m00 = R.at<double>(0, 0), m01 = R.at<double>(0, 1), m02 = R.at<double>(0, 2);
	double m10 = R.at<double>(1, 0), m11 = R.at<double>(1, 1), m12 = R.at<double>(1, 2);
	double m20 = R.at<double>(2, 0), m21 = R.at<double>(2, 1), m22 = R.at<double>(2, 2);
	double trace = m00 + m11 + m22;

	double qx, qy, qz;
	if (trace > 0) {
		double S = sqrt(trace + 1.0) * 2; // S=4*qw
		qx = (m21 - m12) / S;
		qy = (m02 - m20) / S;
		qz = (m10 - m01) / S;
	}
	else if ((m00 > m11)&(m00 > m22)) {
		double S = sqrt(1.0 + m00 - m11 - m22) * 2; // S=4*qx
		qx = 0.25 * S;
		qy = (m01 + m10) / S;
		qz = (m02 + m20) / S;
	}
	else if (m11 > m22) {
		double S = sqrt(1.0 + m11 - m00 - m22) * 2; // S=4*qy
		qx = (m01 + m10) / S;
		qy = 0.25 * S;
		qz = (m12 + m21) / S;
	}
	else {
		double S = sqrt(1.0 + m22 - m00 - m11) * 2; // S=4*qz
		qx = (m02 + m20) / S;
		qy = (m12 + m21) / S;
		qz = 0.25 * S;
	}

	return (Mat_<double>(3, 1) << qx, qy, qz);
}

static Mat quat2rot(const Mat& q)
{
	CV_Assert(q.type() == CV_64FC1 && q.rows == 4 && q.cols == 1);

	double qw = q.at<double>(0, 0);
	double qx = q.at<double>(1, 0);
	double qy = q.at<double>(2, 0);
	double qz = q.at<double>(3, 0);

	Mat R(3, 3, CV_64FC1);
	R.at<double>(0, 0) = 1 - 2 * qy*qy - 2 * qz*qz;
	R.at<double>(0, 1) = 2 * qx*qy - 2 * qz*qw;
	R.at<double>(0, 2) = 2 * qx*qz + 2 * qy*qw;

	R.at<double>(1, 0) = 2 * qx*qy + 2 * qz*qw;
	R.at<double>(1, 1) = 1 - 2 * qx*qx - 2 * qz*qz;
	R.at<double>(1, 2) = 2 * qy*qz - 2 * qx*qw;

	R.at<double>(2, 0) = 2 * qx*qz - 2 * qy*qw;
	R.at<double>(2, 1) = 2 * qy*qz + 2 * qx*qw;
	R.at<double>(2, 2) = 1 - 2 * qx*qx - 2 * qy*qy;

	return R;
}


static Mat quatMinimal2rot(const Mat& q)
{
	CV_Assert(q.type() == CV_64FC1 && q.rows == 3 && q.cols == 1);

	Mat p = q.t()*q;
	double w = sqrt(1 - p.at<double>(0, 0));

	Mat diag_p = Mat::eye(3, 3, CV_64FC1)*p.at<double>(0, 0);
	return 2 * q*q.t() + 2 * w*skew(q) + Mat::eye(3, 3, CV_64FC1) - 2 * diag_p;
}



static Mat homogeneous2dualQuaternion(const Mat& H)
{
	CV_Assert(H.type() == CV_64FC1 && H.rows == 4 && H.cols == 4);

	Mat dualq(8, 1, CV_64FC1);
	Mat R = H(Rect(0, 0, 3, 3));
	Mat t = H(Rect(3, 0, 1, 3));

	Mat q = rot2quat(R);
	Mat qt = Mat::zeros(4, 1, CV_64FC1);
	t.copyTo(qt(Rect(0, 1, 1, 3)));
	Mat qprime = 0.5 * qmult(qt, q);

	q.copyTo(dualq(Rect(0, 0, 1, 4)));
	qprime.copyTo(dualq(Rect(0, 4, 1, 4)));

	return dualq;
}


static Mat dualQuaternion2homogeneous(const Mat& dualq)
{
	CV_Assert(dualq.type() == CV_64FC1 && dualq.rows == 8 && dualq.cols == 1);

	Mat q = dualq(Rect(0, 0, 1, 4));
	Mat qprime = dualq(Rect(0, 4, 1, 4));

	Mat R = quat2rot(q);
	q.at<double>(1, 0) = -q.at<double>(1, 0);
	q.at<double>(2, 0) = -q.at<double>(2, 0);
	q.at<double>(3, 0) = -q.at<double>(3, 0);

	Mat qt = 2 * qmult(qprime, q);
	Mat t = qt(Rect(0, 1, 1, 3));

	Mat H = Mat::eye(4, 4, CV_64FC1);
	R.copyTo(H(Rect(0, 0, 3, 3)));
	t.copyTo(H(Rect(3, 0, 1, 3)));

	return H;
}



static void calibrateHandEyeTsai(const std::vector<Mat>& Hg, const std::vector<Mat>& Hc,
	Mat& R_cam2gripper, Mat& t_cam2gripper)
{
	//Number of unique camera position pairs
	int K = static_cast<int>((Hg.size()*Hg.size() - Hg.size()) / 2.0);
	//Will store: skew(Pgij+Pcij)
	Mat A(3 * K, 3, CV_64FC1);
	//Will store: Pcij - Pgij
	Mat B(3 * K, 1, CV_64FC1);

	std::vector<Mat> vec_Hgij, vec_Hcij;
	vec_Hgij.reserve(static_cast<size_t>(K));
	vec_Hcij.reserve(static_cast<size_t>(K));

	int idx = 0;
	for (size_t i = 0; i < Hg.size(); i++)
	{
		for (size_t j = i + 1; j < Hg.size(); j++, idx++)
		{
			Mat Hgij = (Hg[j]) *homogeneousInverse(Hg[i]); //eq 6
			vec_Hgij.push_back(Hgij);
			Mat Pgij = 2 * rot2quatMinimal(Hgij);

			Mat Hcij = (Hc[j]) *homogeneousInverse(Hc[i]); //eq 7
			vec_Hcij.push_back(Hcij);
			Mat Pcij = 2 * rot2quatMinimal(Hcij);

			skew(Pgij + Pcij).copyTo(A(Rect(0, idx * 3, 3, 3)));
			//Right-hand side: Pcij - Pgij
			Mat diff = Pcij - Pgij;
			diff.copyTo(B(Rect(0, idx * 3, 1, 3)));
		}
	}

	Mat Pcg_;
	//Rotation from camera to gripper is obtained from the set of equations:
	//    skew(Pgij+Pcij) * Pcg_ = Pcij - Pgij    (eq 12)
	solve(A, B, Pcg_, DECOMP_SVD);

	Mat Pcg_norm = Pcg_.t() * Pcg_;
	//Obtained non-unit quaternion is scaled back to unit value that
	//designates camera-gripper rotation
	Mat Pcg = 2 * Pcg_ / sqrt(1 + Pcg_norm.at<double>(0, 0)); //eq 14

	Mat Rcg = quatMinimal2rot(Pcg / 2.0);

	idx = 0;
	for (size_t i = 0; i < Hg.size(); i++)
	{
		for (size_t j = i + 1; j < Hg.size(); j++, idx++)
		{
			//Defines coordinate transformation from Gi to Gj
			//Hgi is from Gi (gripper) to RW (robot base)
			//Hgj is from Gj (gripper) to RW (robot base)
			Mat Hgij = vec_Hgij[static_cast<size_t>(idx)];
			//Defines coordinate transformation from Ci to Cj
			//Hci is from CW (calibration target) to Ci (camera)
			//Hcj is from CW (calibration target) to Cj (camera)
			Mat Hcij = vec_Hcij[static_cast<size_t>(idx)];

			//Left-hand side: (Rgij - I)
			Mat diff = Hgij(Rect(0, 0, 3, 3)) - Mat::eye(3, 3, CV_64FC1);
			diff.copyTo(A(Rect(0, idx * 3, 3, 3)));

			//Right-hand side: Rcg*Tcij - Tgij
			diff = Rcg*Hcij(Rect(3, 0, 1, 3)) - Hgij(Rect(3, 0, 1, 3));
			diff.copyTo(B(Rect(0, idx * 3, 1, 3)));
		}
	}

	Mat Tcg;
	//Translation from camera to gripper is obtained from the set of equations:
	//    (Rgij - I) * Tcg = Rcg*Tcij - Tgij    (eq 15)
	solve(A, B, Tcg, DECOMP_SVD);

	R_cam2gripper = Rcg;
	t_cam2gripper = Tcg;

	Mat    homogeneousMatt(4, 4, CV_64FC1);
	CRAlgoCalcHomogeneousMat(t_cam2gripper, R_cam2gripper, homogeneousMatt);
	for (int i = 0; i < 18; i++)
		//cout << vec_Hgij[i] * homogeneousMatt - homogeneousMatt *vec_Hcij[i] << endl;
	cout << endl;
}

static void calibrateHandEyePark(const std::vector<Mat>& Hg, const std::vector<Mat>& Hc,
	Mat& R_cam2gripper, Mat& t_cam2gripper)
{
	Mat M = Mat::zeros(3, 3, CV_64FC1);

	for (size_t i = 0; i < Hg.size(); i++)
	{
		for (size_t j = i + 1; j < Hg.size(); j++)
		{
			Mat Hgij = (Hg[j]) *homogeneousInverse(Hg[i]);
			Mat Hcij = Hc[j] * homogeneousInverse(Hc[i]);

			Mat Rgij = Hgij(Rect(0, 0, 3, 3));
			Mat Rcij = Hcij(Rect(0, 0, 3, 3));

			Mat a, b;
			Rodrigues(Rgij, a);
			Rodrigues(Rcij, b);

			M += b * a.t();
		}
	}

	Mat eigenvalues, eigenvectors;
	eigen(M.t()*M, eigenvalues, eigenvectors);

	Mat v = Mat::zeros(3, 3, CV_64FC1);
	for (int i = 0; i < 3; i++) {
		v.at<double>(i, i) = 1.0 / sqrt(eigenvalues.at<double>(i, 0));
	}

	Mat R = eigenvectors.t() * v * eigenvectors * M.t();
	R_cam2gripper = R;

	int K = static_cast<int>((Hg.size()*Hg.size() - Hg.size()) / 2.0);
	Mat C(3 * K, 3, CV_64FC1);
	Mat d(3 * K, 1, CV_64FC1);
	Mat I3 = Mat::eye(3, 3, CV_64FC1);

	int idx = 0;
	for (size_t i = 0; i < Hg.size(); i++)
	{
		for (size_t j = i + 1; j < Hg.size(); j++, idx++)
		{
			Mat Hgij = (Hg[j]) * homogeneousInverse(Hg[i]);
			Mat Hcij = Hc[j] * homogeneousInverse(Hc[i]);

			Mat Rgij = Hgij(Rect(0, 0, 3, 3));

			Mat tgij = Hgij(Rect(3, 0, 1, 3));
			Mat tcij = Hcij(Rect(3, 0, 1, 3));

			Mat I_tgij = I3 - Rgij;
			I_tgij.copyTo(C(Rect(0, 3 * idx, 3, 3)));

			Mat A_RB = tgij - R*tcij;
			A_RB.copyTo(d(Rect(0, 3 * idx, 1, 3)));
		}
	}

	Mat t;
	solve(C, d, t, DECOMP_SVD);
	t_cam2gripper = t;

	Mat    homogeneousMatt(4, 4, CV_64FC1);
	CRAlgoCalcHomogeneousMat(t_cam2gripper, R_cam2gripper, homogeneousMatt);

	for (size_t i = 0; i <5; i++)
	{
		for (size_t j = i + 1; j < 5; j++, idx++)
		{
			Mat Hgij = (Hg[j]) * homogeneousInverse(Hg[i]);
			Mat Hcij = Hc[j] * homogeneousInverse(Hc[i]);
			//cout << Hgij * homogeneousMatt - homogeneousMatt *Hcij << endl;
		}
	}
	cout << endl;
}


static void calibrateHandEyeAndreff(const std::vector<Mat>& Hg, const std::vector<Mat>& Hc,
	Mat& R_cam2gripper, Mat& t_cam2gripper)
{
	int K = static_cast<int>((Hg.size()*Hg.size() - Hg.size()) / 2.0);
	Mat A(12 * K, 12, CV_64FC1);
	Mat B(12 * K, 1, CV_64FC1);

	Mat I9 = Mat::eye(9, 9, CV_64FC1);
	Mat I3 = Mat::eye(3, 3, CV_64FC1);
	Mat O9x3 = Mat::zeros(9, 3, CV_64FC1);
	Mat O9x1 = Mat::zeros(9, 1, CV_64FC1);

	int idx = 0;
	for (size_t i = 0; i < Hg.size(); i++)
	{
		for (size_t j = i + 1; j < Hg.size(); j++, idx++)
		{
			Mat Hgij = (Hg[j]) * homogeneousInverse(Hg[i]);
			Mat Hcij = Hc[j] * homogeneousInverse(Hc[i]);

			Mat Rgij = Hgij(Rect(0, 0, 3, 3));
			Mat Rcij = Hcij(Rect(0, 0, 3, 3));

			Mat tgij = Hgij(Rect(3, 0, 1, 3));
			Mat tcij = Hcij(Rect(3, 0, 1, 3));

			//Eq 10
			Mat a00 = I9 - kron(Rgij, Rcij);
			Mat a01 = O9x3;
			Mat a10 = kron(I3, tcij.t());
			Mat a11 = I3 - Rgij;

			a00.copyTo(A(Rect(0, idx * 12, 9, 9)));
			a01.copyTo(A(Rect(9, idx * 12, 3, 9)));
			a10.copyTo(A(Rect(0, idx * 12 + 9, 9, 3)));
			a11.copyTo(A(Rect(9, idx * 12 + 9, 3, 3)));

			O9x1.copyTo(B(Rect(0, idx * 12, 1, 9)));
			tgij.copyTo(B(Rect(0, idx * 12 + 9, 1, 3)));
		}
	}

	Mat X;
	solve(A, B, X, DECOMP_SVD);

	Mat R = X(Rect(0, 0, 1, 9));
	int newSize[] = { 3, 3 };
	R = R.reshape(1, 2, newSize);
	//Eq 15
	double det = determinant(R);
	R = pow(sign_double(det) / abs(det), 1.0 / 3.0) * R;

	Mat w, u, vt;
	SVDecomp(R, w, u, vt);
	R = u*vt;

	if (determinant(R) < 0)
	{
		Mat diag = (Mat_<double>(3, 3) << 1.0, 0.0, 0.0,
			0.0, 1.0, 0.0,
			0.0, 0.0, -1.0);
		R = u*diag*vt;
	}

	R_cam2gripper = R;

	Mat t = X(Rect(0, 9, 1, 3));
	t_cam2gripper = t;

	Mat    homogeneousMatt(4, 4, CV_64FC1);
	CRAlgoCalcHomogeneousMat(t_cam2gripper, R_cam2gripper, homogeneousMatt);

	for (size_t i = 0; i <5; i++)
	{
		for (size_t j = i + 1; j < 5; j++, idx++)
		{
			Mat Hgij = (Hg[j]) * homogeneousInverse(Hg[i]);
			Mat Hcij = Hc[j] * homogeneousInverse(Hc[i]);
			//cout << Hgij * homogeneousMatt - homogeneousMatt *Hcij << endl;
		}
	}
	cout << endl;
}



static void calibrateHandEyeDaniilidis(const std::vector<Mat>& Hg, const std::vector<Mat>& Hc,
	Mat& R_cam2gripper, Mat& t_cam2gripper)
{
	int K = static_cast<int>((Hg.size()*Hg.size() - Hg.size()) / 2.0);
	Mat T = Mat::zeros(6 * K, 8, CV_64FC1);

	int idx = 0;
	for (size_t i = 0; i < Hg.size(); i++)
	{
		for (size_t j = i + 1; j < Hg.size(); j++, idx++)
		{
			Mat Hgij = (Hg[j]) * homogeneousInverse(Hg[i]);
			Mat Hcij = Hc[j] * homogeneousInverse(Hc[i]);

			Mat dualqa = homogeneous2dualQuaternion(Hgij);
			Mat dualqb = homogeneous2dualQuaternion(Hcij);

			Mat a = dualqa(Rect(0, 1, 1, 3));
			Mat b = dualqb(Rect(0, 1, 1, 3));

			Mat aprime = dualqa(Rect(0, 5, 1, 3));
			Mat bprime = dualqb(Rect(0, 5, 1, 3));

			//Eq 31
			Mat s00 = a - b;
			Mat s01 = skew(a + b);
			Mat s10 = aprime - bprime;
			Mat s11 = skew(aprime + bprime);
			Mat s12 = a - b;
			Mat s13 = skew(a + b);

			s00.copyTo(T(Rect(0, idx * 6, 1, 3)));
			s01.copyTo(T(Rect(1, idx * 6, 3, 3)));
			s10.copyTo(T(Rect(0, idx * 6 + 3, 1, 3)));
			s11.copyTo(T(Rect(1, idx * 6 + 3, 3, 3)));
			s12.copyTo(T(Rect(4, idx * 6 + 3, 1, 3)));
			s13.copyTo(T(Rect(5, idx * 6 + 3, 3, 3)));
		}
	}

	Mat w, u, vt;
	SVDecomp(T, w, u, vt);
	Mat v = vt.t();

	Mat u1 = v(Rect(6, 0, 1, 4));
	Mat v1 = v(Rect(6, 4, 1, 4));
	Mat u2 = v(Rect(7, 0, 1, 4));
	Mat v2 = v(Rect(7, 4, 1, 4));

	//Solves Eq 34, Eq 35
	Mat ma = u1.t()*v1;
	Mat mb = u1.t()*v2 + u2.t()*v1;
	Mat mc = u2.t()*v2;

	double a = ma.at<double>(0, 0);
	double b = mb.at<double>(0, 0);
	double c = mc.at<double>(0, 0);

	double s1 = (-b + sqrt(b*b - 4 * a*c)) / (2 * a);
	double s2 = (-b - sqrt(b*b - 4 * a*c)) / (2 * a);

	Mat sol1 = s1*s1*u1.t()*u1 + 2 * s1*u1.t()*u2 + u2.t()*u2;
	Mat sol2 = s2*s2*u1.t()*u1 + 2 * s2*u1.t()*u2 + u2.t()*u2;
	double s, val;
	if (sol1.at<double>(0, 0) > sol2.at<double>(0, 0))
	{
		s = s1;
		val = sol1.at<double>(0, 0);
	}
	else
	{
		s = s2;
		val = sol2.at<double>(0, 0);
	}

	double lambda2 = sqrt(1.0 / val);
	double lambda1 = s * lambda2;

	Mat dualq = lambda1 * v(Rect(6, 0, 1, 8)) + lambda2*v(Rect(7, 0, 1, 8));
	Mat X = dualQuaternion2homogeneous(dualq);

	Mat R = X(Rect(0, 0, 3, 3));
	Mat t = X(Rect(3, 0, 1, 3));
	R_cam2gripper = R;
	t_cam2gripper = t;

	Mat    homogeneousMat(4, 4, CV_64FC1);
	CRAlgoCalcHomogeneousMat(t_cam2gripper, R_cam2gripper, homogeneousMat);


	//double  homogeneousMatttttt_matrix[4][4] = { { -0.666266,	0.705161, -0.242563, -0.234735},
	//{0.710969,	0.69881,	0.0786571, -0.553573},
	//{0.224972, -0.120048, -0.966942,	0.624139},
	//{0,	0,	0,	1} };
	//cv::Mat  homogeneousMatttttt(Size(4, 4), CV_64F, homogeneousMatttttt_matrix);

	for (size_t i = 0; i <5; i++)
	{
		for (size_t j = i + 1; j < 5; j++, idx++)
		{
			Mat Hgij = (Hg[j]) * homogeneousInverse(Hg[i]);
			Mat Hcij = Hc[j] * homogeneousInverse(Hc[i]);
			//cout << abs(Hgij * (homogeneousMat)) - abs((homogeneousMat) *Hcij) << endl;
		}
	}
	cout  << endl;
}