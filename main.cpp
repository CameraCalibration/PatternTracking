#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

Mat src; Mat src_gray;
int thresh = 90;
int max_thresh = 255;
RNG rng(12345);

/// Function header
void thresh_callback(int, void*);

float euclideanDist(Point& p, Point& q) {
	Point diff = p - q;
	return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}

struct node {
	Point2f center;
	int count;
	float w;

	node(Point center, int count, float w)
	{
		this->center = center;
		this->count = count;
		this->w = w;
	}
};

/** @function main */
int main(int argc, char** argv)
{
	/// Load source image and convert it to gray
	src = imread("images/pattern_01.PNG", 1);

	/// Convert image to gray and blur it
	cvtColor(src, src_gray, CV_BGR2GRAY);
	blur(src_gray, src_gray, Size(3, 3));

	// 0: Binary
	// 1: Binary Inverted
	// 2: Threshold Truncated
	// 3: Threshold to Zero
	// 4: Threshold to Zero Inverted

	threshold(src_gray, src_gray, 100, 255, 3);

	/// Create Window
	namedWindow("Source", CV_WINDOW_AUTOSIZE);
	imshow("Source", src);

	createTrackbar(" Threshold:", "Source", &thresh, max_thresh, thresh_callback);
	thresh_callback(0, 0);

	waitKey(0);
	return(0);
}

/** @function thresh_callback */
void thresh_callback(int, void*)
{
	Mat threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	/// Detect edges using Threshold
	threshold(src_gray, threshold_output, 100, 255, THRESH_BINARY);

	/// Detect edges using canny
	Canny(threshold_output, threshold_output, thresh, thresh * 2, 3);

	/// Find contours
	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// Find the rotated rectangles and ellipses for each contour
	vector<RotatedRect> minRect(contours.size());
	vector<RotatedRect> minEllipse(contours.size());

	for (int i = 0; i < contours.size(); i++)
	{
		minRect[i] = minAreaRect(Mat(contours[i]));
		if (contours[i].size() > 5)
		{
			minEllipse[i] = fitEllipse(Mat(contours[i]));
		}
	}

	// Draw ellipses
	Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
	Point2f last(-10, -10);
	std::vector<node> centers;
	int n = 0;
	for (int i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		Point2f np(minEllipse[i].center.x, minEllipse[i].center.y);
		if (norm(last - np) < 5) {
			centers[n - 1].count++;
			float w_np = pow(minEllipse[i].size.area(), 3);
			centers[n - 1].center = (np*w_np + centers[n - 1].center*centers[n - 1].w) / (w_np + centers[n - 1].w);
			if (centers[n - 1].count > 2) {
				ellipse(drawing, minEllipse[i], color, 2, 8);
			}
		}
		else {
			n++;
			centers.push_back(node(np, 1, pow(minEllipse[i].size.area(), 3)));
		}
		last = np;
	}

	std::vector<Point2f> PointBuffer;

	for (auto center : centers) {
		if (center.count > 3)
			PointBuffer.push_back(center.center);
	}

	cv::drawChessboardCorners(src, Size(5, 4), PointBuffer, true);
	namedWindow("Pattern", CV_WINDOW_AUTOSIZE);
	imshow("Pattern", src);

	/// Show in a window
	namedWindow("Contours", CV_WINDOW_AUTOSIZE);
	imshow("Contours", drawing);
}