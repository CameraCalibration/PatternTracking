//#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include<omp.h>

using namespace cv;
using namespace std;

/// Function header
void thresh_callback(int, void*);

float dot(Point2f a, Point2f b) {
	return (a.x*b.x + a.y*b.y) / (cv::sqrt(a.x*a.x + a.y*a.y)*cv::sqrt(b.x*b.x + b.y*b.y));
}

float euclideanDist(Point2f& p, Point2f& q) {
	Point2f diff = p - q;
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

double getDirection(Point2f p1, Point2f p2, Point2f p3)
{
	Point2f a = p3 - p1;
	Point2f b = p2 - p1;
	return a.x * b.y - a.y * b.x;
}

double getSen(Point2f a, Point2f b)
{
	return (a.x * b.y - a.y * b.x)/(cv::sqrt(a.x*a.x + a.y*a.y)*cv::sqrt(b.x*b.x + b.y*b.y));
}

Mat rectifyGrid(Size detectedGridSize, const std::vector<Point2f>& centers,
	const std::vector<Point2f> &keypoints, std::vector<Point2f> &warpedKeypoints)
{
	CV_Assert(!centers.empty());
	const float edgeLength = 30;
	const Point2f offset(150, 150);

	std::vector<Point2f> dstPoints;
	bool isClockwiseBefore = getDirection(centers[0], centers[detectedGridSize.width - 1], centers[centers.size() - 1]) < 0;

	int iStart = isClockwiseBefore ? 0 : detectedGridSize.height - 1;
	int iEnd = isClockwiseBefore ? detectedGridSize.height : -1;
	int iStep = isClockwiseBefore ? 1 : -1;
	for (int i = iStart; i != iEnd; i += iStep)
	{
		for (int j = 0; j < detectedGridSize.width; j++)
		{
			dstPoints.push_back(offset + Point2f(edgeLength * j, edgeLength * i));
		}
	}

	Mat H = findHomography(Mat(centers), Mat(dstPoints), RANSAC);
	//Mat H = findHomography( Mat( corners ), Mat( dstPoints ) );

	if (H.empty())
	{
		H = Mat::zeros(3, 3, CV_64FC1);
		warpedKeypoints.clear();
		return H;
	}

	std::vector<Point2f> srcKeypoints;
	for (size_t i = 0; i < keypoints.size(); i++)
	{
		srcKeypoints.push_back(keypoints[i]);
	}

	Mat dstKeypointsMat;
	transform(Mat(srcKeypoints), dstKeypointsMat, H);
	std::vector<Point2f> dstKeypoints;
	convertPointsFromHomogeneous(dstKeypointsMat, dstKeypoints);

	warpedKeypoints.clear();
	for (size_t i = 0; i < dstKeypoints.size(); i++)
	{
		Point2f pt = dstKeypoints[i];
		warpedKeypoints.push_back(pt);
	}

	return H;
}

// Draw a single point
static void draw_point(Mat& img, Point2f fp, Scalar color)
{
	circle(img, fp, 2, color, CV_FILLED, CV_AA, 0);
}

// Draw delaunay triangles
static void draw_delaunay(Mat& img, Subdiv2D& subdiv, Scalar delaunay_color)
{

	vector<Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);
	vector<Point> pt(3);
	Size size = img.size();
	Rect rect(0, 0, size.width, size.height);

	for (size_t i = 0; i < triangleList.size(); i++)
	{
		Vec6f t = triangleList[i];
		pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
		pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
		pt[2] = Point(cvRound(t[4]), cvRound(t[5]));

		// Draw rectangles completely inside the image.
		if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
		{
			line(img, pt[0], pt[1], delaunay_color, 1, CV_AA, 0);
			line(img, pt[1], pt[2], delaunay_color, 1, CV_AA, 0);
			line(img, pt[2], pt[0], delaunay_color, 1, CV_AA, 0);
		}
	}
}

static void draw_subdiv_point(Mat& img, Point2f fp, Scalar color)
{
	circle(img, fp, 3, color, FILLED, LINE_8, 0);
}

static void draw_subdiv(Mat& img, Subdiv2D& subdiv, Scalar delaunay_color)
{
#if 1
	vector<Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);
	vector<Point> pt(3);

	for (size_t i = 0; i < triangleList.size(); i++)
	{
		Vec6f t = triangleList[i];
		pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
		pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
		pt[2] = Point(cvRound(t[4]), cvRound(t[5]));
		line(img, pt[0], pt[1], delaunay_color, 1, LINE_AA, 0);
		line(img, pt[1], pt[2], delaunay_color, 1, LINE_AA, 0);
		line(img, pt[2], pt[0], delaunay_color, 1, LINE_AA, 0);
	}
#else
	vector<Vec4f> edgeList;
	subdiv.getEdgeList(edgeList);
	for (size_t i = 0; i < edgeList.size(); i++)
	{
		Vec4f e = edgeList[i];
		Point pt0 = Point(cvRound(e[0]), cvRound(e[1]));
		Point pt1 = Point(cvRound(e[2]), cvRound(e[3]));
		line(img, pt0, pt1, delaunay_color, 1, LINE_AA, 0);
	}
#endif
}

static void locate_point(Mat& img, Subdiv2D& subdiv, Point2f fp, Scalar active_color)
{
	int e0 = 0, vertex = 0;

	int fp_idx = subdiv.locate(fp, e0, vertex);

	Point2f nv = subdiv.getVertex(9);
	subdiv.locate(nv, e0, vertex);
	

	if (e0 > 0)
	{
		int e = e0;
		do
		{
			Point2f org, dst;
			if (subdiv.edgeOrg(e, &org) > 0 && subdiv.edgeDst(e, &dst) > 0) {
				line(img, org, dst, active_color, 3, LINE_AA, 0);
				std::cout << org.x << ", " << org.y << endl;
			}

			e = subdiv.getEdge(e, Subdiv2D::NEXT_AROUND_DST);
		} while (e != e0);
	}

	draw_subdiv_point(img, fp, active_color);
}

/** @function main */
int main(int argc, char** argv)
{
	VideoCapture cap("videos/PadronAnillos_03.avi");

	// Check if camera opened successfully
	if (!cap.isOpened()) {
		cout << "Error opening video stream or file" << endl;
		return -1;
	}

	char time[55], id[3];
	double start_time, gray_time, blur_time, threshold_time, canny_time, ellipse_time, grid_time;
	int all = 0, full = 0;

	Mat src, src_gray, src_threshold, src_blur, src_canny, src_ellipses;
	int thresh = 90;
	int max_thresh = 255;
	RNG rng(12345);

	while (true) {
		/// Load source image and convert it to gray
		//src = imread("images/Captura14.PNG", 1);
		cap >> src;

		// If the frame is empty, break immediately
		if (src.empty())
			break;

		/// Convert image to gray
		start_time = omp_get_wtime();
		cvtColor(src, src_gray, CV_BGR2GRAY);
		gray_time = omp_get_wtime() - start_time;

		/// Convert image to blur
		start_time = omp_get_wtime();
		blur(src_gray, src_blur, Size(3, 3));
		blur_time = omp_get_wtime() - start_time;

		// 0: Binary
		// 1: Binary Inverted
		// 2: Threshold Truncated
		// 3: Threshold to Zero
		// 4: Threshold to Zero Inverted

		/// Detect edges using Threshold
		start_time = omp_get_wtime();
		threshold(src_blur, src_threshold, 100, 255, 3);
		threshold_time = omp_get_wtime() - start_time;

		/// Detect edges using canny
		start_time = omp_get_wtime();
		Canny(src_threshold, src_canny, thresh, thresh * 2, 3);
		canny_time = omp_get_wtime() - start_time;

		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		/// Find contours
		start_time = omp_get_wtime();
		findContours(src_canny, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

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
		src_ellipses = Mat::zeros(src_canny.size(), CV_8UC3);
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
					ellipse(src_ellipses, minEllipse[i], color, 2, 8);
				}
			}
			else {
				n++;
				centers.push_back(node(np, 1, pow(minEllipse[i].size.area(), 3)));
			}
			last = np;
		}

		ellipse_time = omp_get_wtime() - start_time;


		start_time = omp_get_wtime();
		std::vector<cv::Point2f> PointBuffer;
		for (auto center : centers) {
			if (center.count > 3)
				PointBuffer.push_back(center.center);
		}

		Scalar color = Scalar(0, 0, 255);
		for (auto p : PointBuffer) {
			draw_point(src, p, color);
		}
		grid_time = omp_get_wtime() - start_time;

		if (PointBuffer.size() == 30)
			full++;
		all++;
		

		sprintf_s(time, "Gray Time: %.3f", gray_time);
		putText(src_gray, time, Point2f(15, 25), FONT_HERSHEY_PLAIN, 1.5, Scalar(0, 0, 255, 255), 2);
		imshow("Gray Format", src_gray);

		sprintf_s(time, "Blur Time: %.3f", blur_time);
		putText(src_blur, time, Point2f(15, 25), FONT_HERSHEY_PLAIN, 1.5, Scalar(0, 0, 255, 255), 2);
		imshow("Blur Filter", src_blur);

		sprintf_s(time, "Threshold Time: %.3f", threshold_time);
		putText(src_threshold, time, Point2f(15, 25), FONT_HERSHEY_PLAIN, 1.5, Scalar(0, 0, 255, 255), 2);
		imshow("Threshold", src_threshold);

		sprintf_s(time, "Canny Time: %.3f", canny_time);
		putText(src_canny, time, Point2f(15, 25), FONT_HERSHEY_PLAIN, 1.5, Scalar(0, 0, 255, 255), 2);
		imshow("Canny Filter", src_canny);

		sprintf_s(time, "Ellipses Time: %.3f", ellipse_time);
		putText(src_ellipses, time, Point2f(15, 25), FONT_HERSHEY_PLAIN, 1.5, Scalar(0, 0, 255, 255), 2);
		imshow("Ellipses", src_ellipses);

		sprintf_s(time, "Grid Time: %.3f - Total Time: %.3f - Acc: %.2f", grid_time, gray_time+blur_time+threshold_time+canny_time+ellipse_time+grid_time, 100.0*(float)full / float(all));
		putText(src, time, Point2f(15, 25), FONT_HERSHEY_PLAIN, 1.5, Scalar(0, 0, 255, 255), 2);
		imshow("Points", src);

		if (all == 200) {
			all = 0;
			full = 0;
		}

		// Press  ESC on keyboard to exit
		char c = (char)waitKey(25);
		if (c == 27)
			break;
	}
	// When everything done, release the video capture object
	cap.release();

	// Closes all the frames
	destroyAllWindows();
	waitKey(0);
	return(0);
}

/** @function thresh_callback */
/*void thresh_callback(int, void*)
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

	std::vector<cv::Point2f> PointBuffer;

	for (auto center : centers) {
		if (center.count > 3)
			PointBuffer.push_back(center.center);
	}

	// Rectangle to be used with Subdiv2D
	Size size = src_gray.size();
	Point2f oo(size);
	Point2f ox(Size(0,0));
	Point2f oy(size);
	Point2f ow(Size(0, 0));

	Rect rect(0, 0, size.width, size.height);
	// Create an instance of Subdiv2D
	Subdiv2D subdiv(rect);
	// Define window names
	string win_delaunay = "Delaunay Triangulation";
	// Define colors for drawing.
	Scalar delaunay_color(255, 255, 255), points_color(0, 0, 255);
	// Turn on animation while drawing triangles
	bool animate = false;
	// Insert points into subdiv
	for (vector<Point2f>::iterator it = PointBuffer.begin(); it != PointBuffer.end(); it++)
	{
		oo = ((*it).x < oo.x) ? (*it) : oo;
		ox = ((*it).y > ox.y) ? (*it) : ox;
		oy = ((*it).y < oy.y) ? (*it) : oy;
		ow = ((*it).x > ow.x) ? (*it) : ow;

		subdiv.insert(*it);
		// Show animation
		if (animate)
		{
			Mat img_copy = src.clone();
			// Draw delaunay triangles
			draw_subdiv(img_copy, subdiv, delaunay_color);
			imshow(win_delaunay, img_copy);
			waitKey(100);
		}

	}

	//cv::Point2f* nearests;
	//subdiv.findNearest(origin, nearests);
	// Draw delaunay triangles
	draw_subdiv(src, subdiv, delaunay_color);
	

	// Draw points
	/*for (int i = 0; i < 3; i++) {
		draw_point(src, nearests[i], points_color);
	}*/

	/*draw_point(src, oo, points_color);
	draw_point(src, ox, points_color);
	//draw_point(src, oy, points_color);
	//draw_point(src, ow, points_color);
	

	//vector<Point2f> pointsForSearch; //Insert all 2D points to this vector
	flann::KDTreeIndexParams indexParams;
	flann::Index kdtree(Mat(PointBuffer).reshape(1), indexParams);

	vector<Point2f> grid;
	float epsilon = 0.1;
	Point2f vox = ox - oo;
	last = oo;
	grid.push_back(last);
	do{
		vector<float> query;
		query.push_back(last.x); //Insert the 2D point we need to find neighbours to the query
		query.push_back(last.y); //Insert the 2D point we need to find neighbours to the query
		vector<int> indices;
		vector<float> dists;
		kdtree.knnSearch(query, indices, dists, 5);
		indices.erase(indices.begin());
		for (int idx: indices) {
			Point2f pbidx = PointBuffer[idx];
			float sen = getSen(PointBuffer[idx] - oo, Point2f(-vox.y, vox.x));
			if ((PointBuffer[idx].x -last.x)> 0.0001 && sen > 0.98 ) {
				draw_point(src, PointBuffer[idx], points_color);
				last = PointBuffer[idx];
				grid.push_back(last);
				break;
			}
		}
	} while (euclideanDist(last, ox) > 0.01);

	if (grid.size() == 5) {
		ox = oo;
		oo = oy;
		oy = ow;


		vox = ox - oo;
		last = oo;
		grid.push_back(last);
		do {
			vector<float> query;
			query.push_back(last.x); //Insert the 2D point we need to find neighbours to the query
			query.push_back(last.y); //Insert the 2D point we need to find neighbours to the query
			vector<int> indices;
			vector<float> dists;
			kdtree.knnSearch(query, indices, dists, 5);
			indices.erase(indices.begin());
			for (int idx : indices) {
				Point2f pbidx = PointBuffer[idx];
				float sen = getSen(PointBuffer[idx] - oo, Point2f(-vox.y, vox.x));
				if ((PointBuffer[idx].x - last.x) > 0.0001 && sen > 0.98) {
					draw_point(src, PointBuffer[idx], points_color);
					last = PointBuffer[idx];
					grid.push_back(last);
					break;
				}
			}
		} while (euclideanDist(last, ox) > 0.01);
	}

	imshow(win_delaunay, src);

	/*cv::drawChessboardCorners(src, Size(6, 5), PointBuffer, true);
	namedWindow("Pattern", CV_WINDOW_AUTOSIZE);
	imshow("Pattern", src);*/

	/// Show in a window
	//namedWindow("Contours", CV_WINDOW_AUTOSIZE);
	//imshow("Contours", drawing);
//}