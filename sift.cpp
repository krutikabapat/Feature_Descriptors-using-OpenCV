#include<stdio.h>
#include<iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace cv:: xfeatures2d;

void reade();

int main( int argc, char** argv)
{
	Mat img_1 = imread(argv[1], IMREAD_GRAYSCALE);
	Mat img_2 = imread(argv[2], IMREAD_GRAYSCALE);

	int minHessian = 400;

	Ptr<SURF> detector = SURF::create(minHessian);
	std::vector<KeyPoint> keypoints_1, keypoints_2;

  detector->detect( img_1, keypoints_1 );
  detector->detect( img_2, keypoints_2 );

  //-- Draw keypoints
  Mat img_keypoints_1; Mat img_keypoints_2;

  drawKeypoints( img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
  drawKeypoints( img_2, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

  //-- Show detected (drawn) keypoints
  imshow("Keypoints 1", img_keypoints_1 );
  imshow("Keypoints 2", img_keypoints_2 );

  waitKey(0);

  return 0;
  }

  void readme()
  { std::cout << " Usage: ./SURF_detector <img1> <img2>" << std::endl; }
