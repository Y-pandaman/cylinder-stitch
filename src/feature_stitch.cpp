#include "opencv2/highgui/highgui.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <iostream>

using namespace cv;
using namespace std;

void OptimizeSeam(Mat &img1, Mat &trans, Mat &dst);

typedef struct {
  Point2f left_top;
  Point2f left_bottom;
  Point2f right_top;
  Point2f right_bottom;
} four_corners_t;

four_corners_t corners;

void CalcCorners(const Mat &H, const Mat &src) {
  double v2[] = {0, 0, 1}; //å·¦ä¸Šè§’
  double v1[3];            //å˜æ¢åŽçš„åæ ‡å€¼
  Mat V2 = Mat(3, 1, CV_64FC1, v2); //åˆ—å‘é‡
  Mat V1 = Mat(3, 1, CV_64FC1, v1); //åˆ—å‘é‡

  V1 = H * V2;
  //å·¦ä¸Šè§’(0,0,1)
  cout << "V2: " << V2 << endl;
  cout << "V1: " << V1 << endl;
  corners.left_top.x = v1[0] / v1[2];
  corners.left_top.y = v1[1] / v1[2];

  //å·¦ä¸‹è§’(0,src.rows,1)
  v2[0] = 0;
  v2[1] = src.rows;
  v2[2] = 1;
  V2 = Mat(3, 1, CV_64FC1, v2); //åˆ—å‘é‡
  V1 = Mat(3, 1, CV_64FC1, v1); //åˆ—å‘é‡
  V1 = H * V2;
  corners.left_bottom.x = v1[0] / v1[2];
  corners.left_bottom.y = v1[1] / v1[2];

  //å³ä¸Šè§’(src.cols,0,1)
  v2[0] = src.cols;
  v2[1] = 0;
  v2[2] = 1;
  V2 = Mat(3, 1, CV_64FC1, v2); //åˆ—å‘é‡
  V1 = Mat(3, 1, CV_64FC1, v1); //åˆ—å‘é‡
  V1 = H * V2;
  corners.right_top.x = v1[0] / v1[2];
  corners.right_top.y = v1[1] / v1[2];

  //å³ä¸‹è§’(src.cols,src.rows,1)
  v2[0] = src.cols;
  v2[1] = src.rows;
  v2[2] = 1;
  V2 = Mat(3, 1, CV_64FC1, v2); //åˆ—å‘é‡
  V1 = Mat(3, 1, CV_64FC1, v1); //åˆ—å‘é‡
  V1 = H * V2;
  corners.right_bottom.x = v1[0] / v1[2];
  corners.right_bottom.y = v1[1] / v1[2];
}

int main(int argc, char *argv[]) {
  Mat image01 = imread("../images/shs/1.jpg", 1); //å³å›¾
  Mat image02 = imread("../image2/shs/2.jpg", 1); //å·¦å›¾
  imshow("p2", image01);
  imshow("p1", image02);

  //ç°åº¦å›¾è½¬æ¢
  Mat image1, image2;
  cvtColor(image01, image1, CV_RGB2GRAY);
  cvtColor(image02, image2, CV_RGB2GRAY);

  //æå–ç‰¹å¾ç‚¹
  SurfFeatureDetector Detector(2000);
  vector<KeyPoint> keyPoint1, keyPoint2;
  Detector.detect(image1, keyPoint1);
  Detector.detect(image2, keyPoint2);

  //ç‰¹å¾ç‚¹æè¿°ï¼Œä¸ºä¸‹è¾¹çš„ç‰¹å¾ç‚¹åŒ¹é…åšå‡†å¤‡
  SurfDescriptorExtractor Descriptor;
  Mat imageDesc1, imageDesc2;
  Descriptor.compute(image1, keyPoint1, imageDesc1);
  Descriptor.compute(image2, keyPoint2, imageDesc2);

  FlannBasedMatcher matcher;
  vector<vector<DMatch>> matchePoints;
  vector<DMatch> GoodMatchePoints;

  vector<Mat> train_desc(1, imageDesc1);
  matcher.add(train_desc);
  matcher.train();

  matcher.knnMatch(imageDesc2, matchePoints, 2);
  cout << "total match points: " << matchePoints.size() << endl;

  // Lowe's algorithm,èŽ·å–ä¼˜ç§€åŒ¹é…ç‚¹
  for (int i = 0; i < matchePoints.size(); i++) {
    if (matchePoints[i][0].distance < 0.4 * matchePoints[i][1].distance) {
      GoodMatchePoints.push_back(matchePoints[i][0]);
    }
  }

  Mat first_match;
  drawMatches(image02, keyPoint2, image01, keyPoint1, GoodMatchePoints,
              first_match);
  imshow("first_match ", first_match);

  vector<Point2f> imagePoints1, imagePoints2;

  for (int i = 0; i < GoodMatchePoints.size(); i++) {
    imagePoints2.push_back(keyPoint2[GoodMatchePoints[i].queryIdx].pt);
    imagePoints1.push_back(keyPoint1[GoodMatchePoints[i].trainIdx].pt);
  }

  //èŽ·å–å›¾åƒ1åˆ°å›¾åƒ2çš„æŠ•å½±æ˜ å°„çŸ©é˜µ
  //å°ºå¯¸ä¸º3*3
  Mat homo = findHomography(imagePoints1, imagePoints2, CV_RANSAC);
  ////ä¹Ÿå¯ä»¥ä½¿ç”¨getPerspectiveTransformæ–¹æ³•èŽ·å¾—é€è§†å˜æ¢çŸ©é˜µï¼Œä¸è¿‡è¦æ±‚åªèƒ½æœ‰4ä¸ªç‚¹ï¼Œæ•ˆæžœç¨å·®
  // Mat   homo=getPerspectiveTransform(imagePoints1,imagePoints2);
  cout << "homo\n"
       << homo << endl
       << endl; //è¾“å‡ºæ˜ å°„çŸ©é˜µ

  //è®¡ç®—é…å‡†å›¾çš„å››ä¸ªé¡¶ç‚¹åæ ‡
  CalcCorners(homo, image01);
  cout << "left_top:" << corners.left_top << endl;
  cout << "left_bottom:" << corners.left_bottom << endl;
  cout << "right_top:" << corners.right_top << endl;
  cout << "right_bottom:" << corners.right_bottom << endl;

  //å›¾åƒé…å‡†
  Mat imageTransform1, imageTransform2;
  warpPerspective(
      image01, imageTransform1, homo,
      Size(MAX(corners.right_top.x, corners.right_bottom.x), image02.rows));
  // warpPerspective(image01, imageTransform2, adjustMat*homo,
  // Size(image02.cols*1.3, image02.rows*1.8));
  imshow("imageTransform1",
         imageTransform1);
  imwrite("trans1.jpg", imageTransform1);

  //åˆ›å»ºæ‹¼æŽ¥åŽçš„å›¾,éœ€æå‰è®¡ç®—å›¾çš„å¤§å°
  int dst_width =
      imageTransform1
          .cols; //å–æœ€å³ç‚¹çš„é•¿åº¦ä¸ºæ‹¼æŽ¥å›¾çš„é•¿åº¦
  int dst_height = image02.rows;

  Mat dst(dst_height, dst_width, CV_8UC3);
  dst.setTo(0);

  imageTransform1.copyTo(
      dst(Rect(0, 0, imageTransform1.cols, imageTransform1.rows)));
  image02.copyTo(dst(Rect(0, 0, image02.cols, image02.rows)));

  imshow("b_dst", dst);

  OptimizeSeam(image02, imageTransform1, dst);

  imshow("dst", dst);
  imwrite("dst.jpg", dst);

  waitKey();

  return 0;
}

//ä¼˜åŒ–ä¸¤å›¾çš„è¿žæŽ¥å¤„ï¼Œä½¿å¾—æ‹¼æŽ¥è‡ªç„¶
void OptimizeSeam(Mat &img1, Mat &trans, Mat &dst) {
  int start = MIN(
      corners.left_top.x,
      corners.left_bottom
          .x); //å¼€å§‹ä½ç½®ï¼Œå³é‡å åŒºåŸŸçš„å·¦è¾¹ç•Œ

  double processWidth =
      img1.cols - start; //é‡å åŒºåŸŸçš„å®½åº¦
  int rows = dst.rows;
  int cols =
      img1.cols; //æ³¨æ„ï¼Œæ˜¯åˆ—æ•°*é€šé“æ•°
  double alpha = 1; // img1ä¸­åƒç´ çš„æƒé‡
  for (int i = 0; i < rows; i++) {
    uchar *p = img1.ptr<uchar>(
        i); //èŽ·å–ç¬¬iè¡Œçš„é¦–åœ°å€
    uchar *t = trans.ptr<uchar>(i);
    uchar *d = dst.ptr<uchar>(i);
    for (int j = start; j < cols; j++) {
      //å¦‚æžœé‡åˆ°å›¾åƒtransä¸­æ— åƒç´ çš„é»‘ç‚¹ï¼Œåˆ™å®Œå…¨æ‹·è´img1ä¸­çš„æ•°æ®
      if (t[j * 3] == 0 && t[j * 3 + 1] == 0 && t[j * 3 + 2] == 0) {
        alpha = 1;
      } else {
        // img1中像素的权重，与当前处理点距重叠区域左边界的距离成正比
        alpha = (processWidth - (j - start)) / processWidth;
      }

      d[j * 3] = p[j * 3] * alpha + t[j * 3] * (1 - alpha);
      d[j * 3 + 1] = p[j * 3 + 1] * alpha + t[j * 3 + 1] * (1 - alpha);
      d[j * 3 + 2] = p[j * 3 + 2] * alpha + t[j * 3 + 2] * (1 - alpha);
    }
  }
}
