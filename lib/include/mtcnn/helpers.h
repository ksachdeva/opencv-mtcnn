#ifndef _include_opencv_helpers_h_
#define _include_opencv_helpers_h_

#include <opencv2/core.hpp>

inline cv::Mat cropImage(const cv::Mat &img, cv::Rect r) {
  cv::Mat m = cv::Mat::zeros(r.height, r.width, img.type());
  int dx = std::abs(std::min(0, r.x));
  if (dx > 0) {
    r.x = 0;
  }
  r.width -= dx;
  int dy = std::abs(std::min(0, r.y));
  if (dy > 0) {
    r.y = 0;
  }
  r.height -= dy;
  int dw = std::abs(std::min(0, img.cols - 1 - (r.x + r.width)));
  r.width -= dw;
  int dh = std::abs(std::min(0, img.rows - 1 - (r.y + r.height)));
  r.height -= dh;
  if (r.width > 0 && r.height > 0) {
    img(r).copyTo(m(cv::Range(dy, dy + r.height), cv::Range(dx, dx + r.width)));
  }
  return m;
}

#endif