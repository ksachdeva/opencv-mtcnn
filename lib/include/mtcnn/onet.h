#ifndef _include_opencv_onet_h_
#define _include_opencv_onet_h_

#include "face.h"
#include <opencv2/dnn.hpp>

class OutputNetwork {
public:
  struct Config {
  public:
    std::string protoText;
    std::string caffeModel;
    float threshold;
  };

private:
  cv::dnn::Net _net;
  float _threshold;

public:
  OutputNetwork(const OutputNetwork::Config &config);
  OutputNetwork();

private:
  OutputNetwork(const OutputNetwork &rhs) = delete;
  OutputNetwork &operator=(const OutputNetwork &rhs) = delete;

public:
  std::vector<Face> run(const cv::Mat &img, const std::vector<Face> &faces);
};

#endif
