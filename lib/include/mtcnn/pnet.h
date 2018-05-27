#ifndef _include_opencv_pnet_h_
#define _include_opencv_pnet_h_

#include "face.h"
#include <opencv2/dnn.hpp>

class ProposalNetwork {
public:
  struct Config {
  public:
    std::string protoText;
    std::string caffeModel;
    float threshold;
    float nmsThreshold;
  };

private:
  cv::dnn::Net _net;
  float _threshold;

private:
  std::vector<Face> buildFaces(const cv::Mat &scores,
                               const cv::Mat &regressions,
                               const float scaleFactor, const float threshold);

public:
  ProposalNetwork(const ProposalNetwork::Config &config);
  ~ProposalNetwork();

private:
  ProposalNetwork(const ProposalNetwork &rhs) = delete;
  ProposalNetwork &operator=(const ProposalNetwork &rhs) = delete;

public:
  std::vector<Face> run(const cv::Mat &img, const float minFaceSize,
                        const float scaleFactor);
};

#endif
