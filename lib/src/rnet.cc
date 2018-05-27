#include "mtcnn/rnet.h"
#include "mtcnn/helpers.h"

const int INPUT_DATA_WIDTH = 24;
const int INPUT_DATA_HEIGHT = 24;

const float IMG_MEAN = 127.5f;
const float IMG_INV_STDDEV = 1.f / 128.f;

RefineNetwork::RefineNetwork(const RefineNetwork::Config &config) {
  _net = cv::dnn::readNetFromCaffe(config.protoText, config.caffeModel);
  if (_net.empty()) {
    throw std::invalid_argument("invalid protoText or caffeModel");
  }
  _threshold = config.threshold;
}

RefineNetwork::~RefineNetwork() {}

std::vector<Face> RefineNetwork::run(const cv::Mat &img,
                                     const std::vector<Face> &faces) {
  cv::Size windowSize = cv::Size(INPUT_DATA_WIDTH, INPUT_DATA_HEIGHT);

  std::vector<cv::Mat> inputs;
  for (auto &f : faces) {
    cv::Mat roi = cropImage(img, f.bbox.getRect());
    cv::resize(roi, roi, windowSize, 0, 0, cv::INTER_AREA);
    inputs.push_back(roi);
  }

  // build blob images from the inputs
  auto blobInputs =
      cv::dnn::blobFromImages(inputs, IMG_INV_STDDEV, cv::Size(),
                              cv::Scalar(IMG_MEAN, IMG_MEAN, IMG_MEAN), false);

  _net.setInput(blobInputs, "data");

  const std::vector<cv::String> outBlobNames{"conv5-2", "prob1"};
  std::vector<cv::Mat> outputBlobs;

  _net.forward(outputBlobs, outBlobNames);

  cv::Mat regressionsBlob = outputBlobs[0];
  cv::Mat scoresBlob = outputBlobs[1];

  std::vector<Face> totalFaces;

  const float *scores_data = (float *)scoresBlob.data;
  const float *reg_data = (float *)regressionsBlob.data;

  for (int k = 0; k < faces.size(); ++k) {
    if (scores_data[2 * k + 1] >= _threshold) {
      Face info = faces[k];
      info.score = scores_data[2 * k + 1];
      for (int i = 0; i < 4; ++i) {
        info.regression[i] = reg_data[4 * k + i];
      }
      totalFaces.push_back(info);
    }
  }

  // nms and regression
  totalFaces = Face::runNMS(totalFaces, 0.7f);
  Face::applyRegression(totalFaces, true);
  Face::bboxes2Squares(totalFaces);

  return totalFaces;
}
