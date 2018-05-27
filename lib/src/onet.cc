#include "mtcnn/onet.h"
#include "mtcnn/helpers.h"

const int INPUT_DATA_WIDTH = 48;
const int INPUT_DATA_HEIGHT = 48;

const float IMG_MEAN = 127.5f;
const float IMG_INV_STDDEV = 1.f / 128.f;

OutputNetwork::OutputNetwork(const OutputNetwork::Config &config) {
  _net = cv::dnn::readNetFromCaffe(config.protoText, config.caffeModel);
  if (_net.empty()) {
    throw std::invalid_argument("invalid protoText or caffeModel");
  }
  _threshold = config.threshold;
}

OutputNetwork::OutputNetwork() {}

std::vector<Face> OutputNetwork::run(const cv::Mat &img,
                                     const std::vector<Face> &faces) {
  cv::Size windowSize = cv::Size(INPUT_DATA_WIDTH, INPUT_DATA_HEIGHT);

  std::vector<Face> totalFaces;

  for (auto &f : faces) {
    cv::Mat roi = cropImage(img, f.bbox.getRect());
    cv::resize(roi, roi, windowSize, 0, 0, cv::INTER_AREA);

    // we will run the ONet on each face
    // TODO : see how this can be optimized such that we run
    // it only 1 time

    // build blob images from the inputs
    auto blobInput =
        cv::dnn::blobFromImage(roi, IMG_INV_STDDEV, cv::Size(),
                               cv::Scalar(IMG_MEAN, IMG_MEAN, IMG_MEAN), false);

    _net.setInput(blobInput, "data");

    const std::vector<cv::String> outBlobNames{"conv6-2", "conv6-3", "prob1"};
    std::vector<cv::Mat> outputBlobs;

    _net.forward(outputBlobs, outBlobNames);

    cv::Mat regressionsBlob = outputBlobs[0];
    cv::Mat landMarkBlob = outputBlobs[1];
    cv::Mat scoresBlob = outputBlobs[2];

    const float *scores_data = (float *)scoresBlob.data;
    const float *landmark_data = (float *)landMarkBlob.data;
    const float *reg_data = (float *)regressionsBlob.data;

    if (scores_data[1] >= _threshold) {
      Face info = f;
      info.score = scores_data[1];
      for (int i = 0; i < 4; ++i) {
        info.regression[i] = reg_data[i];
      }

      float w = info.bbox.x2 - info.bbox.x1 + 1.f;
      float h = info.bbox.y2 - info.bbox.y1 + 1.f;

      for (int p = 0; p < NUM_PTS; ++p) {
        info.ptsCoords[2 * p] =
            info.bbox.x1 + landmark_data[NUM_PTS + p] * w - 1;
        info.ptsCoords[2 * p + 1] = info.bbox.y1 + landmark_data[p] * h - 1;
      }

      totalFaces.push_back(info);
    }
  }

  Face::applyRegression(totalFaces, true);
  totalFaces = Face::runNMS(totalFaces, 0.7f, true);

  return totalFaces;
}
