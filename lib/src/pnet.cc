#include "mtcnn/pnet.h"

const float P_NET_WINDOW_SIZE = 12.f;
const int P_NET_STRIDE = 2;

const float IMG_MEAN = 127.5f;
const float IMG_INV_STDDEV = 1.f / 128.f;

ProposalNetwork::ProposalNetwork(const ProposalNetwork::Config &config) {
  _net = cv::dnn::readNetFromCaffe(config.protoText, config.caffeModel);
  if (_net.empty()) {
    throw std::invalid_argument("invalid protoText or caffeModel");
  }
  _threshold = config.threshold;
}

ProposalNetwork::~ProposalNetwork() {}

std::vector<Face> ProposalNetwork::buildFaces(const cv::Mat &scores,
                                              const cv::Mat &regressions,
                                              const float scaleFactor,
                                              const float threshold) {

  auto w = scores.size[3];
  auto h = scores.size[2];
  auto size = w * h;

  const float *scores_data = (float *)(scores.data);
  scores_data += size;

  const float *reg_data = (float *)(regressions.data);

  std::vector<Face> boxes;

  for (int i = 0; i < size; i++) {
    if (scores_data[i] >= (threshold)) {
      int y = i / w;
      int x = i - w * y;

      Face faceInfo;
      BBox &faceBox = faceInfo.bbox;

      faceBox.x1 = (float)(x * P_NET_STRIDE) / scaleFactor;
      faceBox.y1 = (float)(y * P_NET_STRIDE) / scaleFactor;
      faceBox.x2 =
          (float)(x * P_NET_STRIDE + P_NET_WINDOW_SIZE - 1.f) / scaleFactor;
      faceBox.y2 =
          (float)(y * P_NET_STRIDE + P_NET_WINDOW_SIZE - 1.f) / scaleFactor;
      faceInfo.regression[0] = reg_data[i];
      faceInfo.regression[1] = reg_data[i + size];
      faceInfo.regression[2] = reg_data[i + 2 * size];
      faceInfo.regression[3] = reg_data[i + 3 * size];
      faceInfo.score = scores_data[i];
      boxes.push_back(faceInfo);
    }
  }

  return boxes;
}

std::vector<Face> ProposalNetwork::run(const cv::Mat &img,
                                       const float minFaceSize,
                                       const float scaleFactor) {

  std::vector<Face> finalFaces;
  float maxFaceSize = static_cast<float>(std::min(img.rows, img.cols));
  float faceSize = minFaceSize;

  while (faceSize <= maxFaceSize) {
    float currentScale = (P_NET_WINDOW_SIZE) / faceSize;
    int imgHeight = std::ceil(img.rows * currentScale);
    int imgWidth = std::ceil(img.cols * currentScale);
    cv::Mat resizedImg;
    cv::resize(img, resizedImg, cv::Size(imgWidth, imgHeight), 0, 0,
               cv::INTER_AREA);

    // feed it to the proposal network
    cv::Mat inputBlob =
        cv::dnn::blobFromImage(resizedImg, IMG_INV_STDDEV, cv::Size(),
                               cv::Scalar(IMG_MEAN, IMG_MEAN, IMG_MEAN), false);

    _net.setInput(inputBlob, "data");

    const std::vector<cv::String> outBlobNames{"conv4-2", "prob1"};
    std::vector<cv::Mat> outputBlobs;

    _net.forward(outputBlobs, outBlobNames);

    cv::Mat regressionsBlob = outputBlobs[0];
    cv::Mat scoresBlob = outputBlobs[1];

    auto faces =
        buildFaces(scoresBlob, regressionsBlob, currentScale, _threshold);

    if (!faces.empty()) {
      faces = Face::runNMS(faces, 0.5f);
    }

    if (!faces.empty()) {
      finalFaces.insert(finalFaces.end(), faces.begin(), faces.end());
    }

    faceSize /= scaleFactor;
  }

  if (!finalFaces.empty()) {
    finalFaces = Face::runNMS(finalFaces, 0.7f);
    if (!finalFaces.empty()) {
      Face::applyRegression(finalFaces, false);
      Face::bboxes2Squares(finalFaces);
    }
  }

  return finalFaces;
}
