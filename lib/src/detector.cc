#include "mtcnn/detector.h"

MTCNNDetector::MTCNNDetector(const ProposalNetwork::Config &pConfig,
                             const RefineNetwork::Config &rConfig,
                             const OutputNetwork::Config &oConfig) {
  _pnet = std::unique_ptr<ProposalNetwork>(new ProposalNetwork(pConfig));
  _rnet = std::unique_ptr<RefineNetwork>(new RefineNetwork(rConfig));
  _onet = std::unique_ptr<OutputNetwork>(new OutputNetwork(oConfig));
}

std::vector<Face> MTCNNDetector::detect(const cv::Mat &img,
                                        const float minFaceSize,
                                        const float scaleFactor) {

  cv::Mat rgbImg;
  if (img.channels() == 3) {
    cv::cvtColor(img, rgbImg, CV_BGR2RGB);
  } else if (img.channels() == 4) {
    cv::cvtColor(img, rgbImg, CV_BGRA2RGB);
  }
  if (rgbImg.empty()) {
    return std::vector<Face>();
  }
  rgbImg.convertTo(rgbImg, CV_32FC3);
  rgbImg = rgbImg.t();

  // Run Proposal Network to find the initial set of faces
  std::vector<Face> faces = _pnet->run(rgbImg, minFaceSize, scaleFactor);

  std::cout << "Total faces after pnet --> " << faces.size() << std::endl;

  // Run Refine network on the output of the Proposal network
  faces = _rnet->run(rgbImg, faces);

  std::cout << "Total faces after rnet --> " << faces.size() << std::endl;

  // Run Output network on the output of the Refine network
  faces = _onet->run(rgbImg, faces);

  std::cout << "Total faces after onet --> " << faces.size() << std::endl;

  for (size_t i = 0; i < faces.size(); ++i) {
    std::swap(faces[i].bbox.x1, faces[i].bbox.y1);
    std::swap(faces[i].bbox.x2, faces[i].bbox.y2);
    for (int p = 0; p < NUM_PTS; ++p) {
      std::swap(faces[i].ptsCoords[2 * p], faces[i].ptsCoords[2 * p + 1]);
    }
  }

  return faces;
}
