#ifndef PLATEDETECTION_HPP
#define PLATEDETECTION_HPP
#include "Structures.hpp"
#include <algorithm>
#include <math.h>

class PlateDetection {
public:
  PlateDetection() {}
  ~PlateDetection() {}
  int numberPls;

public:
  std::vector<possiblePlate> detectPlate(cv::Mat img, int min_area,
                                         int max_area, int min_ratio,
                                         int max_ratio);

private:
  // Main functions
  void enhance(const cv::Mat1b &src, cv::Mat1b &dst, int step = 1,
               cv::Vec2i in = cv::Vec2i(0, 255),
               cv::Vec2i out = cv::Vec2i(0, 255));
  std::vector<possibleChar> detectChars(cv::Mat img);
  std::vector<possiblePlate>
  findGoodCandidates(cv::Mat candidate, std::vector<possibleChar> characters);
  std::vector<std::vector<possiblePlate>>
  groupSampPlates(std::vector<possiblePlate> plates);
  int getPatch(const cv::Mat &_grayImg, cv::Mat &_grayPatch,
               const cv::RotatedRect _targetBound);
  // Supported functions
  bool isChar(possibleChar pchar);
  std::vector<possibleChar> matchChars(possibleChar ref_char,
                                       std::vector<possibleChar> characters);
  bool isSameChars(possibleChar char1, possibleChar char2);
  float calcEuclidianDist(possibleChar char1, possibleChar char2);
  float calcInline(possibleChar char1, possibleChar char2);
  bool isSameWords(std::vector<possibleChar> matched_chars,
                   std::vector<possibleChar> last_matched_chars);
  float calcIoU(cv::Rect rect1, cv::Rect rect2);
  bool isInsideRotRect(cv::Point2f point, cv::Point2f corners[]);
  double product(cv::Point2f checked_point, cv::Point2f ref_point1,
                 cv::Point2f ref_point2);
};

#endif // DETECTION_HPP
