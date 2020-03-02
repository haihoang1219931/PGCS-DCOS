#include "Detection.h"

//=========================
// PUBLIC FUNCTIONS       "
//=========================

/**
 * @brief PlateDetection::detectPlate
 * @param img       : input object image.
 * @param min_area  : the possible minimal value of plate area.
 * @param max_area  : the possible maximal value of plate area.
 * @param min_ratio : the possible minimal ratio of plate width by plate height.
 * @param max_ratio : the possible maximal ratio of plate width by plate height.
 * @return          : detected plates.
 */
std::vector<possiblePlate>
PlateDetection::detectPlate(const cv::Mat img, int min_area, int max_area,
                            int min_ratio, int max_ratio) {
  std::vector<possiblePlate> final_plates;
  int width = img.cols;
  int height = img.rows;
  bool is_scale = false;
  float scale = 1;
  cv::Mat scaled_img = img.clone();

  if (width > 1000 || height > 800) {
    is_scale = true;
    scale = 2.5;
    cv::resize(img, scaled_img,
               cv::Size((int)width / scale, (int)height / scale));
  } else if ((600 < width && width <= 1000) ||
             (500 < height && height <= 800)) {
    is_scale = true;
    scale = 1.5;
    cv::resize(img, scaled_img,
               cv::Size((int)width / scale, (int)height / scale));
  }

  cv::Mat hsv_img;
  std::vector<cv::Mat> channels;
  cv::cvtColor(scaled_img, hsv_img, cv::COLOR_RGB2HSV);
  cv::split(hsv_img, channels);

  cv::Mat1b byte_img = channels[2].clone();
  cv::Mat1b enhanced_img;
  enhance(byte_img, enhanced_img);
  hsv_img.release();
  byte_img.release();

  cv::Mat kernel1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(19, 21));
  cv::morphologyEx(enhanced_img, enhanced_img, cv::MORPH_TOPHAT, kernel1);

  cv::Mat blurred_img;
  cv::blur((cv::Mat)enhanced_img, blurred_img, cv::Size(3, 3));
  enhanced_img.release();

  cv::Mat thresh;
  cv::threshold(blurred_img, thresh, 50, 255, CV_THRESH_BINARY_INV);

  cv::Mat kernel2;
  kernel2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  cv::erode(thresh, thresh, kernel2, cv::Point(-1, -1), 2);
  cv::dilate(thresh, thresh, kernel2, cv::Point(-1, -1), 2);

  cv::Mat labels, stats, centroids;
  int num_labels = cv::connectedComponentsWithStats(~thresh, labels, stats,
                                                    centroids, 4, CV_32S);

  for (int i = 1; i < num_labels - 2; i++) {
    int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
    int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
    int area = stats.at<int>(i, cv::CC_STAT_AREA);
    float r = float(w / h);
    cv::Rect rect = cv::Rect(stats.at<int>(i, cv::CC_STAT_LEFT),
                             stats.at<int>(i, cv::CC_STAT_TOP), w, h);

    if (area > min_area && r < max_ratio) {
      cv::Mat candidate_img = scaled_img(rect);
      std::vector<possibleChar> possible_chars = detectChars(candidate_img);
      if (possible_chars.size() >= 3) {
        std::vector<possiblePlate> candidates =
            findGoodCandidates(candidate_img, possible_chars);
        if (candidates.size() > 0)
          for (uint j = 0; j < candidates.size(); j++) {
            possiblePlate plate = candidates[j];
            cv::Rect out_rect(
                plate.s_plate_box.x + rect.x, plate.s_plate_box.y + rect.y,
                plate.s_plate_box.width, plate.s_plate_box.height);

            cv::RotatedRect rot_rect = plate.s_plate_rot_box;
            rot_rect.center.x += rect.x;
            rot_rect.center.y += rect.y;

            cv::Mat outpoints =
                cv::Mat(plate.s_plate_angle_point.size(), CV_32F);

            outpoints.at<float>(0, 0) =
                plate.s_plate_angle_point.at<float>(0, 0) + rect.x;
            outpoints.at<float>(0, 1) =
                plate.s_plate_angle_point.at<float>(0, 1) + rect.y;
            outpoints.at<float>(1, 0) =
                plate.s_plate_angle_point.at<float>(1, 0) + rect.x;
            outpoints.at<float>(1, 1) =
                plate.s_plate_angle_point.at<float>(1, 1) + rect.y;
            outpoints.at<float>(2, 0) =
                plate.s_plate_angle_point.at<float>(2, 0) + rect.x;
            outpoints.at<float>(2, 1) =
                plate.s_plate_angle_point.at<float>(2, 1) + rect.y;
            outpoints.at<float>(3, 0) =
                plate.s_plate_angle_point.at<float>(3, 0) + rect.x;
            outpoints.at<float>(3, 1) =
                plate.s_plate_angle_point.at<float>(3, 1) + rect.y;

            if (is_scale) {
              out_rect.width *= scale;
              out_rect.height *= scale;
              out_rect.x *= scale;
              out_rect.y *= scale;

              rot_rect.center.x *= scale;
              rot_rect.center.y *= scale;
              rot_rect.size.width *= scale;
              rot_rect.size.height *= scale;

              outpoints.at<float>(0, 0) *= scale;
              outpoints.at<float>(0, 1) *= scale;
              outpoints.at<float>(1, 0) *= scale;
              outpoints.at<float>(1, 1) *= scale;
              outpoints.at<float>(2, 0) *= scale;
              outpoints.at<float>(2, 1) *= scale;
              outpoints.at<float>(3, 0) *= scale;
              outpoints.at<float>(3, 1) *= scale;
            }

            plate.s_plate_id = i;
            plate.s_plate_box = out_rect;
            plate.s_plate_rot_box = rot_rect;
            plate.s_plate_angle_point = outpoints;
            getPatch(img, plate.s_plate_gray_img, plate.s_plate_rot_box);
            final_plates.push_back(plate);
          }
      }
    }
  }

  //    std::cout << "size of final_plates: " << final_plates.size() <<
  //    std::endl;
  std::vector<possiblePlate> outplates;
  if (final_plates.size() <= 1) {
    outplates = final_plates;
  } else {
    std::vector<std::vector<possiblePlate>> same_plates_groups =
        groupSampPlates(final_plates);
    for (uint i = 0; i < same_plates_groups.size(); i++) {
      std::vector<possiblePlate> same_plates = same_plates_groups[i];
      std::sort(same_plates.begin(), same_plates.end(),
                [](possiblePlate a, possiblePlate b) {
                  return a.s_plate_num_chars > b.s_plate_num_chars;
                });
      outplates.push_back(same_plates[0]);
    }
  }

  for (uint i = 0; i < outplates.size(); i++)
    for (uint j = 1; j < outplates.size(); j++) {
      double distance = std::sqrt((outplates[i].s_plate_rot_box.center.x -
                                   outplates[j].s_plate_rot_box.center.x) *
                                      (outplates[i].s_plate_rot_box.center.x -
                                       outplates[j].s_plate_rot_box.center.x) +
                                  (outplates[i].s_plate_rot_box.center.y -
                                   outplates[j].s_plate_rot_box.center.y) *
                                      (outplates[i].s_plate_rot_box.center.y -
                                       outplates[j].s_plate_rot_box.center.y));
      if (distance < 0.75 * (outplates[i].s_plate_rot_box.size.width +
                             outplates[j].s_plate_rot_box.size.width))
        outplates[j].s_plate_id = outplates[i].s_plate_id;
    }

  return outplates;
}

//=========================
// PRIVATE FUNCTIONS      "
//=========================
/**
 * @brief PlateDetection::enhance
 * @param src   : input byte image.
 * @param dst   : output byte image after equalizing histogram.
 * @param step  : distance of stride when equalizing histogram. Default is 1.
 * @param in    : unknown.
 * @param out   : unknown.
 */
void PlateDetection::enhance(const cv::Mat1b &src, cv::Mat1b &dst, int step,
                             cv::Vec2i in, cv::Vec2i out) {
  dst = src.clone();
  step = std::max(0, std::min(100, step));

  if (step > 0) {
    std::vector<int> hist(256, 0);
    for (int i = 0; i < src.rows; i++)
      for (int j = 0; j < src.cols; j++)
        hist[src(i, j)]++;

    std::vector<int> cum = hist;
    for (uint i = 1; i < hist.size(); i++)
      cum[i] = cum[i - 1] + hist[i];

    int total = src.cols * src.rows;
    int low_bound = (int)total * step / 100;
    int high_bound = (int)total * (100 - step) / 100;
    in[0] = std::distance(cum.begin(),
                          std::lower_bound(cum.begin(), cum.end(), low_bound));
    in[1] = std::distance(cum.begin(),
                          std::lower_bound(cum.begin(), cum.end(), high_bound));
  }

  float scale = float(out[1] - out[0]) / float(in[1] - in[0]);
  for (int i = 0; i < dst.rows; i++)
    for (int j = 0; j < dst.cols; j++) {
      int vs = std::max(src(i, j) - in[0], 0);
      int vd = std::min(int(vs * scale + 0.5) + out[0], out[1]);
      dst(i, j) = cv::saturate_cast<uchar>(vd);
    }
}

/**
 * @brief PlateDetection::detectChars
 * @param img       : a RGB input image that bounds object.
 * @return          : a vector of contour points of image patch considered as
 * characters.
 */
std::vector<possibleChar> PlateDetection::detectChars(cv::Mat img) {
  // 1. Declare parameters
  std::vector<possibleChar>
      found_chars;               /**< saves found characters in input image */
  cv::Mat hsv_img;               /**< image in HSV colorspace */
  std::vector<cv::Mat> channels; /**< components of HSV colorspace */
  cv::Mat top_hat, black_hat;
  cv::Mat addition, subtraction;
  cv::Mat blur_img; /**< blurred image by Gaussian filter */
  cv::Mat thres_img;
  std::vector<std::vector<cv::Point>>
      contour_points; /**< a vector contains sets of points belonging to
                         contours of different charaters */
  std::vector<cv::Vec4i> hierarchy;

  // 2. Convert colorspace
  cv::cvtColor(img, hsv_img, cv::COLOR_RGB2HSV);
  cv::split(hsv_img, channels);

  // 3. Apply connected components and find contour points
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  cv::morphologyEx(channels[2], top_hat, cv::MORPH_TOPHAT, kernel);
  cv::morphologyEx(channels[2], black_hat, cv::MORPH_BLACKHAT, kernel);

  cv::add(channels[2], top_hat, addition);
  cv::subtract(addition, black_hat, subtraction);
  cv::GaussianBlur(subtraction, blur_img, cv::Size(3, 3), 0);
  cv::adaptiveThreshold(blur_img, thres_img, 255,
                        cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV,
                        7, 5);

  cv::findContours(thres_img, contour_points, hierarchy, CV_RETR_LIST,
                   CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

  for (uint i = 0; i < contour_points.size(); i++) {
    possibleChar pchar = possibleChar(contour_points[i]);
    if (isChar(pchar))
      found_chars.push_back(pchar);
  }

  return found_chars;
}

/**
 * @brief PlateDetection::findGoodCandidates
 * @param candidate     : the candidate as a plate should be certificated.
 * @param characters    : characters detected inside the candidate.
 * @return              : a verified candidate.
 */
std::vector<possiblePlate>
PlateDetection::findGoodCandidates(cv::Mat candidate,
                                   std::vector<possibleChar> characters) {
  std::vector<possiblePlate> good_plates;
  std::vector<std::vector<possibleChar>> matched_chars_list;

  // Integrate possible characters into a block
  for (uint k = 0; k < characters.size(); k++) {
    std::vector<possibleChar> matched_chars =
        matchChars(characters[k], characters);
    if (matched_chars.size() >= 3) {
      std::vector<possibleChar> correctchars = matched_chars;
      std::sort(correctchars.begin(), correctchars.end(),
                [](possibleChar a, possibleChar b) {
                  return a.s_char_centerX < b.s_char_centerX;
                });

      if (matched_chars_list.size() == 0) {
        matched_chars_list.push_back(correctchars);
      } else {
        if (!isSameWords(correctchars, matched_chars_list.back()))
          matched_chars_list.push_back(correctchars);
      }
    }
  }

  // Process affine transformations
  for (uint i = 0; i < matched_chars_list.size(); i++) {
    std::vector<possibleChar> correctchars = matched_chars_list[i];
    possibleChar head_char = correctchars[0];
    possibleChar rear_char = correctchars[correctchars.size() - 1];

    int centerX =
        (int)(head_char.s_char_centerX + rear_char.s_char_centerX) / 2.0;
    int centerY =
        (int)(head_char.s_char_centerY + rear_char.s_char_centerY) / 2.0;
    int plate_width =
        (int)((rear_char.s_char_box.x + rear_char.s_char_box.width -
               head_char.s_char_box.x) *
              1.2);
    int sum = 0;

    for (uint p = 0; p < correctchars.size(); p++) {
      sum += correctchars[p].s_char_box.height;
    }
    int plate_height = (int)((sum / correctchars.size()) * 1.5);

    int dy = rear_char.s_char_centerY - head_char.s_char_centerY;
    float dist = calcEuclidianDist(head_char, rear_char);
    float correct_angle = (float)asin(dy / dist) * (180.0 / CV_PI);

    cv::RotatedRect rot_rect = cv::RotatedRect(
        cv::Point2f((float)centerX, (float)centerY),
        cv::Size2f((float)plate_width, (float)plate_height), correct_angle);
    cv::Rect bounding_rect =
        cv::Rect((int)centerX - plate_width / 2.0,
                 (int)centerY - plate_height / 2.0, plate_width, plate_height);
    float ratio = (float)bounding_rect.width / bounding_rect.height;

    if (ratio >= MIN_RATIO_PLATE_SIZE && ratio <= MAX_RATIO_PLATE_SIZE) {
      cv::Mat angle_points;
      cv::boxPoints(rot_rect, angle_points);
      //            cv::Mat rot_mat =
      //            cv::getRotationMatrix2D(cv::Point2f(centerX, centerY),
      //            correct_angle, 1.0);

      //            cv::Mat rot_img;
      //            cv::warpAffine(candidate,  rot_img, rot_mat,
      //            cv::Size(candidate.cols, candidate.rows));

      //            cv::Mat croppped_img;
      //            cv::getRectSubPix(rot_img, cv::Size(plate_width,
      //            plate_height), cv::Point(centerX, centerY), croppped_img);

      possiblePlate plate;
      //            plate.s_plate_gray_img =  croppped_img;
      plate.s_plate_angle_point = angle_points;
      plate.s_plate_box = bounding_rect;
      plate.s_plate_num_chars = correctchars.size();
      plate.s_plate_rot_box = rot_rect;

      good_plates.push_back(plate);
    }
  }
  return good_plates;
}

/**
 * @brief PlateDetection::groupSampPlates
 * @param plates    : a vector of plates.
 * @return          : a vector of group of same plates.
 */
std::vector<std::vector<possiblePlate>>
PlateDetection::groupSampPlates(std::vector<possiblePlate> plates) {
  std::vector<std::vector<possiblePlate>> plate_groups;
  std::vector<possiblePlate> same_plates;

  bool *checked = new bool[plates.size()];

  for (uint i = 0; i < plates.size(); i++)
    checked[i] = false;

  for (uint i = 0; i < plates.size(); i++) {
    plates[i].s_plate_id = i;
    if (!checked[i]) {
      checked[i] = true;
      for (uint j = i + 1; j < plates.size(); j++)
        if (!checked[j]) {
          float iou = calcIoU(plates[i].s_plate_box, plates[j].s_plate_box);
          if (iou >= 0.2) {
            checked[j] = true;
            plates[j].s_plate_id = i;
            same_plates.push_back(plates[j]);
          }
        }
      same_plates.push_back(plates[i]);
      plate_groups.push_back(same_plates);
      same_plates.clear();
    }
  }
  delete[] checked;
  return plate_groups;
}

/**
 * @brief PlateDetection::isChar
 * @param pchar     : a candidate for character.
 * @return          : true if candidate is character; false if not.
 */
bool PlateDetection::isChar(possibleChar pchar) {
  if (pchar.s_char_area > MIN_AREA_CHAR &&
      pchar.s_char_box.width > MIN_WIDTH_CHAR &&
      pchar.s_char_box.height > MIN_HEIGHT_CHAR &&
      pchar.s_char_ratio > MIN_RATIO_CHAR_SIZE &&
      pchar.s_char_ratio < MAX_RATIO_PLATE_SIZE)
    return true;
  return false;
}

/**
 * @brief PlateDetection::matchChars
 * @param ref_char      : the reference character.
 * @param characters    : a vector of characters needed to compare with the
 * reference character.
 * @return              : a vector of characters that are the same line with the
 * reference char.
 */
std::vector<possibleChar>
PlateDetection::matchChars(possibleChar ref_char,
                           std::vector<possibleChar> characters) {
  std::vector<possibleChar>
      online_chars; /**< contains characters on the same line */

  for (uint i = 0; i < characters.size(); i++) {
    if (!isSameChars(ref_char, characters[i])) {
      float dist = calcEuclidianDist(characters[i], ref_char);
      float angle = calcInline(characters[i], ref_char);
      float dArea =
          (float)std::abs(characters[i].s_char_area - ref_char.s_char_area) /
          ref_char.s_char_area;
      float dWidth = (float)std::abs(characters[i].s_char_box.width -
                                     ref_char.s_char_box.width) /
                     ref_char.s_char_box.width;
      float dHeight = (float)std::abs(characters[i].s_char_box.height -
                                      ref_char.s_char_box.height) /
                      ref_char.s_char_box.height;
      float iou = calcIoU(ref_char.s_char_box, characters[i].s_char_box);

      if (dist < ref_char.s_char_diagonal * 5.0 &&
          std::abs(angle) < ANGLE_CENTROID_CHAR &&
          dArea < DIFF_AREA_BETWEEN_CHARS &&
          dWidth < DIFF_WIDTH_BETWEEN_CHARS &&
          dHeight < DIFF_HEIGHT_BETWEEN_CHARS && iou < IOU_CHARS) {
        characters[i].s_char_index = i;
        online_chars.push_back(characters[i]);
      }
    }
  }
  online_chars.push_back(ref_char);
  return online_chars;
}

/**
 * @brief PlateDetection::isSameChars
 * @param char1     : the first possible character.
 * @param char2     : the second possible character.
 * @return          : true if 2 characters are identity else false.
 */
bool PlateDetection::isSameChars(possibleChar char1, possibleChar char2) {
  if (char1.s_char_area == char2.s_char_area &&
      char1.s_char_box.x == char2.s_char_box.x &&
      char1.s_char_box.y == char2.s_char_box.y &&
      char1.s_char_centerX == char2.s_char_centerX &&
      char1.s_char_centerY == char2.s_char_centerY) {
    return true;
  }
  return false;
}

/**
 * @brief PlateDetection::calcEuclidianDist
 * @param char1     : the first possible character.
 * @param char2     : the second possible character.
 * @return          : the distance between 2 characters considered as 2 central
 * points.
 */
float PlateDetection::calcEuclidianDist(possibleChar char1,
                                        possibleChar char2) {
  float dx = float(std::abs(char1.s_char_centerX - char2.s_char_centerX));
  float dy = float(std::abs(char1.s_char_centerY - char2.s_char_centerY));
  return std::sqrt(dx * dx + dy * dy);
}

/**
 * @brief calcInline
 * @param char1     : the first possible character.
 * @param char2     : the second possible character.
 * @return          : the angle in degree is formed by 2 vertical axes of 2
 * characters.
 */
float PlateDetection::calcInline(possibleChar char1, possibleChar char2) {
  cv::Point2f rot_box_points[4];
  char1.s_char_rot_box.points(rot_box_points);

  cv::Point2f vec1 =
      cv::Vec2f(char2.s_char_rot_box.center.x, char2.s_char_rot_box.center.y) -
      cv::Vec2f(char1.s_char_rot_box.center.x, char1.s_char_rot_box.center.y);
  cv::Point2f vec2 =
      -cv::Vec2f(char1.s_char_box.tl().x, char1.s_char_box.br().y) +
      cv::Vec2f(char1.s_char_box.br().x, char1.s_char_box.br().y);

  float dot = vec1.x * vec2.x + vec1.y * vec2.y;
  float det = vec1.x * vec2.y - vec1.y * vec2.x;
  return ((float)std::atan2(det, dot) * 180.f / CV_PI);
}

/**
 * @brief PlateDetection::isSameWords
 * @param matched_chars     : the reference word.
 * @param last_matched_chars: the checked word.
 * @return                  : true if they are the same word else false.
 */
bool PlateDetection::isSameWords(std::vector<possibleChar> matched_chars,
                                 std::vector<possibleChar> last_matched_chars) {
  if (matched_chars.size() == last_matched_chars.size()) {
    uint num_same_chars = 0;
    for (uint i = 0; i < matched_chars.size(); i++) {
      if (isSameChars(matched_chars[i], last_matched_chars[i]))
        num_same_chars++;
    }
    if (num_same_chars == matched_chars.size())
      return true;
  }
  return false;
}

/**
 * @brief PlateDetection::calcIoU
 * @param rect1     : rectangle bounds the first textline.
 * @param rect2     : rectangle bounds the second textline.
 * @return          : area ratio of intersection part by union part of 2
 * textlines.
 */
float PlateDetection::calcIoU(cv::Rect rect1, cv::Rect rect2) {
  cv::Point tl_bbox1(rect1.x, rect1.y);
  cv::Point br_bbox1(rect1.x + rect1.width, rect1.y + rect1.height);
  cv::Point tl_bbox2(rect2.x, rect2.y);
  cv::Point br_bbox2(rect2.x + rect2.width, rect2.y + rect2.height);

  int p1x = std::max(tl_bbox1.x, tl_bbox2.x);
  int p1y = std::max(tl_bbox1.y, tl_bbox2.y);
  int p2x = std::min(br_bbox1.x, br_bbox2.x);
  int p2y = std::min(br_bbox1.y, br_bbox2.y);

  int intersection_area =
      std::max(0, p2x - p1x + 1) * std::max(0, p2y - p1y + 1);
  if (intersection_area == rect1.area() || intersection_area == rect2.area()) {
    return 1.f;
  }
  int union_area = rect1.area() + rect2.area() - intersection_area;
  return (float)intersection_area / union_area;
}
int PlateDetection::getPatch(const cv::Mat &_grayImg, cv::Mat &_grayPatch,
                             const cv::RotatedRect _targetBound) {
  //===== 1. Extract data int the rectangular bounding box of the rotated
  //rectangle
  cv::RotatedRect expandedRotRect = _targetBound;
  expandedRotRect.size.width *= 1.0;
  expandedRotRect.size.height *= 1.0;

  cv::Rect rect = expandedRotRect.boundingRect();

  // Rectange that is inside the image boundary
  int top = rect.y, left = rect.x, bot = rect.y + rect.height,
      right = rect.x + rect.width;
  if (top < 0)
    top = 0;
  if (left < 0)
    left = 0;
  if (bot >= _grayImg.rows)
    bot = _grayImg.rows - 1;
  if (right >= _grayImg.cols)
    right = _grayImg.cols - 1;

  if ((top >= bot) || (left >= right)) {
    fprintf(stderr, "[ERR] %s:%d: stt = %d: Invalid target ROI\n", __FUNCTION__,
            __LINE__, -1);
    return -1;
  }

  cv::Rect validRect(left, top, right - left, bot - top);

  int deltaTop = top - rect.y, deltaLeft = left - rect.x,
      deltaBot = rect.y + rect.height - bot,
      deltaRight = rect.x + rect.width - right;

  // Extract valid image patch
  cv::Mat rectPatch = cv::Mat::zeros(rect.height, rect.width, CV_8UC1);
  cv::copyMakeBorder(_grayImg(validRect), rectPatch, deltaTop, deltaBot,
                     deltaLeft, deltaRight, cv::BORDER_CONSTANT,
                     cv::Scalar::all(0.0));

  //===== 2. Extract rotated patch from its rectangular bounding box patch
  // Compute rotation matrix
  cv::Point2f center =
      cv::Point2f((float)rectPatch.cols / 2.0, (float)rectPatch.rows / 2.0);
  cv::Mat R = cv::getRotationMatrix2D(center, expandedRotRect.angle, 1.0);

  // Perform warp affine the bounding box patch so that the extracted patch is
  // vertical
  cv::Mat rotated;
  cv::warpAffine(rectPatch, rotated, R, rectPatch.size(), cv::INTER_CUBIC);

  // Crop the resulting image to obtain RGB rotated image patch
  cv::getRectSubPix(rotated,
                    cv::Size((int)expandedRotRect.size.width,
                             (int)expandedRotRect.size.height),
                    center, _grayPatch);
  return 0;
}

/**
 * @brief PlateDetection::isInsideRotRect
 * @param point     : a point need checked.
 * @param corners   : coordinates of 4 corners of the rotated rectangle checked.
 * @return          : true if it is inside the rotated rectangle or false if
 * others.
 */
bool PlateDetection::isInsideRotRect(cv::Point2f point, cv::Point2f corners[]) {
  double prod[4];
  for (int i = 0; i < 4; i++)
    prod[i] = product(point, corners[i], corners[(i + 1) % 4]);

  if (prod[0] * prod[2] < 0 && prod[1] * prod[3] < 0)
    return true;
  return false;
}

/**
 * @brief PlateDetection::product
 * @param checked_point
 * @param ref_point1
 * @param ref_point2
 * @return
 */
double PlateDetection::product(cv::Point2f checked_point,
                               cv::Point2f ref_point1, cv::Point2f ref_point2) {
  double slop = (ref_point1.y - ref_point2.y) / (ref_point1.x - ref_point2.x);
  double intercept = ref_point1.y - slop * ref_point1.x;
  return (slop * checked_point.x - checked_point.y + intercept) /
         std::sqrt(slop * slop + intercept * intercept);
}
