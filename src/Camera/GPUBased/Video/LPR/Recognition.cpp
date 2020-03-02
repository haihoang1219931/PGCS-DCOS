#include "Recognition.h"

//==============================
// PUBLIC FUNCTIONS            "
//==============================

bool OCR::recognize(cv::Mat plate, const int type)
{
    m_result = "";
    cv::Mat desImg = deskewImage(plate);
    std::string result;
    //    std::cout << "1 Input size : ****" << desImg.size() << std::endl;

    if (type == 0) {
        //        std::cout << "2 Input size : ****" << desImg.size() << std::endl;
        cv::Mat binImg, filIng;
        binarize(desImg, binImg, 9);
        filter(binImg, filIng);
        cv::Mat thresMat0 = ~filIng;
        std::vector<std::vector<cv::Point>> contourPoints0;
        std::vector<cv::Vec4i> hierachy0;
        cv::findContours(thresMat0, contourPoints0, hierachy0, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
        cv::Mat plate1;

        if (removeNoise(filIng, contourPoints0, plate1)) {
            if (!plate1.empty()) {
                cv::Mat newPlate;
                insertPadding(plate1, newPlate, 5, 255);
                cv::Mat thresMat1 = ~newPlate;
                std::vector<std::vector<cv::Point>> contourPoints1;
                std::vector<cv::Vec4i> hierachy1;
                cv::findContours(thresMat1, contourPoints1, hierachy1, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
                std::string t = returnCode(newPlate, contourPoints1);
                std::string tempPlate = rawCorrect(t);

                if (tempPlate.size() > 6) {
                    printf("Go to check condition to update.\n");

                    if (m_counter > 0 || (m_counter == 0 && newPlate.rows > 15))
                        updateCodeTable(tempPlate, m_codeTable);
                }
            }
        }
    }

    return false;
}

void OCR::contrastEnhance(cv::Mat &src, cv::Mat &dst, int dist)
{
    cv::Mat smooth;
    cv::GaussianBlur(src, smooth, cv::Size(0, 0), 3);
    int a, b;
    int val, smt;

    for (int x = 0; x < src.cols; x++)
        for (int y = 0 ; y < src.rows; y++) {
            val = (int) src.at<uchar>(y, x);
            smt = (int) smooth.at<uchar>(y, x);

            if ((val - smt) > dist) smt = smt + (val - smt) * 0.5;

            smt = smt < 0.5 * dist ? 0.5 * dist : smt;
            b = smt + 0.5 * dist;
            b = b > 255 ? 255 : b;
            a = b - dist;
            a = a < 0 ? 0 : a;

            if (val >= a && val <= b) {
                dst.at<uchar>(y, x) = (int)(((val - a) / (0.5 * dist)) * 255);
            } else if (val < a) {
                dst.at<uchar>(y, x) = 0;
            } else if (val > b) {
                dst.at<uchar>(y, x) = 255;
            }
        }
}

cv::Mat OCR::deskewImage(cv::Mat image)
{
    //    std::cout << "Die here\n Size: " << image.size() << std::endl;
    cv::Mat enhanced(image.size(), CV_8UC1);
    contrastEnhance(image, enhanced);
    cv::Mat thresh1;
    cv::threshold(enhanced, thresh1, 90, 255, cv::THRESH_OTSU);
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    //    thresh1 = ~thresh1;
    cv::findContours(thresh1, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    cv::Mat contourMap = cv::Mat::zeros(cv::Size(thresh1.cols, thresh1.rows), CV_8U);
    int bigestContourIdx = -1;
    float bigestContourArea = 0;
    cv::Rect ctBox;
    float ctArea;
    std::vector<std::vector<cv::Point>> charcontours;

    for (int i = 0; i < contours.size(); i++) {
        ctArea = cv::contourArea(contours[i]);

        if (ctArea < 30) {
            continue;
        }

        ctBox = cv::boundingRect(contours[i]);

        if (ctArea > bigestContourArea) {
            bigestContourArea = ctArea;
            bigestContourIdx = i;
        }
    }

    cv::Mat plgray;

    if (bigestContourIdx > 0) {
        cv::RotatedRect boundingBox = cv::minAreaRect(contours[bigestContourIdx]);
        float angle = boundingBox.angle;

        if (angle <= -45.0 && angle >= -90.0) {
            angle = 90.0 + angle;
        }

        if (angle >= 90.0) {
            angle = angle - 90.0;
        }

        if (angle >= 45.0 && angle <= 90.0) {
            angle = 90.0 - angle;
        }

        if (abs(angle) > 4) {
            cv::Point2f center = cv::Point2f((float)thresh1.cols / 2.0, (float)thresh1.rows / 2.0);
            cv::Mat R = cv::getRotationMatrix2D(center, angle, 1.0);
            //            std::cout << "=============================>>> Die here 0 \n";
            cv::warpAffine(image, image, R, thresh1.size(), cv::INTER_CUBIC);
            //            cout << " size of sub [ W - H ]" << (int)boundingBox.size.width << "-" <<
            //                    boundingBox.size.height << endl;
            float ratio = boundingBox.size.width / boundingBox.size.height;

            if (ratio < 0.8) {
                cv::getRectSubPix(image, cv::Size((int)boundingBox.size.height, (int)boundingBox.size.width),
                                  boundingBox.center, plgray);
            } else {
                cv::getRectSubPix(image, cv::Size((int)boundingBox.size.width, (int)boundingBox.size.height),
                                  boundingBox.center, plgray);
            }

            //            std::cout << "=============================>>> Die here 1 \n";
        } else plgray = image.clone();
    } else plgray = image.clone();

    return plgray;
}

/**
 * @brief insertPadding
 * @param input         : image in grayscale need to insert padding.
 * @param output        : padded image.
 * @param padding_size  : band width will be inserted outside image edge.
 * @param padding_value : intensity of padding.
 */
void OCR::insertPadding(cv::Mat input, cv::Mat &output, const int paddingSize, const int paddingValue)
{
    int width = input.cols;
    int height = input.rows;
    //    std::cout << "insert input size : " << input.size() << std::endl;
    output = cv::Mat(height + 2 * paddingSize, width + 2 * paddingSize, input.type(), paddingValue);
    cv::Rect roi(paddingSize, paddingSize, width, height);
    input.copyTo(output(roi));
}

/**
 * @brief binarize
 * @param input     : image in grayscale need to binarize.
 * @param output    : binary image.
 * @param win_size  : size of stride square window.
 */
void OCR::binarize(cv::Mat input, cv::Mat &output, const int windowSize)
{
    cv::Mat newInput = input.clone();
    int paddingSize = windowSize / 2;
    //    std::cout << "Bin input image: " << input.size() << std::endl;
    cv::Mat paddedImg;
    insertPadding(newInput, paddedImg, paddingSize, 127);
    int height      = paddedImg.rows;
    int width       = paddedImg.cols;
    output          = cv::Mat(input.size(), CV_8UC1, 255);
    cv::Mat thress  = cv::Mat::zeros(input.size(), CV_32FC1);
    cv::parallel_for_(cv::Range(0, (width - windowSize + 1) * (height - windowSize + 1)), [&](const cv::Range & range) {
        for (int k = range.start; k < range.end; k++) {
            int i = k / (height - windowSize + 1);
            int j = k % (height - windowSize + 1);
            cv::Scalar mean, stdev;
            cv::Rect roi(i, j , windowSize, windowSize);
            cv::meanStdDev(paddedImg(roi), mean, stdev);

            if (stdev[0] < 6.f)
                thress.at<float>(j, i) = 0.8 * (mean[0] + 2.0 * stdev[0]);
            else
                thress.at<float>(j, i) = mean[0] * (1.f + stdev[0] / 1024.f);
        }
    });
    cv::Mat temp;
    paddedImg(cv::Rect(paddingSize, paddingSize, input.cols, input.rows)).copyTo(temp);
    temp.convertTo(temp, CV_32FC1);
    cv::compare(temp, thress, output, cv::CMP_GE);
}

/**
 * @brief filter
 * @param input     : a raw binary image.
 * @param output    : the image after removing blob noise and edges of the plate.
 * @return          : coordinates of ending points.
 */
void OCR::filter(const cv::Mat input, cv::Mat &output)
{
    int filterSize = 3;
    //    cv::Mat filImg;
    filterBlobNoise(input, output, filterSize);
}

/**
 * @brief filterBlobNoise
 * @param input         : a binary image needed to eleminate blob noise.
 * @param output        : an image that was reduced blob noise.
 * @param windowSize    : size of slide window. Default is 3x3 pixels with stride by 1.
 */
void OCR::filterBlobNoise(const cv::Mat input, cv::Mat &output, const int windowSize)
{
    int paddingSize = windowSize / 2;
    cv::Mat paddedImg;
    insertPadding(input, paddedImg, paddingSize, 255);
    output = paddedImg.clone();
    int rows = paddedImg.rows;
    int cols = paddedImg.cols;

    for (int i = 0; i <= rows - 2 * paddingSize; i++)
        for (int j = 0; j <= cols - 2 * paddingSize; j++)
            if (paddedImg.at<uchar>(i + paddingSize, j + paddingSize) == 0)
                if (paddedImg.at<uchar>(i + paddingSize - 1, j + paddingSize - 1) +
                    paddedImg.at<uchar>(i + paddingSize - 1, j + paddingSize) +
                    paddedImg.at<uchar>(i + paddingSize - 1, j + paddingSize + 1) +
                    paddedImg.at<uchar>(i + paddingSize, j + paddingSize - 1) +
                    paddedImg.at<uchar>(i + paddingSize, j + paddingSize) +
                    paddedImg.at<uchar>(i + paddingSize, j + paddingSize + 1) +
                    paddedImg.at<uchar>(i + paddingSize + 1, j + paddingSize - 1) +
                    paddedImg.at<uchar>(i + paddingSize + 1, j + paddingSize) +
                    paddedImg.at<uchar>(i + paddingSize + 1, j + paddingSize + 1) >= 1020)
                    output.at<uchar>(i + paddingSize, j + paddingSize) = 255;
}

bool OCR::removeNoise(cv::Mat input, std::vector<std::vector<cv::Point> > contoursPoints, cv::Mat &output)
{
    int numContoursPoints = contoursPoints.size();
    int width = input.cols;
    struct Region {
        int numElements;
        float avgHeight;
        std::vector<cv::Rect> rects;
    };
    std::vector<Region> regions;

    if (numContoursPoints > 0) {
        for (int k = 0; k < numContoursPoints; k++) {
            cv::Rect rect = cv::boundingRect(contoursPoints[k]);

            if (rect.width < 0.5 * width && rect.width * rect.height > 60 && rect.height > 12 && rect.width > 4) {
                if (regions.size() < 1) {
                    Region rgn;
                    rgn.numElements = 1;
                    rgn.avgHeight = (float)rect.height;
                    rgn.rects.push_back(rect);
                    regions.push_back(rgn);
                } else {
                    bool isInserted = false;

                    for (uint l = 0; l < regions.size(); l++) {
                        if (std::abs(regions[l].avgHeight - rect.height) <= 2) {
                            regions[l].numElements += 1;
                            regions[l].rects.push_back(rect);
                            regions[l].avgHeight = ((float)(regions[l].numElements - 1) * regions[l].avgHeight + (float)rect.height) / (float)regions[l].numElements;
                            isInserted = true;
                            break;
                        }
                    }

                    if (!isInserted) {
                        Region rgn;
                        rgn.numElements = 1;
                        rgn.avgHeight = (float)rect.height;
                        rgn.rects.push_back(rect);
                        regions.push_back(rgn);
                    }
                }
            }
        }
    } else {
        return false;
    }

    // Find right rectangle
    if (regions.size() > 0) {
        int idx = 0;
        int count = regions[0].numElements;

        for (int i = 0; i < regions.size(); i++) {
            if (regions[i].numElements > count) {
                count = regions[i].numElements;
                idx = i;
            }
        }

        Region rgn = regions[idx];
        int idxMin = 0,
            idxMax = 0;
        int minX = rgn.rects[0].x;
        int maxX = rgn.rects[0].x + rgn.rects[0].width;

        for (int i = 0; i < rgn.numElements; i++) {
            if (rgn.rects[i].x < minX) {
                idxMin = i;
                minX = rgn.rects[i].x;
            }

            if (rgn.rects[i].x + rgn.rects[i].width > maxX) {
                idxMax = i;
                maxX = rgn.rects[i].x + rgn.rects[i].width;
            }
        }

        cv::RotatedRect rotRect;
        cv::Point lPoint(rgn.rects[idxMin].x, rgn.rects[idxMin].y);
        cv::Point rPoint(rgn.rects[idxMax].x, rgn.rects[idxMax].y);
        rotRect.center.x = (lPoint.x + rPoint.x + rgn.rects[idxMax].width) / 2;
        rotRect.center.y = (lPoint.y + rPoint.y + rgn.rects[idxMax].height) / 2;
        cv::Point vtcp(rPoint.x - lPoint.x, (rPoint.y + rgn.rects[idxMax].height / 2) - (lPoint.y + rgn.rects[idxMin].height / 2));
        float lengthVtcp = std::sqrt(vtcp.x * vtcp.x + vtcp.y * vtcp.y);
        rotRect.angle = std::acos((float)vtcp.x / lengthVtcp) / CV_PI * 180.f;

        if (rotRect.angle < 1.0) {
            cv::Point tl(lPoint.x, lPoint.y < rPoint.y ? lPoint.y : rPoint.y);
            cv::Point br(rPoint.x + rgn.rects[idxMax].width,
                         rPoint.y + rgn.rects[idxMax].height > lPoint.y + rgn.rects[idxMin].height ? rPoint.y + rgn.rects[idxMax].height : lPoint.y + rgn.rects[idxMin].height);
            cv::Rect roi = cv::Rect(tl.x, tl.y, br.x - tl.x, br.y - tl.y);
            input(roi).copyTo(output);
            //        cv::imshow("output", output);
        } else {
            if (vtcp.y < 0)
                rotRect.angle = -rotRect.angle;

            //        std::cout << rotRect.angle << std::endl;
            rotRect.size.width = rPoint.x - lPoint.x + rgn.rects[idxMax].width;
            rotRect.size.height = (int)(1.2 * rgn.avgHeight);
            //        cv::Mat M = cv::getRotationMatrix2D(rotRect.center, rotRect.angle, 1.0);
            //        cv::warpAffine(input, output, M, input.size());
            getPatch(input, output, rotRect);
        }

        return true;
    } else {
        return false;
    }
}

int OCR::getPatch(const cv::Mat &_grayImg, cv::Mat &_grayPatch, const cv::RotatedRect _targetBound)
{
    //===== 1. Extract data int the rectangular bounding box of the rotated rectangle
    cv::RotatedRect expandedRotRect = _targetBound;
    expandedRotRect.size.width  *= 1.0;
    expandedRotRect.size.height *= 1.0;
    cv::Rect rect = expandedRotRect.boundingRect();
    // Rectange that is inside the image boundary
    int top     = rect.y,
        left    = rect.x,
        bot     = rect.y + rect.height,
        right   = rect.x + rect.width;

    if (top < 0) top = 0;

    if (left < 0) left = 0;

    if (bot >= _grayImg.rows) bot = _grayImg.rows - 1;

    if (right >= _grayImg.cols) right = _grayImg.cols - 1;

    if ((top >= bot) || (left >= right)) {
        fprintf(stderr, "[ERR] %s:%d: stt = %d: Invalid target ROI\n",
                __FUNCTION__, __LINE__, -1);
        return -1;
    }

    cv::Rect validRect(left, top, right - left, bot - top);
    int deltaTop   = top - rect.y,
        deltaLeft  = left - rect.x,
        deltaBot   = rect.y + rect.height - bot,
        deltaRight = rect.x + rect.width - right;
    // Extract valid image patch
    cv::Mat rectPatch = cv::Mat::zeros(rect.height, rect.width, CV_8UC1);
    cv::copyMakeBorder(_grayImg(validRect), rectPatch, deltaTop, deltaBot, deltaLeft, deltaRight,
                       cv::BORDER_CONSTANT, cv::Scalar::all(0.0));
    //===== 2. Extract rotated patch from its rectangular bounding box patch
    // Compute rotation matrix
    cv::Point2f center = cv::Point2f((float)rectPatch.cols / 2.0, (float)rectPatch.rows / 2.0);
    cv::Mat R = cv::getRotationMatrix2D(center, expandedRotRect.angle, 1.0);
    // Perform warp affine the bounding box patch so that the extracted patch is vertical
    cv::Mat rotated;
    cv::warpAffine(rectPatch, rotated, R, rectPatch.size(), cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(255));
    // Crop the resulting image to obtain RGB rotated image patch
    cv::getRectSubPix(rotated, cv::Size((int)expandedRotRect.size.width, (int)expandedRotRect.size.height),
                      center, _grayPatch);
    return 0;
}

std::string OCR::returnCode(cv::Mat &input, std::vector<std::vector<cv::Point>> contoursPoints)
{
    int numContourPoints = contoursPoints.size();
    std::string code;

    if (numContourPoints > 0) {
        float avgAngle;
        std::vector<float> charAngles;

        for (int k = 0; k < numContourPoints; k++) {
            cv::Rect rect = cv::boundingRect(contoursPoints[k]);

            if (rect.width * rect.height > 75 && rect.height > 0.75 * (input.rows - 10) && rect.width > 4) {
                cv::RotatedRect rotRect = cv::minAreaRect(contoursPoints[k]);
                avgAngle = std::abs(rotRect.angle);

                if (avgAngle < 30)
                    charAngles.push_back(avgAngle);
                else if (avgAngle > 60)
                    charAngles.push_back(avgAngle - 90.f);
            }
        }

        avgAngle = findRightAngle(charAngles);

        if (std::abs(avgAngle) >= 2) {
            cv::Mat M = cv::Mat::zeros(2, 3, CV_32FC1);
            M.at<float>(0, 0) = 1.f;
            M.at<float>(1, 1) = 1.f;
            M.at<float>(0, 1) = std::tan(-avgAngle * CV_PI / 180.0);
            cv::warpAffine(input, input, M, input.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255));
        }

        printf("Set input for recoginization.\n\n");
        m_api->SetImage((uchar *)input.data, input.cols, input.rows, input.channels(), input.step1());

        try {
            code = m_api->GetUTF8Text();
        } catch (...) {
            printf("Throw the exception: ");
        }
    }

    return code;
}

float OCR::findRightAngle(std::vector<float> angles)
{
    int size = angles.size();

    if (size == 0) {
        return 0.f;
    } else if (size == 1) {
        return angles[0];
    } else {
        std::vector<Angles> angleVect;
        Angles a;
        a.s_angle = angles[0];
        a.s_count = 1;
        angleVect.push_back(a);

        for (int i = 1; i < size; i++) {
            bool state = false;

            for (uint j = 0; j < angleVect.size(); j++) {
                if (std::abs(angleVect[j].s_angle - angles[i]) < 1.5) {
                    state = true;
                    angleVect[j].s_angle = (angleVect[j].s_angle + angles[i]) / 2.0;
                    angleVect[j].s_count++;
                    break;
                }
            }

            if (state == false) {
                a.s_angle = angles[i];
                a.s_count = 1;
                angleVect.push_back(a);
            }
        }

        int maxIdx = 0;
        int maxVal = angleVect[0].s_count;

        for (uint i = 1; i < angleVect.size(); i++) {
            if (angleVect[i].s_count > maxVal) {
                maxIdx = i;
                maxVal = angleVect[i].s_count;
            }
        }

        int count = 0;

        for (uint i = 0; i < angleVect.size(); i++) {
            if (angleVect[i].s_count == maxVal) {
                count++;
            }
        }

        return angleVect[maxIdx].s_angle;
    }
}

std::string OCR::rawCorrect(std::string code)
{
    std::string newCode = "";
    std::string newCode0 = "";
    bool haveCharacter = false;

    for (uint i = 0; i < code.size(); i++) {
        if (isDigit(code[i]) || isUpperCase(code[i]))
            newCode += code[i];
        else if (isLowerCase(code[i])) {
            if (code[i] == 'v') {
                newCode += 'V';
                haveCharacter = true;
            } else if (code[i] == 'x') {
                newCode += 'X';
                haveCharacter = true;
            } else if (code[i] == 'u') {
                newCode += 'U';
                haveCharacter = true;
            } else if (code[i] == 'g' || code[i] == 'q' || code[i] == 'y')
                newCode += '9';
            else if (code[i] == 'b')
                newCode += '6';
            else if (code[i] == 's')
                newCode += '5';
            else if (code[i] == 'z')
                newCode += '2';
        }
    }

    int len = newCode.size();

    if (len < 6) {
        if (haveCharacter) {
            for (uint i = 0; i < len; i++) {
                if (newCode[i] == 'A')
                    newCode += '4';
                else if (newCode[i] == 'B')
                    newCode += '8';
                else if (newCode[i] == 'C' || newCode[i] == 'D' || newCode[i] == 'O' || newCode[i] == 'U' || newCode[i] == 'Q')
                    newCode += '0';
                else if (newCode[i] == 'G')
                    newCode += '6';
                else if (newCode[i] == 'I' || newCode[i] == 'T' || newCode[i] == 'Y')
                    newCode += '1';
                else if (newCode[i] == 'J')
                    newCode += '3';
                else if (newCode[i] == 'L')
                    newCode += '4';
                else if (newCode[i] == 'S')
                    newCode += '5';
                else if (newCode[i] == 'Z')
                    newCode += '2';
            }
        }
    } else {
        if (newCode[0] == '0') {
            newCode = '3' + newCode;
            len++;
        }

        for (int i = 0; i < len; i++) {
            if (i == 2) {
                if (newCode[i] == '0')
                    newCode0 += 'D';
                else if (newCode[i] == '1')
                    newCode0 += 'Y';
                else if (newCode[i] == '2')
                    newCode0 += 'Z';
                else if (newCode[i] == '4')
                    newCode0 += 'A';
                else if (newCode[i] == '5')
                    newCode0 += 'S';
                else if (newCode[i] == '6')
                    newCode0 += 'G';
                else if (newCode[i] == '7')
                    newCode0 += 'Z';
                else if (newCode[i] == '8')
                    newCode0 += 'B';
                else if (isUpperCase(newCode[i])) {
                    newCode0 += newCode[i];
                }
            } else {
                if (newCode[i] == 'A')
                    newCode0 += '4';
                else if (newCode[i] == 'B')
                    newCode0 += '8';
                else if (newCode[i] == 'C' || newCode[i] == 'D' || newCode[i] == 'O' || newCode[i] == 'U' || newCode[i] == 'Q')
                    newCode0 += '0';
                else if (newCode[i] == 'G')
                    newCode0 += '6';
                else if (newCode[i] == 'I' || newCode[i] == 'T' || newCode[i] == 'Y')
                    newCode0 += '1';
                else if (newCode[i] == 'J')
                    newCode0 += '3';
                else if (newCode[i] == 'L')
                    newCode0 += '4';
                else if (newCode[i] == 'S')
                    newCode0 += '5';
                else if (newCode[i] == 'Z')
                    newCode0 += '2';
                else if (isDigit(newCode[i]))
                    newCode0 += newCode[i];
            }
        }
    }

    return newCode0;
}

/**
 * @brief OCR::isDigits
 * @param c     : an input character.
 * @return      : true if the input character is a number or false if not.
 */
bool OCR::isDigit(const char c)
{
    if ((int)c >= 48 && (int)c <= 57)
        return true;

    return false;
}

/**
 * @brief OCR::isUpperCase
 * @param c     : an input character.
 * @return      : true if the input character is an upper case or false if not.
 */
bool OCR::isUpperCase(const char c)
{
    if ((int)c >= 65 && (int)c <= 90)
        return true;

    return false;
}

/**
 * @brief OCR::isLowerCase
 * @param c     : an input character.
 * @return      : true if the input character is an lower case or false if not.
 */
bool OCR::isLowerCase(const char c)
{
    if ((int)c >= 97 && (int)c <= 122)
        return true;

    return false;
}

void OCR::updateCodeTable(std::string code, std::vector<std::vector<Char>> &codeTable)
{
    int len = code.size() < codeTable.size() ? code.size() : codeTable.size();
    bool checked = false;

    for (int i = 0; i < len; i++) {
        checked = false;
        int charLen = codeTable[i].size();

        for (int j = 0; j < charLen; j++) {
            if (code[i] == codeTable[i][j].c) {
                codeTable[i][j].score++;
                checked = true;
            }
        }

        if (!checked) {
            Char c;
            c.c = code[i];
            c.score = 1;
            codeTable[i].push_back(c);
        }
    }

    m_counter++;

    if (m_counter == 5) {
        m_finalResult = getResult(m_codeTable, 0);
        m_stop = true;

        if (m_finalResult.size() < 7) {
            m_counter -= 2;
            m_stop = false;
        } /*else {

            m_stop = true;
        }*/
    }
}

std::string OCR::getResult(std::vector<std::vector<Char> > &codeTable, int plateType)
{
    std::string result;

    if (plateType == 0)
        for (int i = 0; i < 8; i++) {
            int charLen = codeTable[i].size();

            if (charLen > 1) {
                result += codeTable[i][1].c;

                if (charLen > 2) {
                    int scoreMax = codeTable[i][1].score;

                    for (int j = 2; j < charLen; j++) {
                        if (codeTable[i][j].score > scoreMax) {
                            result = codeTable[i][j].c;
                        }
                    }
                }
            }
        }

    return result;
}
