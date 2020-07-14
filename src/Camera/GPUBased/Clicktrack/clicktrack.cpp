#include "clicktrack.h"

ClickTrack::ClickTrack()
{
    m_plateDetector = new PlateDetector();
}

void ClickTrack::setDetector(Detector *_detector)
{
    m_plateDetector->setDetector( _detector );
}

std::string ClickTrack::getPlateNumber_I420()
{
    return m_recognizor->m_result;
}

void ClickTrack::setOCR(OCR* _OCR)
{
    m_recognizor = _OCR;
}

int ClickTrack::updateNewImage_I420(image_t input, cv::Mat h_gray, cv::Rect objectPosition, cv::Mat bgr_img)
{
//    printf("1***********************************************************************************\n");
    ///////////////////////////////
    /// \brief first get plate bounding box
    ///////////////////////////////

    bbox_t objTrack;
    bool isResultValid = m_plateDetector->detect_I420(input, objectPosition, objTrack);
    if (!isResultValid)
    {
//        printf("No Plate\n");
        return PLATE_FAIL;
    }
    // bien so is objTrack
//    return PLATE_DETECTED;

    // Check type of plate: White or blue?
    cv::Rect p_rect(objTrack.x >= 0 ? objTrack.x : 0,
                    objTrack.y >= 0 ? objTrack.y : 0,
                    objTrack.x + objTrack.w > h_gray.cols ? h_gray.cols - objTrack.x : objTrack.w,
                    objTrack.y + objTrack.h > h_gray.rows ? h_gray.rows - objTrack.y : objTrack.h);
    cv::Mat bgr_plate = bgr_img(p_rect).clone();
    double minVal, maxVal;
    cv::Mat float_plate;
    int blue = 0;
    int white = 0;
    int black = 0;
    cv::minMaxLoc(bgr_plate, &minVal, &maxVal);
    bgr_plate.convertTo(float_plate, CV_32FC3);

    for(uint r = 0; r < float_plate.rows; r++)
        for(uint c = 0; c  < float_plate.cols; c++)
        {
            for(int d = 0; d < 3; d++)
                float_plate.at<float>(r, 3 * c + d) = std::floor(255 * (float_plate.at<float>(r, 3 * c + d) - minVal) / (maxVal - minVal));
            if(std::abs(float_plate.at<float>(r, 3 * c) - float_plate.at<float>(r, 3 * c + 1)) >= 50.f || std::abs(float_plate.at<float>(r, 3 * c) - float_plate.at<float>(r, 3 * c + 2)) >= 50.f)
                blue++;
            else
            {
                if(float_plate.at<float>(r, 3 * c) < 50.f && float_plate.at<float>(r, 3 * c + 1) < 50.f && float_plate.at<float>(r, 3 * c + 2) < 50.f)
                    black++;
                else
                    white++;
            }
        }

    float blue_rate = (float)blue / (float)(bgr_plate.rows * bgr_plate.cols - black);
//    float white_rate = (float)white / (float)(bgr_plate.rows * bgr_plate.cols - black);
//    cv::imwrite("/home/pgcs-01/Desktop/giap0.png", bgr_plate, {CV_IMWRITE_PNG_COMPRESSION, 0});

    ///////////////////////////////
    /// \brief extract host gray plate image
    ///////////////////////////////
    // expand the plate area
//    printf("%d-%d-%d-%d\n", objTrack.x, objTrack.y, objTrack.w, objTrack.h);

    int dw = (int)(0.1 * objTrack.w);
    int dh = (int)(0.15 * objTrack.h);
    cv::Rect plateArea(objTrack.x >= dw ? objTrack.x - dw : 0,
                       objTrack.y >= dh ? objTrack.y - dh : 0,
                       objTrack.x + objTrack.w + dw > h_gray.cols ? h_gray.cols - objTrack.x + dw : objTrack.w + 2 * dw,
                       objTrack.y + objTrack.h + dh > h_gray.rows ? h_gray.rows - objTrack.y + dh : objTrack.h + 2 * dh);
    cv::Mat plateImage = h_gray(plateArea).clone();
//    cv::imwrite("img/plateImage" + std::to_string(rand()) + ".png", plateImage);
//    printf("Get plate successfully\n");

    if(!plateImage.empty())
    {
        if(blue_rate >= 0.1f)
        {
            plateImage = ~plateImage;
        }
        int plateType = objTrack.obj_id;
        int sign = -1;
//        std::cout << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXxxx" << plateImage.size() << std::endl;
        std::vector<cv::Mat> chars = preprocess(plateImage, plateType, &sign);
//        printf("Preprocess successfully with charsSize = %d\n", chars.size());
        if(chars.size() > 6)
        {
            std::string code = m_recognizor->recognize(chars, sign);
//            printf("Recognize successfully\n");
            int cc = 0;
            for(uint l = 0; l < code.size(); l++)
            {
                if(code[l] != '_')
                    cc++;
            }
            if(cc > 7)
            {
//                cv::putText(, code, cv::Point(roi.x, roi.y - 2), CV_FONT_HERSHEY_SIMPLEX, 3.f, cv::Scalar(0, 0, 255), 3);
                m_recognizor->combine(code);
                m_recognizor->m_counter++;
                if(m_recognizor->m_counter > 4)
                {
                    m_recognizor->m_result = m_recognizor->findBest();
                    return PLATE_SUCCESS;
                }
            }
        }
        else
        {
//            printf("Char = 0\n");
            return PLATE_FAIL;
        }
    }

    std::string plateReturn("hehehe ");
    plateReturn += isResultValid ? "true" : "false";
//    plateNumber = plateReturn;
    return PLATE_FAIL;
}

void ClickTrack::resetSequence()
{
    m_recognizor->m_counter = 0;
    m_recognizor->m_codeVector.clear();
    m_recognizor->m_result = "";
}
