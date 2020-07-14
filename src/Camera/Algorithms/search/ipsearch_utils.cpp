#include "ipsearch_utils.h"
#include <fstream>
#include <iostream>

namespace ip {

namespace objsearch {

    void drawBoudingBox(cv::Mat image, std::vector<cv::Point2f> bb)
    {
        for(unsigned i = 0; i < bb.size() - 1; i++)
        {
            line(image, bb[i], bb[i+1], cv::Scalar(0, 0, 255), 2);
        }
        line(image, bb[bb.size() - 1], bb[0], cv::Scalar(0, 0, 255), 2);
    }

    void drawBoundingCirle(cv::Mat image, cv::Point2f center, int radius)
    {
        cv::circle(image, center, radius, cv::Scalar(0, 0, 255), 2);
    }

    void drawStatistics(cv::Mat image, const Stats& stats, double fps)
    {
        static const int front = cv::FONT_HERSHEY_PLAIN;
        std::stringstream str1, str2, str3, str4;

        str1 <<"Matches: " << stats.matches;
        str2 << "Inliers: " << stats.inliers;
        str3 <<"Inlier ratio: " << std::setprecision(2) << stats.ratio;
        str4 << "Frame ps: " << fps;

        putText(image, str1.str(), cv::Point(0, image.rows - 90), front, 2, cv::Scalar::all(255), 3);
        putText(image, str2.str(), cv::Point(0, image.rows - 60), front, 2, cv::Scalar::all(255), 3);
        putText(image, str3.str(), cv::Point(0, image.rows - 30), front, 2, cv::Scalar::all(255), 3);
        putText(image, str4.str(), cv::Point(0, image.rows - 10), front, 2, cv::Scalar::all(255), 3);

    }

    void printStatistics(std::string name, Stats stats)
    {
        std::cout << name << std::endl;
        std::cout<< "-----------------"<<std::endl;

        std::cout << "Matches " << stats.matches << std::endl;
        std::cout << "Inliers " << stats.inliers << std::endl;
        std::cout << "Inlier ratio " << std::setprecision(2) << stats.ratio << std::endl;
        std::cout << "Keypoints " << stats.keypoints << std::endl;
        std::cout << std::endl;
    }

    std::vector<cv::Point2f> Points(std::vector<cv::KeyPoint> keypoints)
    {
        std::vector<cv::Point2f> res;
        for(unsigned i = 0; i < keypoints.size(); i++)
        {
            res.push_back(keypoints[i].pt);
        }
        return res;
    }

    cv::Rect2d seclectROI(const cv::String & video_name, const cv::Mat &frame)
    {
        struct Data
        {
            cv::Point center;
            cv::Rect2d box;

            static void mouseHandler(int event, int x, int y, int flags, void *param)
            {
                Data *data = (Data*)param;
                switch(event)
                {
                // start to select the bounding box
                case cv::EVENT_LBUTTONDOWN:
                    data->box = cvRect(x, y, 0, 0);
                    data->center = cv::Point2f((float)x, (float)y);
                    break;
                case cv::EVENT_MOUSEMOVE:
                    if(flags == 1)
                    {
                        data->box.width = 2*(x - data->center.x);
                        data->box.height = 2*(y - data->center.y);
                        data->box.x = data->center.x - data->box.width/2.0;
                        data->box.y = data->center.y - data->box.height/2.0;
                    }
                    break;
                case cv::EVENT_LBUTTONUP:
                    if(data->box.width < 0)
                    {
                        data->box.x += data->box.width;
                        data->box.width *= -1;
                    }
                    if(data->box.height < 0)
                    {
                        data->box.y += data->box.height;
                        data->box.height *= -1;
                    }
                    break;
                }
            }

        } data;

        setMouseCallback(video_name, Data::mouseHandler, &data);
        while(cv::waitKey(1) < 0)
        {
            cv::Mat draw = frame.clone();
            rectangle(draw, data.box, cv::Scalar(255, 0, 0), 2, 1);
            imshow(video_name, draw);
        }
        return data.box;
    }

//    void load_w2c_DD11(std::vector<std::vector<double> > &w2c)
//    {
//        std::ifstream ifstr("DD11_w2c_fast.txt");
//        std::vector<double> tmp(4, 0);
//        for (int i = 0; i < 32786; i++)
//        {
//            w2c.push_back(tmp);
//        }
//        double tmp_val;
//        for (size_t i = 0; i < 32768; i++)
//        {
//            for (size_t j = 0; j < 4; j++)
//            {
//                ifstr >> tmp_val;
//                w2c[i][j] = tmp_val;
//            }
//        }

//        ifstr.close();
//    }

}

}
