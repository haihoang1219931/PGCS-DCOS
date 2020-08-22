#ifndef ZBARLIBS_H
#define ZBARLIBS_H

#include <iostream>
#include <string.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <zbar.h>
#include <zbar/ImageScanner.h>
typedef struct {
    std::string type;
    std::string data;
    std::vector<cv::Point> location;
} decodedObject;
class ZbarLibs
{
public:
    ZbarLibs();
    static void decode(cv::Mat &im, std::vector<decodedObject> &decodedObjects);
    static void display(cv::Mat &im, std::vector<decodedObject> &decodedObjects);
};

#endif // ZBARLIBS_H
