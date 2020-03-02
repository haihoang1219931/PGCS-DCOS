#ifndef ZBARLIBS_H
#define ZBARLIBS_H

#include <iostream>
#include <string.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <zbar.h>
#include <zbar/ImageScanner.h>

using namespace std;
using namespace zbar;

typedef struct {
    string type;
    string data;
	vector<cv::Point> location;
} decodedObject;
class ZbarLibs
{
public:
    ZbarLibs();
	static void decode(cv::Mat &im, vector<decodedObject> &decodedObjects);
	static void display(cv::Mat &im, vector<decodedObject> &decodedObjects);
};

#endif // ZBARLIBS_H
