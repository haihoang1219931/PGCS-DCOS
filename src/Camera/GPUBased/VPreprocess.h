#ifndef VPREPROCESS_H
#define VPREPROCESS_H

#include "../Cache/Cache.h"
#include "../../Zbar/ZbarLibs.h"
#include "../Cache/FixedMemory.h"
#include <QObject>
#include <QThread>
#include <chrono>
#include <gst/gst.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string.h>
using namespace rva;

class VPreprocess : public QThread
{
        Q_OBJECT
    public:
        VPreprocess();
        ~VPreprocess();

        void run();
        void stop();
    private:
        index_type readBarcode(const cv::Mat &_rgbImg);
        cv::Size getImageSize(int _dataSize);

    private:
        index_type m_currID;
        RollBuffer_<GstFrameCacheItem> *m_gstFrameBuff;
        RollBuffer_<ProcessImageCacheItem> *m_matImageBuff;
        bool m_enStab = true;
        bool m_running = true;

};

#endif // VPREPROCESS_H
