#ifndef CVRecord_H
#define CVRecord_H

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <gst/gst.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <gst/gst.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/cv.h>
#include <QObject>
#include <QDebug>
#include <QMutex>
#include <QVideoFrame>
#include "gstreamer_element.h"
#include <deque>
#include <thread>
#include <chrono>
#include <utility>
#include <map>
#include "../utils/filenameutils.h"
#include "../recorder/gstsaver.hpp"
class  CVRecord : public QObject
{
    Q_OBJECT
public:
    explicit CVRecord(QObject *parent = 0);
    virtual ~CVRecord();
public:
    std::deque<std::pair<int,GstSample*>>* m_imageQueue;
    cv::Mat* m_imgShow;
    QMutex* m_mutexCapture;
    QMutex* m_mutexProcess;
    cv::Mat m_img;
    std::string m_logFolder;
    std::string m_logFile;
//    cv::Mat m_imgRaw;
    int *m_frameID;
    const int SLEEP_TIME = 1;
    bool m_stop = false;
    shared_ptr<EyePhoenix::VideoSaver> m_recorderOriginal;
    shared_ptr<EyePhoenix::VideoSaver> m_recorderProcessed;
Q_SIGNALS:
    void processDone();
    void stopped();
public Q_SLOTS:
    void capture();
    void doWork();
public:
    void msleep(int ms);
};

#endif // CVRecord_H
