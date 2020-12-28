#ifndef VODWORKER_H
#define VODWORKER_H

#include "../Cache/Cache.h"
#include "../Zbar/ZbarLibs.h"
#include <QObject>
#include <QThread>
#include <chrono>
#include <gst/gst.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string.h>

// OD
#include "OD/yolo_v2_class.hpp"
#define Detector YoloDetector

using namespace rva;
class PlateDetector;
class VODWorker : public QThread
{
        Q_OBJECT
    public:
        explicit VODWorker();
        ~VODWorker();
        void run();

    public:
        bool isActive();
        void stop();
        void enableOD();
        void disableOD();

        void setObjectClassifier(Detector *_detector);
        void setPlateIdentifier(PlateDetector* plateDetector);

    private:
        index_type m_currID;
        RollBuffer<ProcessImageCacheItem> *m_matImageBuff;
        RollBuffer<DetectedObjectsCacheItem> *m_rbDetectedObjs;
//        RollBuffer<Eye::MotionImage> *m_rbIPCEO;
//        RollBuffer<Eye::MotionImage> *m_rbIPCIR;
        bool m_running = true;
        std::mutex m_mtx;
        std::condition_variable m_cvEnOD;
        bool m_enOD = false;

        Detector* m_objectClassifier;
        PlateDetector* m_plateDetector;
};

#endif // VODWORKER_H
