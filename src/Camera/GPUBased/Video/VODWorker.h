#ifndef VODWORKER_H
#define VODWORKER_H

#include "../Camera/Cache/Cache.h"
#include "../Zbar/ZbarLibs.h"
#include <QObject>
#include <QThread>
#include <chrono>
#include <gst/gst.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string.h>
#include "Multitracker/multitrack.h"
#include "Multitracker/PlateOCR.h"

using namespace rva;

class VODWorker : public QThread
{
        Q_OBJECT
    public:
        explicit VODWorker();
        ~VODWorker();
        void run();

    public:
        bool isRunning();
        void stop();
        void enableOD();
        void disableOD();
        void setMultiTracker(MultiTrack *_mulTracker);
		void setPlateOCR(PlateOCR *_plateOCR);

    private:
        index_type m_currID;
        RollBuffer_<ProcessImageCacheItem> *m_matImageBuff;
        RollBuffer_<DetectedObjectsCacheItem> *m_rbDetectedObjs;
        RollBuffer<Eye::MotionImage> *m_rbIPCEO;
        RollBuffer<Eye::MotionImage> *m_rbIPCIR;
        bool m_running = true;
        std::mutex m_mtx;
        std::condition_variable m_cvEnOD;
        bool m_enOD = false;
        MultiTrack *m_mulTracker;
		PlateOCR *m_plateOCR;
		int dropDetection;
};

#endif // VODWORKER_H
