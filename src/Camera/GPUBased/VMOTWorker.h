#ifndef VMOTWORKER_H
#define VMOTWORKER_H

#include "../Cache/Cache.h"
#include "../../Zbar/ZbarLibs.h"
#include <QObject>
#include <QThread>
#include <chrono>
#include <gst/gst.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string.h>
#include "Multitrack/multitrack.h"

using namespace rva;

class VMOTWorker : public QThread
{
        Q_OBJECT
    public:
		explicit VMOTWorker();
		~VMOTWorker();
        void run();

    public:
        bool isRunning();
        void stop();
		void enableMOT();
        void disableMOT();

    private:
        index_type m_currID = 0;
        RollBuffer_<ProcessImageCacheItem> *m_matImageBuff;
        RollBuffer_<DetectedObjectsCacheItem> *m_rbDetectedObjs;
        RollBuffer_<DetectedObjectsCacheItem> *m_rbMOTObjs;
        RollBuffer<Eye::MotionImage> *m_rbIPCEO;
        RollBuffer<Eye::MotionImage> *m_rbIPCIR;
        bool m_running = true;
        std::mutex m_mtx;
		std::condition_variable m_cvEnMOT;
        bool m_enOD = false;

        MultiTrack *m_mulTracker;
};

#endif // VMOTWORKER_H
