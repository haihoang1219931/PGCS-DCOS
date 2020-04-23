#ifndef VSEARCHWORKER_H
#define VSEARCHWORKER_H

#include "../Cache/Cache.h"
#include "../Zbar/ZbarLibs.h"
#include <QObject>
#include <QThread>
#include <chrono>
#include <gst/gst.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string.h>
#include "plateOCR/PlateOCR.h"
#include "Clicktrack/recognition.h"

using namespace rva;

class VSearchWorker : public QThread
{
        Q_OBJECT
    public:
		explicit VSearchWorker();
		~VSearchWorker();
        void run();

    public:
        bool isRunning();
        void stop();
		void enableSearch();
		void disableSearch();
        void setOCR(OCR *_OCR);
		void setPlateDetector(Detector *_plateDetector);

    private:
        index_type m_currID = 0;
        RollBuffer_<ProcessImageCacheItem> *m_matImageBuff;
//        RollBuffer_<DetectedObjectsCacheItem> *m_rbDetectedObjs;
        RollBuffer_<DetectedObjectsCacheItem> *m_rbMOTObjs;
        RollBuffer_<DetectedObjectsCacheItem> *m_rbSearchObjs;
        RollBuffer<Eye::MotionImage> *m_rbIPCEO;
        RollBuffer<Eye::MotionImage> *m_rbIPCIR;
        bool m_running = true;
        std::mutex m_mtx;
		std::condition_variable m_cvEnMOT;
        bool m_enOD = false;

		PlateOCR* m_plateOCR;
};

#endif // VSEARCHWORKER_H
