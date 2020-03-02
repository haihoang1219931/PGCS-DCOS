#ifndef VTRACKWORKER_H
#define VTRACKWORKER_H

#include "../Camera/Cache/Cache.h"
#include "../Camera/ControllerLib/Command/IPCCommands.h"
#include "../Camera/ControllerLib/Packet/XPoint.h"
#include "../Zbar/ZbarLibs.h"
#include <QObject>
#include <QThread>
#include <chrono>
#include <gst/gst.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string.h>

#include "Multitracker/yolo_v2_class.hpp"
#include "Multitracker/multitrack.h"
#include "Multitracker/PlateOCR.h"
#include "OCR/preprocessing.h"
#include "OCR/recognition.h"

using namespace rva;

class VTrackWorker : public QThread
{
        Q_OBJECT
    public:
        explicit VTrackWorker();
        ~VTrackWorker();

        void run();

    public:
        void stop();
        void hasNewTrack(int _id, double _px, double _py, double _w, double _h);
        void hasNewMode();
        void setPlateOCR(PlateOCR *_plateOCR);
        bool isRunning();

    private:
        void init();
        int findNearestObject(const std::vector<bbox_t> &result_vec,
                              const cv::Point &point, double &minDistance);

    Q_SIGNALS:
        void determinedTrackObjected(int _id, double _px, double _py, double _w, double _h, double _oW, double _oH);
        void determinedPlateOnTracking(QString _imgPath, QString _plateID);


    private:
        bool m_running = true;
        index_type m_currID;
        RollBuffer_<ProcessImageCacheItem> *m_matImageBuff;
        RollBuffer_<DetectedObjectsCacheItem> *m_rbDetectedObjs;
        RollBuffer<Eye::TrackResponse> *m_rbTrackResIR;
        RollBuffer<Eye::TrackResponse> *m_rbTrackResEO;
        RollBuffer<Eye::SystemStatus> *m_rbSystem;
        RollBuffer<Eye::MotionImage> *m_rbIPCEO;
        RollBuffer<Eye::MotionImage> *m_rbIPCIR;
        Detector *m_detector;
        const float m_threshold = 0.3f;

        XPoint m_trackPoint;
        bool m_hasNewTrack = false;
        bool m_hasNewMode = false;
        bool m_trackEn = false;
        std::mutex m_mtx;
        std::condition_variable m_cvHasNewTrack;

        MultilevelDetector *m_multiDetector;
        PlateOCR *m_plateOCR;
        // For giapvn
        OCR m_recognizor;
        QMap<QString, QString> m_plates;
};

#endif // VTRACKWORKER_H
