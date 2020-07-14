#ifndef VDISPLAYWORKER_H
#define VDISPLAYWORKER_H

#include "../Cache/Cache.h"
#include "../Cache/ProcessImageCacheItem.h"
#include "Camera/Buffer/RollBuffer.h"
#include "Camera/Packet/Common_type.h"
#include "Camera/Packet/XPoint.h"
#include "../Cache/FixedMemory.h"
#include "../../Zbar/ZbarLibs.h"
#include "Cuda/ipcuda_image.h"
#include "../../Files/FileControler.h"
#include "../../Files/PlateLog.h"
#include "../Cache/Cache.h"
#include "../Cache/TrackObject.h"
#include <QObject>
#include <QVideoFrame>
#include <QMutex>
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <QMap>

using namespace Eye;
using namespace rva;

class VDisplayWorker : public QObject
{
    Q_OBJECT    
    public:
        VDisplayWorker(QObject *_parent = 0);
        ~VDisplayWorker();
        void setListObjClassID(std::vector<int> _listObjClassID);
        void setVideoSavingState(bool _state);
        void capture();
    public Q_SLOTS:
        void process();

    Q_SIGNALS:
        void readyDrawOnRenderID(int viewerID, unsigned char *data, int width, int height,float* warpMatrix = nullptr,unsigned char* dataOut = nullptr);
        void finished();
        void error(QString _err);

    private:
        void init();
        void drawDetectedObjects(cv::Mat &imgY,cv::Mat &imgU,cv::Mat &imgV,
                                 const std::vector<bbox_t> &m_listObj);
        std::vector<std::string> objects_names_from_file(std::string const filename);

        index_type readBarcode(const cv::Mat &_rgbImg);

        bool checkIDExisted(int _idx);

        cv::Point convertPoint(cv::Point originPoint, cv::Mat stabMatrix);
    public:
        index_type m_currID;
        RollBuffer_<ProcessImageCacheItem> *m_matImageBuff;
        RollBuffer_<GstFrameCacheItem> *m_gstRTSPBuff;
        RollBuffer_<GstFrameCacheItem> *m_gstEOSavingBuff;
        RollBuffer_<GstFrameCacheItem> *m_gstIRSavingBuff;
        RollBuffer_<DetectedObjectsCacheItem> *m_rbSearchObjs;
        RollBuffer_<DetectedObjectsCacheItem> *m_rbMOTObjs;
//        RollBuffer<Eye::TrackResponse> *m_rbTrackResEO;
//        RollBuffer<Eye::XPoint> *m_rbXPointEO;
//        RollBuffer<Eye::TrackResponse> *m_rbTrackResIR;
//        RollBuffer<Eye::XPoint> *m_rbXPointIR;
//        RollBuffer<Eye::SystemStatus> *m_rbSystem;
//        RollBuffer<Eye::MotionImage> *m_rbIPCEO;
//        RollBuffer<Eye::MotionImage> *m_rbIPCIR;
        std::vector<std::string> m_objName;
        cv::Mat m_imgI420;
        cv::Mat m_imgI420Warped;
        std::vector<float> m_warpDataRender;
        cv::Mat m_imgGray;
        cv::Mat m_imgIRColor;
        std::string m_ipStream;
        int m_portStream;
        std::mutex m_mtxShow;
        Eye::XPoint m_prevSteeringPoint;
        std::vector<int> m_listObjClassID;
        bool m_enShare = true;
        bool m_enSaving = true;
        bool m_enOD = false;
        QMap<QString,QString> m_mapPlates;
        PlateLog* m_plateLog;
        QMutex m_captureMutex;
        bool m_captureSet = false;
        int m_countUpdateOD = 0;
        std::vector<bbox_t> m_listObj;
};
#endif // VDISPLAYWORKER_H
