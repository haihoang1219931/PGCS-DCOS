#ifndef VDISPLAYWORKER_H
#define VDISPLAYWORKER_H

#include <QObject>
#include <QVideoFrame>
#include <QMutex>
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <QMap>
#include "Utils/Zbar/ZbarLibs.h"
#include "Utils/Files/FileControler.h"
#include "Utils/Files/PlateLog.h"
#include "Payload/Cache/Cache.h"
#include "Payload/Cache/ProcessImageCacheItem.h"
#include "Payload/Buffer/RollBuffer.h"
#include "Payload/Packet/Common_type.h"
#include "Payload/Packet/XPoint.h"
#include "Payload/Cache/FixedMemory.h"
#include "Payload/Cache/Cache.h"
#include "Payload/Cache/TrackObject.h"
#include "Cuda/ipcuda_image.h"

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
        RollBuffer<ProcessImageCacheItem> *m_matImageBuff;
        RollBuffer<GstFrameCacheItem> *m_gstRTSPBuff;
        RollBuffer<GstFrameCacheItem> *m_gstEOSavingBuff;
        RollBuffer<GstFrameCacheItem> *m_gstIRSavingBuff;
        RollBuffer<DetectedObjectsCacheItem> *m_rbSearchObjs;
        RollBuffer<DetectedObjectsCacheItem> *m_rbMOTObjs;
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
