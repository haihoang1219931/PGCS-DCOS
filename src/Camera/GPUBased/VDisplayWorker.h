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
class GimbalInterface;
class VDisplayWorker : public QObject
{
    Q_OBJECT    
    public:
        VDisplayWorker(QObject *_parent = 0);
        ~VDisplayWorker();
        void setListObjClassID(std::vector<int> _listObjClassID);
        void setVideoSavingState(bool _state);
        void capture(bool writeTime = true, bool writeLocation = true);
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
        std::vector<std::string> readLabelNamesToMap(std::string const filename);

        index_type readBarcode(const cv::Mat &_rgbImg);

        bool checkIDExisted(int _idx);

        cv::Point convertPoint(cv::Point originPoint, cv::Mat stabMatrix);
    public:
        GimbalInterface* m_gimbal = nullptr;
        index_type m_currID;
        RollBuffer<ProcessImageCacheItem> *m_matImageBuff;
        RollBuffer<GstFrameCacheItem> *m_gstRTSPBuff;
        RollBuffer<GstFrameCacheItem> *m_gstEOSavingBuff;
        RollBuffer<GstFrameCacheItem> *m_gstIRSavingBuff;
        RollBuffer<DetectedObjectsCacheItem> *m_rbDetectedObjs;
        RollBuffer<DetectedObjectsCacheItem> *m_rbSearchObjs;
//        RollBuffer<DetectedObjectsCacheItem> *m_rbMOTObjs;
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
        bool m_writeTime = true;
        bool m_writeLocation = true;
        int m_countUpdateOD = 0;
        std::vector<bbox_t> m_listObj;

};
#endif // VDISPLAYWORKER_H
