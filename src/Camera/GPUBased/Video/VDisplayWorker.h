#ifndef VDISPLAYWORKER_H
#define VDISPLAYWORKER_H

#include "../Camera/Cache/Cache.h"
#include "../Camera/Cache/FixedPinnedMemory.h"
#include "../Camera/Cache/ProcessImageCacheItem.h"
#include "../Camera/ControllerLib/Buffer/RollBuffer.h"
#include "../Camera/ControllerLib/Packet/Common_type.h"
#include "../Video/Multitracker/yolo_v2_class.hpp"
#include "../Zbar/ZbarLibs.h"
#include "Cuda/ipcuda_image.h"
#include <QObject>
#include <QVideoFrame>
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>

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

    public Q_SLOTS:
        void process();

    Q_SIGNALS:
        void receivedFrame(int _id, QVideoFrame _frame);
        void receivedFrame();
        void finished();
        void error(QString _err);

    private:
        void init();
        void drawCenter(cv::Mat &_img, int _r, int _centerX, const int _centerY,
                        cv::Scalar _color);
        void drawSteeringCenter(cv::Mat &_img, int _wBoundary, int _centerX,
                                const int _centerY, cv::Scalar _color);
        void drawObjectBoundary(cv::Mat &_img, cv::Rect _objBoundary,
                                cv::Scalar _color);
        void drawDetectedObjects(cv::Mat &_img, const std::vector<bbox_t> &m_listObj);

        std::vector<std::string> objects_names_from_file(std::string const filename);

        void draw_boxes_center(cv::Mat mat_img, std::vector<bbox_t> result_vec,
                               std::vector<std::string> obj_names, float fps = -1);

        index_type readBarcode(const cv::Mat &_rgbImg);

        bool checkIDExisted(int _idx);

    public:
        index_type m_currID;

        RollBuffer_<ProcessImageCacheItem> *m_matImageBuff;
        RollBuffer_<GstFrameCacheItem>* m_gstRTSPBuff;
        RollBuffer_<GstFrameCacheItem>* m_gstEOSavingBuff;
        RollBuffer_<GstFrameCacheItem>* m_gstIRSavingBuff;
        RollBuffer<Eye::TrackResponse> *m_rbTrackResEO;
        RollBuffer<Eye::XPoint> *m_rbXPointEO;
        RollBuffer<Eye::TrackResponse> *m_rbTrackResIR;
        RollBuffer<Eye::XPoint> *m_rbXPointIR;
        RollBuffer_<DetectedObjectsCacheItem> *m_rbDetectedObjs;
        RollBuffer<Eye::SystemStatus> *m_rbSystem;
        RollBuffer<Eye::MotionImage> *m_rbIPCEO;
        RollBuffer<Eye::MotionImage> *m_rbIPCIR;
        std::vector<std::string> m_objName;
        cv::Mat m_imgShow;
        std::string m_ipStream;
        int m_portStream;
        std::mutex m_mtxShow;
        Eye::XPoint m_prevSteeringPoint;
        std::vector<int> m_listObjClassID;
        bool m_enSaving = true;
};
#endif // VDISPLAYWORKER_H
