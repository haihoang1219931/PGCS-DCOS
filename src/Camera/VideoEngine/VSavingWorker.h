#ifndef VSAVINGWORKER_H
#define VSAVINGWORKER_H


#include <QObject>
#include <QThread>
#include <chrono>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include "../Cache/Cache.h"
using namespace rva;

class VSavingWorker : public QThread
{
        Q_OBJECT
    public:
        //  explicit VFrameGrabber(QObject *_parent = 0);
        VSavingWorker();
        ~VSavingWorker();

        static bool wrapper_run(void *pointer);

        bool initPipeline();

        void run();

        static void wrapperOnNeedData(GstAppSrc *_appSrc, guint _size, gpointer _uData);

        static void wrapperOnEnoughData(GstAppSrc *_appSrc, gpointer _uData);

        static gboolean wrapperOnSeekData(GstAppSrc *_appSrc, guint64 _offset, gpointer _uData);

        void setStreamSize(int _width, int _height);

        void setSensorMode(int _mode);

        void stopPipeline();
    private:
        void onNeedData(GstAppSrc *_appSrc, guint _size, gpointer _uData);

        void onEnoughData(GstAppSrc *_appSrc, gpointer _uData);

        gboolean onSeekData(GstAppSrc *_appSrc, guint64 _offset, gpointer _uData);

        std::string getFileNameByTime();

        void correctTimeLessThanTen(std::string &inputStr, int time);

        bool createFolder(std::string _folderName);

        bool checkIfFolderExist(std::string _folderName);

    public:
        int m_sensorMode;
        RollBuffer_<GstFrameCacheItem> *m_buffVideoSaving;
        GMainLoop *m_loop = nullptr;
        GstPipeline *m_pipeline = nullptr;
        std::string m_pipeline_str ;
        GstBus *m_bus = nullptr;
        GError *m_err  = nullptr;
        guint m_bus_watch_id;
        GstAppSrc *m_appSrc = nullptr;
        int m_currID = -1;
        int m_width = 1080;
        int m_height = 720;
        int m_fps = 25;
        int m_bitrate;
        int m_frameRate;
        std::string m_filename;
        uint m_countFrame = 0;
};

#endif // VSAVINGWORKER_H
