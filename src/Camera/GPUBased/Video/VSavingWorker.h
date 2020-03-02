#ifndef VSAVINGWORKER_H
#define VSAVINGWORKER_H

#include "../Camera/Cache/Cache.h"
#include "../Zbar/ZbarLibs.h"
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

using namespace rva;

class VSavingWorker : public QThread
{
    Q_OBJECT
public:
    //  explicit VFrameGrabber(QObject *_parent = 0);
    VSavingWorker(std::string _mode);
    ~VSavingWorker();

    static bool wrapper_run(void *pointer);

    bool initPipeline();

    void run();

    static void wrapperOnNeedData(GstAppSrc* _appSrc, guint _size, gpointer _uData);

    static void wrapperOnEnoughData(GstAppSrc* _appSrc, gpointer _uData);

    static gboolean wrapperOnSeekData(GstAppSrc* _appSrc, guint64 _offset, gpointer _uData);

private:
    void onNeedData(GstAppSrc* _appSrc, guint _size, gpointer _uData);

    void onEnoughData(GstAppSrc* _appSrc, gpointer _uData);

    gboolean onSeekData(GstAppSrc* _appSrc, guint64 _offset, gpointer _uData);

    std::string getFileNameByTime();

    void correctTimeLessThanTen(std::string& inputStr, int time);

    bool createFolder(std::string _folderName);

    bool checkIfFolderExist(std::string _folderName);

private:
    Status::SensorMode m_sensorMode;                        /**< Mode of image sensing (EO/IR) */
    RollBuffer_<GstFrameCacheItem> *m_buffVideoSaving;
    GMainLoop *m_loop;
    GstPipeline *m_pipeline;
    std::string m_pipeline_str;
    GstBus *m_bus;
    GError* m_err = NULL;
    guint m_bus_watch_id;
    GstAppSrc *m_appSrc;
    index_type m_currID = 0;
    int m_width;
    int m_height;
    int m_bitrate;
    int m_frameRate;
    std::string m_filename;
    uint m_countFrame = 0;
};

#endif // VSAVINGWORKER_H
