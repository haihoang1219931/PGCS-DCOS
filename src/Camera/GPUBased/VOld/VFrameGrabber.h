#ifndef VFrameGrabber_H
#define VFrameGrabber_H

#include "../Cache/Cache.h"
#include "../Zbar/ZbarLibs.h"
#include <QObject>
#include <QThread>
#include <chrono>
#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>

using namespace rva;

class VFrameGrabber : public QThread
{
        Q_OBJECT
    public:
        //  explicit VFrameGrabber(QObject *_parent = 0);
        VFrameGrabber();
        ~VFrameGrabber();

        static GstFlowReturn wrapperOnNewSample(GstAppSink *_vsink, gpointer _uData);

        static void wrapperOnEOS(_GstAppSink *_sink, void *_uData);

        static GstFlowReturn wrapperOnNewPreroll(_GstAppSink *_sink, void *_uData);

        static gboolean wrapperOnBusCall(GstBus *_bus, GstMessage *_msg,
                                         gpointer uData);

        static GstPadProbeReturn
        wrapperPadDataMod(GstPad *_pad, GstPadProbeInfo *_info, gpointer _uData);

        static void wrapperRun(void *_pointer);

        bool initPipeline();

        void stopPipeline();

        void run();

        bool stop();

        void restartPipeline();

        void setSource(std::string _ip, int _port);
    private:
        GstFlowReturn onNewSample(GstAppSink *vsink, gpointer user_data);

        void onEOS(_GstAppSink *sink, void *user_data);

        GstFlowReturn onNewPreroll(_GstAppSink *sink, void *user_data);

        gboolean onBusCall(GstBus *bus, GstMessage *msg, gpointer data);

        GstPadProbeReturn padDataMod(GstPad *_pad, GstPadProbeInfo *_info,
                                     gpointer _uData);

        gint64 getTotalTime();

        gint64 getPosCurrent();

        std::string getFileNameByTime();

        void correctTimeLessThanTen(std::string &inputStr, int time);

        bool createFolder(std::string _folderName);

        bool checkIfFolderExist(std::string _folderName);

    private:
        GMainLoop *m_loop = nullptr;
        GstPipeline *m_pipeline = nullptr;
        std::string m_pipelineStr;
        GstBus *m_bus = nullptr;
        GError *m_err = nullptr;
        guint m_busWatchID;
        GstAppSink *m_appSink = nullptr;
        std::string m_ip;
        uint16_t m_port;
        gint64 m_totalTime;
        bool m_stop = false;
        index_type m_currID;
        std::string m_filename;
        RollBuffer_<GstFrameCacheItem> *m_gstFrameBuff;
};

#endif // VFrameGrabber_H