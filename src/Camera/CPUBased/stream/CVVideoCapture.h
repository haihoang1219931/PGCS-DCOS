#ifndef CVVIDEOCAPTURE_H
#define CVVIDEOCAPTURE_H

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <gst/gst.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <QGuiApplication>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/cv.h>
#include <QObject>
#include <QDebug>
#include <QMutex>
#include <QVideoFrame>
#include "gstreamer_element.h"
#include "../utils/filenameutils.h"
#include <deque>
#include <thread>
#include <chrono>
#include <utility>
#include <map>
#ifdef __linux__
//linux code goes here
#include <unistd.h>
#elif _WIN32
// windows code goes here
#else

#endif
#define DEBUG
#define LICENSE_DIR     "."
#define TIMEOUT_MS      1000
//#define SLEEP_TIME      25
#define QUEUE_SIZE      3
#define MAX_FRAME_ID    50000
class GimbalInterface;
class CVVideoCapture : public QObject
{
        Q_OBJECT
    public:
        explicit CVVideoCapture(QObject *parent = 0);
        virtual ~CVVideoCapture();

    Q_SIGNALS:
        void stopped();

    public Q_SLOTS:
        void doWork();
    public:
        void create_pipeline();
        GstPadProbeReturn pad_data_mod(GstPad *pad, GstPadProbeInfo *info, gpointer user_data);
        static GstPadProbeReturn wrap_pad_data_mod(GstPad *pad, GstPadProbeInfo *info, gpointer user_data);
        GstFlowReturn read_frame_buffer(GstAppSink *vsink, gpointer user_data);
        static GstFlowReturn wrap_read_frame_buffer(GstAppSink *vsink, gpointer user_data);
        static gboolean wrapNeedKlv(void* userPointer);

        static void wrapStartFeedKlv(GstElement * pipeline, guint size, void* userPointer);

        gboolean needKlv(void* userPointer);
        gboolean gstreamer_pipeline_operate();
        void setStateRun(bool running);
        void msleep(int ms);
        void stop();
        void setSource(std::string source);
        void pause(bool pause);
        gint64 getTotalTime();
        gint64 getPosCurrent();
        void setSpeed(float speed);
        void goToPosition(float percent);
private:
        std::string getFileNameByTime();
        void correctTimeLessThanTen(std::string &_inputStr, int _time);
    public:
        float m_speed = 1;
        int m_frameID = 0;
        std::deque<std::pair<int, GstSample *>> *m_imageQueue;
        QMutex *m_mutexCapture;
        std::string m_ip;
        int m_port;
        std::string m_source;
        std::string m_logFolder;
        std::string m_logFile;
        bool m_stop = false;
        //    shared_ptr<EyePhoenix::VideoSaver> m_recorder;
        GError *err = NULL;
        GMainLoop *loop  = NULL;
        GstPipeline *pipeline = NULL;
        GstElement *vsink = NULL;
        GstElement *m_pipeline = NULL;
        GstClockTime m_time_bef = 0;
        GstClockTime m_time = 0 ;
        int m_frameCaptureID = 0;
        gint64 m_totalTime = 1800000000000;
        // metadata
        GstAppSrc* m_klvAppSrc = nullptr;
        int m_metaID = 0;
        GimbalInterface* m_gimbal = nullptr;
};

#endif // CVVIDEOCAPTURE_H
