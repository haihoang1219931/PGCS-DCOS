#ifndef VRTSPSERVER_H
#define VRTSPSERVER_H

#include <QObject>
#include <QThread>
#include <QMutex>
#include <chrono>
#include <gst/gst.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string.h>
#include <gst/app/gstappsrc.h>
#include <gst/video/gstvideometa.h>
#include <gst/video/video.h>
#include <map>
#include <gst/rtsp-server/rtsp-server.h>
#include "../Cache/Cache.h"

using namespace rva;

class VRTSPServer : public QThread
{
        Q_OBJECT
    public:
        explicit VRTSPServer();
        ~VRTSPServer();
        void run();

    public:
        void enough_data(GstElement *appsrc, guint unused, gpointer user_data);
        static void wrap_enough_data(GstElement *appsrc, guint unused, gpointer user_data);
        GstFlowReturn need_data(GstElement *appsrc, guint unused, gpointer user_data);
        static GstFlowReturn wrap_need_data(GstElement *appsrc, guint unused, gpointer user_data);
        gboolean seek_data(GstElement *object, guint64 arg0, gpointer user_data);
        static gboolean wrap_seek_data(GstElement *object, guint64 arg0,  gpointer user_data);
        void media_configure(GstRTSPMediaFactory *factory, GstRTSPMedia *media, gpointer user_data);
        static void wrap_media_configure(GstRTSPMediaFactory *factory, GstRTSPMedia *media, gpointer user_data);

    public:
        gboolean gstreamer_pipeline_operate();
        void setStateRun(bool running);
        void pause(bool pause);
        void setStreamMount(std::string _streamMount);
        void setStreamSize(int _width, int _height);
    public:
        std::string m_streamMount = "/stream";
        int m_width = 1280;
        int m_height = 720;
        int m_fps = 30;
        int m_currID = -1;
        bool m_stop = false;
        GstElement *element;
        GstElement *appsrc;
        GError *err = NULL;
        GMainLoop *loop  = NULL;
        GstPipeline *pipeline = NULL;
        GstElement *mPipeline = NULL;
        GstClockTime timestamp = 0;
        GstRTSPMedia *media;
        GstRTSPServer *server;
        GstRTSPMountPoints *mounts;
        GstRTSPMediaFactory *factory;
        RollBuffer_<GstFrameCacheItem> *m_gstRTSPBuff;
};

#endif // VRTSPSERVER_H
