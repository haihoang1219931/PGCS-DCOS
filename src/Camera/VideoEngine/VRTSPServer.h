#ifndef VRTSPSERVER_H
#define VRTSPSERVER_H

#include <QObject>
#include <QThread>
#include <QMutex>
#include <chrono>
#include <gst/gst.h>
#include <csignal>
#include <iostream>
#include <thread>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <string.h>
#include <gst/app/gstappsrc.h>
#include <gst/video/gstvideometa.h>
#include <gst/video/video.h>
#include <map>
#include <gst/rtsp-server/rtsp-server.h>
#include "../Cache/Cache.h"
using namespace Eye;
using namespace rva;
class VRTSPServer : public QThread
{
    Q_OBJECT
public:
    explicit VRTSPServer();
    ~VRTSPServer();


public:
    GstRTSPFilterResult clientFilter(GstRTSPServer* server, GstRTSPClient* client, gpointer user);
    static GstRTSPFilterResult wrap_clientFilter(GstRTSPServer* server, GstRTSPClient* client, gpointer user);
    void clientClosed(GstRTSPClient* client, gpointer user);
    static void wrap_clientClosed(GstRTSPClient* client, gpointer user);
    void clientConnected(GstRTSPServer* server, GstRTSPClient* client, gpointer user);
    static void wrap_clientConnected(GstRTSPServer* server, GstRTSPClient* client, gpointer user);
    void closeClients(GstRTSPServer* server,gpointer user);
    static void wrap_closeClients(GstRTSPServer* server,gpointer user);
    void enough_data(GstElement *appsrc, guint unused, gpointer user_data);
    static void wrap_enough_data(GstElement *appsrc, guint unused, gpointer user_data);
    GstFlowReturn need_data(GstElement *appsrc, guint unused, gpointer user_data);
    static GstFlowReturn wrap_need_data(GstElement *appsrc, guint unused, gpointer user_data);
    gboolean seek_data(GstElement *object, guint64 arg0, gpointer user_data);
    static gboolean wrap_seek_data(GstElement *object, guint64 arg0,  gpointer user_data);
    void media_configure(GstRTSPMediaFactory *factory, GstRTSPMedia *media, gpointer user_data);
    static void wrap_media_configure(GstRTSPMediaFactory *factory, GstRTSPMedia *media, gpointer user_data);
    bool gstreamer_pipeline_operate();
public Q_SLOTS:
    void run();
    void setStateRun(bool running);
    void pause(bool pause);
    void setStreamPort(int port);
    void setStreamMount(QString _streamMount);
    void setStreamSize(int _width, int _height);
public:
    std::string m_streamMount = "/stream";
    int m_port = 8554;
    int m_width = 1920;
    int m_height = 1080;
    int m_fps = 30;
    bool m_stop = false;
    QString m_source;
    GMainContext * context = nullptr;
    GstElement *element;
    GstElement *appsrc;
    GError *err = nullptr;
    GMainLoop *loop  = nullptr;
    GstPipeline *pipeline = nullptr;
    GstElement *mPipeline = nullptr;
    GstClockTime timestamp = 0;
    GstRTSPMedia *media = nullptr;
    GstRTSPServer *server = nullptr;
    int m_currID = -1;
    GstRTSPMountPoints *mounts = nullptr;
    GstRTSPMediaFactory *factory = nullptr;
    guint m_rtspAttachID;
    uint32_t m_clientCount=0;
    RollBuffer<GstFrameCacheItem> *m_gstRTSPBuff;
};

#endif // VRTSPSERVER_H
