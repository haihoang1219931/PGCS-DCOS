#ifndef CVRTSPSERVER_H
#define CVRTSPSERVER_H

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/cv.h>

#include "gstreamer_element.h"
#include <deque>
#include <thread>
#include <chrono>
#include <utility>
#include <map>
#include <gst/app/gstappsrc.h>
#include <gst/video/gstvideometa.h>
#include <gst/video/video.h>
#include <gst/rtsp-server/rtsp-server.h>

#include <QTimer>
#include <QObject>
#include <QDebug>
#include <QMutex>
#include <QWaitCondition>
#include <QVideoFrame>
class CVRTSPServer : public QObject
{
    Q_OBJECT
public:
    explicit CVRTSPServer(QObject *parent = nullptr);
    virtual ~CVRTSPServer();
public:
    std::deque<std::pair<int,GstSample*>>* m_imageQueue;
public:
    QMutex *m_mutexCapture;
    QWaitCondition *m_pauseCond;
public:
    int m_width = 1920;
    int m_height = 1080;
    int m_fps = 30;
    int m_frameID = -1;
    bool white = false;
    bool m_stop = false;
    QTimer* m_timer;
    bool isNeedData;
    GstElement* element;
    GstElement* appsrc;
    GError* err = NULL;
    GMainLoop *loop  = NULL;
    GstPipeline *pipeline = NULL;
    GstElement* mPipeline = NULL;
    GstClockTime timestamp = 0;
    GstRTSPMedia * media;
    GstRTSPServer *server;
    GstRTSPMountPoints *mounts;
    GstRTSPMediaFactory *factory;
public:
    void enough_data (GstElement * appsrc, guint unused,
                    gpointer user_data);
    static void wrap_enough_data (GstElement * appsrc, guint unused,
                                gpointer user_data);
    GstFlowReturn need_data (GstElement * appsrc, guint unused,
                    gpointer user_data);
    static GstFlowReturn wrap_need_data (GstElement * appsrc, guint unused,
                                gpointer user_data);
    gboolean seek_data (GstElement* object,
                                             guint64 arg0,
                                             gpointer user_data);
    static gboolean wrap_seek_data (GstElement* object,
                                             guint64 arg0,
                                             gpointer user_data);
    void media_configure (GstRTSPMediaFactory * factory, GstRTSPMedia * media,
                          gpointer user_data);
    static void wrap_media_configure (GstRTSPMediaFactory * factory, GstRTSPMedia * media,
                                      gpointer user_data);
public:
    gboolean gstreamer_pipeline_operate();
    void setStateRun(bool running);

    void pause(bool pause);
Q_SIGNALS:
    void stopped();

public Q_SLOTS:
    void doWork();
    void changeResolution(int width,int height, int fps);
};

#endif // CVRTSPSERVER_H
