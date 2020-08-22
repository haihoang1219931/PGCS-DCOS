#ifndef GSTREAMER_ELEMENT_H
#define GSTREAMER_ELEMENT_H

#include<iostream>
#include<gst/gst.h>
#include <gst/app/gstappsink.h>
#include<stdio.h>
//#include<unistd.h>
typedef struct _Data {
    GMainLoop *loop;
    GstBus *bus;
    GstCaps *caps;
    GstPad *pad;
    GstPipeline *pipeline;
    GstElement *udpsrc;
    GstElement *text;
    GstElement *demuxer;
    GstElement *vqueue;
    GstElement *parser;
    GstElement *decoder;
    GstElement *conv;
    GstElement *vsink;
} gstData;

#endif // GSTREAMER_ELEMENT_H
