/**
 *=============================================================================
 * Project: Eyephoenix CSI Camera Writing
 * Project Short Description: Save video to file with input
 * Module: VIDEO_UTILITY implementation
 * Author: Trung Nguyen
 * Date: 08/06/2018
 * Viettel Aerospace Institude - Viettel Group
 * ============================================================================
*/

#ifndef GSTUTILITY_HPP
#define GSTUTILITY_HPP


//============== Inluding Gstreamer-1.0 Libraries ======//
#include <glib.h>
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>

enum class gstEncodeType {
    GST_CODEC_H264 = 0,
    GST_CODEC_H265,
    GST_CODEC_MPEG
};

/**
 * LOG_GSTREAMER printf prefix
 */
#define LOG_GSTREAMER "[gstreamer] "

/**
 * gstreamerInit
 */
extern bool gstreamerInit();

/**
 * gst_message_print
 */
gboolean gst_message_print(_GstBus* bus, _GstMessage* message, void* user_data);

#endif // GSTUTILITY_HPP
