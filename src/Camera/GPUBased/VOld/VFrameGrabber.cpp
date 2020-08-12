#include "VFrameGrabber.h"

// VFrameGrabber::VFrameGrabber(QObject *_parent) : QObject(_parent) {
//  gst_init(0, NULL);
//  m_loop = g_main_loop_new(NULL, FALSE);
//}

VFrameGrabber::VFrameGrabber()
{
    gst_init(0, NULL);
    m_loop = g_main_loop_new(NULL, FALSE);
    m_gstFrameBuff = Cache::instance()->getGstFrameCache();
    m_ip = "232.4.130.146";
    m_port = 18888;
}

VFrameGrabber::~VFrameGrabber()
{
    delete m_appSink;
    delete m_bus;
    delete m_loop;
    delete m_pipeline;
    delete  m_err;
}

void VFrameGrabber::run()
{
    GstStateChangeReturn result =
        gst_element_set_state(GST_ELEMENT(m_pipeline), GST_STATE_PLAYING);

    if (result != GST_STATE_CHANGE_SUCCESS) {
        printf("ReadingCam-gstreamer failed to set pipeline state to PLAYING "
               "(error %u)\n",
               result);
        return;
    }

    g_main_loop_run(m_loop);
    gst_element_set_state(GST_ELEMENT(m_pipeline), GST_STATE_NULL);
    gst_object_unref(m_pipeline);
    printf("\n === 6");
    return;
}

void VFrameGrabber::setSource(std::string _ip, int _port)
{
    m_ip = _ip;
    m_port = (uint16_t)_port;

    if (m_pipeline != nullptr) {
        gst_element_set_state(GST_ELEMENT(m_pipeline), GST_STATE_NULL);
    }

    this->initPipeline();
}

bool VFrameGrabber::stop()
{
    if (m_loop != NULL && g_main_loop_is_running(m_loop) == TRUE) {
        g_main_loop_quit(m_loop);
        m_stop = true;
        return true;
    }

    return false;
}

void VFrameGrabber::wrapperRun(void *_pointer)
{
    VFrameGrabber *itseft = (VFrameGrabber *)_pointer;
    return itseft->run();
}

GstFlowReturn VFrameGrabber::wrapperOnNewSample(GstAppSink *_vsink,
        gpointer _uData)
{
    VFrameGrabber *itseft = (VFrameGrabber *)_uData;
    return itseft->onNewSample(_vsink, _uData);
}

void VFrameGrabber::wrapperOnEOS(_GstAppSink *_sink, void *_uData)
{
    VFrameGrabber *itseft = (VFrameGrabber *)_uData;
    return itseft->onEOS(_sink, _uData);
}

gboolean VFrameGrabber::wrapperOnBusCall(GstBus *_bus, GstMessage *_msg,
        gpointer _uData)
{
    VFrameGrabber *itseft = (VFrameGrabber *)_uData;
    return itseft->onBusCall(_bus, _msg, _uData);
}

GstFlowReturn VFrameGrabber::wrapperOnNewPreroll(_GstAppSink *_sink,
        void *_uData)
{
    VFrameGrabber *itseft = (VFrameGrabber *)_uData;
    return itseft->onNewPreroll(_sink, _uData);
}

GstPadProbeReturn VFrameGrabber::wrapperPadDataMod(GstPad *_pad,
        GstPadProbeInfo *_info,
        gpointer _uData)
{
    VFrameGrabber *itseft = (VFrameGrabber *)_uData;
    return itseft->padDataMod(_pad, _info, _uData);
}

GstFlowReturn VFrameGrabber::onNewSample(GstAppSink *_vsink, gpointer _uData)
{
    GstSample *sample = gst_app_sink_pull_sample((GstAppSink *)_vsink);

    if (sample == NULL) {
        printf("\nError while pulling new sample");
        gst_sample_unref(sample);
        return GST_FLOW_ERROR;
    }

    m_currID++;
    GstFrameCacheItem gstFrame;
    gstFrame.setIndex(m_currID);
    gstFrame.setGstBuffer(gst_buffer_copy(gst_sample_get_buffer(sample)));
    m_gstFrameBuff->add(gstFrame);
    //    printf("\nReadFrame %d - %d", m_currID, gst_buffer_get_size(gst_sample_get_buffer(sample)));
    gst_sample_unref(sample);
    return GST_FLOW_OK;
}

void VFrameGrabber::onEOS(_GstAppSink *_sink, void *_uData)
{
    g_main_loop_quit(m_loop);
    printf("\ngstreamer decoder onEOS");
}

gboolean VFrameGrabber::onBusCall(GstBus *_bus, GstMessage *_msg,
                                  gpointer _uData)
{
    GMainLoop *loop = (GMainLoop *)_uData;

    switch (GST_MESSAGE_TYPE(_msg)) {
    case GST_MESSAGE_EOS: {
        g_print("\nEnd of stream");
        g_main_loop_quit(loop);
        break;
    }

    case GST_MESSAGE_ERROR: {
        gchar *debug;
        GError *error;
        gst_message_parse_error(_msg, &error, &debug);
        g_free(debug);
        g_printerr("\nError: %s", error->message);
        g_error_free(error);
        g_main_loop_quit(loop);
        break;
    }

    default: {
        break;
    }
    }

    return TRUE;
}

GstFlowReturn VFrameGrabber::onNewPreroll(_GstAppSink *_sink, void *_uData)
{
    printf("\ngstreamer decoder onPreroll");
    return GST_FLOW_OK;
}

GstPadProbeReturn VFrameGrabber::padDataMod(GstPad *_pad, GstPadProbeInfo *_info,
        gpointer _uData)
{
    return GST_PAD_PROBE_OK;
}

gint64 VFrameGrabber::getTotalTime()
{
    return 1800000000000;
}

gint64 VFrameGrabber::getPosCurrent()
{
    gint64 pos;
    gst_element_query_position(GST_ELEMENT(m_pipeline), GST_FORMAT_TIME, &pos);
    return pos;
}

void VFrameGrabber::restartPipeline()
{
    GstStateChangeReturn result =
        gst_element_set_state(GST_ELEMENT(m_pipeline), GST_STATE_PLAYING);

    if (result != GST_STATE_CHANGE_SUCCESS) {
        printf("ReadingCam-gstreamer failed to set pipeline state to PLAYING "
               "(error %u)\n",
               result);
        return;
    }
}

bool VFrameGrabber::initPipeline()
{
    m_filename =  getFileNameByTime();

    if (createFolder("flights")) {
        m_filename = "flights/" + m_filename;
    }

    //
    //----- 1
    //    m_pipelineStr =
    //        "udpsrc multicast-group=" + m_ip + " port=" + std::to_string(m_port) +
    //        " ! application/x-rtp,media=video,clock-rate=90000,encoding-name=MP2T-ES "
    //        "! rtpjitterbuffer mode=synced ! rtpmp2tdepay ! video/mpegts ! tsdemux ! "
    //        "queue name=myqueue ! tee name=t t. ! h265parse ! "
    //        "video/x-h265,stream-format=byte-stream ! nvh265dec ! glcolorconvert ! video/x-raw(memory:GLMemory),format=BGRA ! " +
    //        "appsink  name=mysink sync=false async=false ";
    //    +  "t. ! h265parse ! queue ! matroskamux ! filesink location=" + m_filename + ".mkv";
    //----- 2
    //    m_pipelineStr =
    //        "udpsrc multicast-group=" + m_ip + " port=" + std::to_string(m_port) +
    //        " ! application/x-rtp,media=video,clock-rate=90000,encoding-name=MP2T-ES "
    //        "! rtpjitterbuffer mode=synced ! rtpmp2tdepay ! video/mpegts ! tsdemux ! "
    //        "queue name=myqueue ! tee name=t t. ! h265parse ! "
    //        "video/x-h265,stream-format=byte-stream ! nvh265dec ! " +
    //        "appsink  name=mysink sync=false async=false ";
    //
    //----- 3
    //    m_pipelineStr =
    //        "udpsrc multicast-group=" + m_ip + " port=" + std::to_string(m_port) +
    //        " ! application/x-rtp,media=video,clock-rate=90000,encoding-name=MP2T-ES "
    //        "! rtpjitterbuffer mode=synced ! rtpmp2tdepay ! video/mpegts ! tsdemux ! "
    //        "queue name=myqueue ! tee name=t t. ! h265parse ! "
    //        "video/x-h265,stream-format=byte-stream ! avdec_h265 ! " +
    //        "appsink  name=mysink sync=false async=false ";
    //----- 4
    //    m_pipelineStr = "rtspsrc location=" + m_ip + " ! queue ! rtph265depay ! h265parse ! nvh265dec ! glcolorconvert ! video/x-raw(memory:GLMemory),format=I420 ! appsink name=mysink sync=false async=false";
//    m_pipelineStr = "rtspsrc location=" + m_ip + " ! decodebin ! appsink name=mysink sync=true async=true";
    m_pipelineStr = "filesrc location=/home/pgcs-04/Videos/IMG_3250_fullHD.mp4 ! decodebin ! appsink name=mysink sync=true async=true";
    //
    //------ 2
    //    m_pipelineStr =
    //        "udpsrc multicast-group=" + m_ip + " port=" + std::to_string(m_port) +
    //        " ! application/x-rtp,media=video,clock-rate=90000,encoding-name=MP2T-ES "
    //        "! " +
    //        "rtpjitterbuffer mode=synced ! rtpmp2tdepay ! video/mpegts ! tsdemux ! "
    //        "queue name=myqueue ! tee name=t t. ! h265parse ! "
    //        "video/x-h265,stream-format=byte-stream ! avdec_h265 max-threads=4 ! " +
    //        "appsink  name=mysink sync=false async=false "
    //        +  "t. ! h265parse ! queue ! matroskamux ! filesink location=" + m_filename + ".mkv";
    //
    //
    //------ 3
    //    m_pipelineStr = "filesrc location=/home/pgcs-03/Videos/flights/ahihi.avi ! avidemux ! "
    //                    "h264parse ! avdec_h264 ! appsink name=mysink sync=true";
    //    m_pipelineStr = "filesrc location=/home/qdt/Videos/fullhd_60.mp4 ! avidemux ! "
    //                    "h264parse ! avdec_h264 ! appsink name=mysink sync=true";
    //------ 4
    //    m_pipelineStr = "filesrc location=/home/pgcs-03/Videos/flights/ahihi.mkv ! matroskademux ! "
    //                    "h265parse ! nvh265dec ! glcolorconvert ! video/x-raw(memory:GLMemory),format=BGRA ! appsink name=mysink sync=true";
    printf("\nReading pipeline: %s", m_pipelineStr.data());
    m_pipeline = GST_PIPELINE(gst_parse_launch(m_pipelineStr.data(), &m_err));

    if ((m_err != NULL) || (!m_pipeline)) {
        g_print("gstreamer decoder failed to create pipeline\n");
        g_error_free(m_err);
        return false;
    } else {
        g_print("gstreamer decoder create pipeline success\n");
    }

    GstElement *m_sink = gst_bin_get_by_name((GstBin *)m_pipeline, "mysink");
    GstAppSink *m_appsink = (GstAppSink *)m_sink;

    if (!m_sink || !m_appsink) {
        g_print("Fail to get element \n");
        return false;
    }

    // drop
    gst_app_sink_set_drop(m_appsink, true);
    g_object_set(m_appsink, "emit-signals", TRUE, NULL);
    // check end of stream
    m_bus = gst_pipeline_get_bus(GST_PIPELINE(m_pipeline));
    m_busWatchID = gst_bus_add_watch(m_bus, wrapperOnBusCall, (void *)this);
    gst_object_unref(m_bus);
    // add call back received video data
    GstAppSinkCallbacks cbs;
    memset(&cbs, 0, sizeof(GstAppSinkCallbacks));
    cbs.new_sample = wrapperOnNewSample;
    cbs.eos = wrapperOnEOS;
    cbs.new_preroll = wrapperOnNewPreroll;
    gst_app_sink_set_callbacks(m_appsink, &cbs, (void *)this, NULL);
    // add call back received meta data
    GstElement *vqueue = gst_bin_get_by_name(GST_BIN(m_pipeline), "myqueue");
    GstPad *pad = gst_element_get_static_pad(vqueue, "sink");
    gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER,
                      (GstPadProbeCallback)wrapperPadDataMod, (void *)this, NULL);
    gst_object_unref(pad);
    return true;
}

std::string VFrameGrabber::getFileNameByTime()
{
    std::string fileName = "";
    std::time_t t = std::time(0);
    std::tm *now = std::localtime(&t);
    fileName += std::to_string(now->tm_year + 1900);
    correctTimeLessThanTen(fileName, now->tm_mon + 1);
    correctTimeLessThanTen(fileName, now->tm_mday);
    correctTimeLessThanTen(fileName, now->tm_hour);
    correctTimeLessThanTen(fileName, now->tm_min);
    correctTimeLessThanTen(fileName, now->tm_sec);
    return fileName;
}

void VFrameGrabber::correctTimeLessThanTen(std::string &_inputStr, int _time)
{
    _inputStr += "_";

    if (_time < 10) {
        _inputStr += "0";
        _inputStr += std::to_string(_time);
    } else {
        _inputStr += std::to_string(_time);
    }
}


bool VFrameGrabber::createFolder(std::string _folderName)
{
    if (!checkIfFolderExist(_folderName)) {
        const int dir_err = mkdir(_folderName.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

        if (-1 == dir_err) {
            return false;
        }
    }

    return true;
}

bool VFrameGrabber::checkIfFolderExist(std::string _folderName)
{
    struct stat st;

    if (stat(_folderName.c_str(), &st) == 0) {
        return true;
    }

    return false;
}

void VFrameGrabber::stopPipeline()
{
    if (m_loop != nullptr &&  g_main_loop_is_running(m_loop) == TRUE) {
        g_main_loop_quit(m_loop);
    }
}
