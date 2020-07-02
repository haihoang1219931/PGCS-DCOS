#include "VFrameGrabber.h"

VFrameGrabber::VFrameGrabber()
{
    gst_init(0, NULL);
    m_loop = g_main_loop_new(NULL, FALSE);
    m_gstFrameBuff = Cache::instance()->getGstFrameCache();
    m_gstEOSavingBuff = Cache::instance()->getGstEOSavingCache();
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

    pause(true);
    this->initPipeline();
    pause(false);
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
gboolean VFrameGrabber::wrapNeedKlv(GstAppSrc* appsrc,void* userPointer){
    VFrameGrabber *itseft = (VFrameGrabber *)userPointer;
    return itseft->needKlv(appsrc, userPointer);
}
gboolean VFrameGrabber::needKlv(GstAppSrc* appsrc,void* userPointer)
{
    printf("NeedKLV\r\n");
    uint8_t data[] = {
        0X06,0X0E,0X2B,0X34,0X02,0X0B,0X01,0X01,0X0E,0X01,0X03,0X01,0X01,0X00,0X00,0X00,0X81,0XF1,0X02,0X08,0X00,0X04,0XCA,0X14,0X28,0X28,0XCD,0X02,0X03,0X15,0X45,0X53,0X52,0X49,0X5F,0X4D,0X65,0X74,0X61,0X64,0X61,0X74,0X61,0X5F,0X43,0X6F,0X6C,0X6C,0X65,0X63,0X74,0X04,0X06,0X4E,0X39,0X37,0X38,0X32,0X36,0X05,0X02,0X6C,0X97,0X06,0X02,0X13,0XDA,0X07,0X02,0XDD,0X0B,0X0A,0X05,0X43,0X32,0X30,0X38,0X42,0X0B,0X00,0X0C,0X00,0X0D,0X04,0X3A,0X72,0X10,0X0A,0X0E,0X04,0XB5,0X6D,0X14,0X7C,0X0F,0X02,0X31,0X4E,0X10,0X02,0X04,0X5D,0X11,0X02,0X02,0X74,0X12,0X04,0XB6,0XE0,0XB6,0X0C,0X13,0X04,0XF6,0X3B,0XBB,0XBC,0X14,0X04,0X00,0X00,0X00,0X00,0X15,0X04,0X00,0X1E,0X02,0XFE,0X16,0X02,0X00,0X00,0X17,0X04,0X3A,0X76,0X50,0X62,0X18,0X04,0XB5,0X70,0X74,0XAF,0X19,0X02,0X23,0X9F,0X1A,0X02,0X01,0X7F,0X1B,0X02,0X00,0X5F,0X1C,0X02,0X00,0X01,0X1D,0X02,0X02,0X04,0X1E,0X02,0XFE,0X97,0X1F,0X02,0XFF,0XA7,0X20,0X02,0X00,0X00,0X21,0X02,0XFE,0X1A,0X2F,0X01,0X00,0X30,0X2A,0X01,0X01,0X01,0X02,0X01,0X01,0X03,0X04,0X2F,0X2F,0X43,0X41,0X04,0X00,0X05,0X00,0X06,0X02,0X43,0X41,0X15,0X10,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X16,0X02,0X00,0X05,0X38,0X01,0X00,0X3B,0X08,0X46,0X69,0X72,0X65,0X62,0X69,0X72,0X64,0X41,0X01,0X01,0X48,0X08,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X01,0X02,0XBB,0X33,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X00,0X22,0X32,0X01,0X00,0X00,0X00,0X00
    };
    GstBuffer *buffer = gst_buffer_new_allocate(nullptr, sizeof (data), nullptr);
    GstMapInfo map;
    GstClock *clock;
    GstClockTime abs_time, base_time;

    gst_buffer_map (buffer, &map, GST_MAP_WRITE);
    memcpy(map.data, data, sizeof (data));
    gst_buffer_unmap (buffer, &map);

    GST_OBJECT_LOCK (appsrc);
    clock = GST_ELEMENT_CLOCK (appsrc);
    base_time = GST_ELEMENT (appsrc)->base_time;
    gst_object_ref (clock);
    GST_OBJECT_UNLOCK (appsrc);
    abs_time = gst_clock_get_time (clock);
    gst_object_unref (clock);

    GST_BUFFER_PTS (buffer) = abs_time - base_time;
    GST_BUFFER_DURATION (buffer) = GST_SECOND / 30;

    gst_app_src_push_buffer(GST_APP_SRC(appsrc), buffer);
    int ms = 1000 / 30 * 5;
    struct timespec ts = { ms / 1000, (ms % 1000) * 1000 * 1000 };
    nanosleep(&ts, NULL);
    return true;
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
    GstBuffer *gstItem = gst_sample_get_buffer(sample);

    GstFrameCacheItem gstFrame;
    gstFrame.setIndex(m_currID);
    gstFrame.setGstBuffer(gst_buffer_copy(gstItem));
    m_gstFrameBuff->add(gstFrame);
    if (*m_enSaving){
        GstFrameCacheItem gstEOSaving;
        gstEOSaving.setIndex(m_currID);
        gstEOSaving.setGstBuffer(gst_buffer_copy(gstItem));
        m_gstEOSavingBuff->add(gstEOSaving);
    }
    //    printf("\nReadFrame %d - %d", m_currID, gst_buffer_get_size(gst_sample_get_buffer(sample)));
    gst_sample_unref(sample);
    return GST_FLOW_OK;
}

void VFrameGrabber::onEOS(_GstAppSink *_sink, void *_uData)
{
    printf("\ngstreamer decoder onEOS");
}

gboolean VFrameGrabber::onBusCall(GstBus *_bus, GstMessage *_msg,
                                  gpointer _uData)
{
    GMainLoop *loop = (GMainLoop *)_uData;

    switch (GST_MESSAGE_TYPE(_msg)) {
    case GST_MESSAGE_EOS: {
        g_print("\nEnd of stream");
        //        g_main_loop_quit(loop);
        break;
    }

    case GST_MESSAGE_ERROR: {
        gchar *debug;
        GError *error;
        gst_message_parse_error(_msg, &error, &debug);
        g_free(debug);
        g_printerr("\nError: %s", error->message);
        g_error_free(error);
        //        g_main_loop_quit(loop);
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
    return m_totalTime;
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
void VFrameGrabber::setSpeed(float speed){
    if(m_pipeline == NULL) {
        printf("m_pipeline == NULL\r\n");
        return;
    }
    printf("Change speed to %f\r\n",speed);
    gint64 posCurrent = getPosCurrent();
    printf("%ld = posCurrent\r\n",posCurrent);
    m_speed = speed;
    pause(true);
    gst_element_seek(GST_ELEMENT(m_pipeline),speed,
                     GST_FORMAT_TIME,
                     GstSeekFlags(GST_SEEK_FLAG_FLUSH | GST_SEEK_FLAG_ACCURATE),
                     GST_SEEK_TYPE_SET,(gint64)(posCurrent),
                     GST_SEEK_TYPE_SET,GST_CLOCK_TIME_NONE);
    pause(false);
}
void VFrameGrabber::pause(bool pause){
    if(m_pipeline == NULL) return;
    if(pause){
        gst_element_set_state(GST_ELEMENT(m_pipeline), GST_STATE_PAUSED);
    }else{
        gst_element_set_state(GST_ELEMENT(m_pipeline), GST_STATE_PLAYING);
    }
}
void VFrameGrabber::goToPosition(float percent){
    printf("goToPosition %f%\r\n",percent);
    if(m_pipeline == NULL) {
        printf("m_pipeline == NULL\r\n");
        return;
    }
    gint64 posNext = (gint64)((double)m_totalTime*(double)percent);
    //    printf("%ld = (gint64)(percent*100*GST_SECOND)\r\n",(gint64)(percent*100*GST_SECOND));
    //    printf("%ld = posNext\r\n",posNext);
    //    printf("%f = m_speed\r\n",m_speed);
    gst_element_seek(GST_ELEMENT(m_pipeline),m_speed,
                     GST_FORMAT_TIME,
                     GstSeekFlags(GST_SEEK_FLAG_FLUSH | GST_SEEK_FLAG_ACCURATE),
                     GST_SEEK_TYPE_SET,(gint64)(posNext),
                     GST_SEEK_TYPE_SET,GST_CLOCK_TIME_NONE);
}
bool VFrameGrabber::initPipeline()
{
    if (m_pipeline != nullptr) {
        gst_element_set_state(GST_ELEMENT(m_pipeline), GST_STATE_NULL);
    }
    m_filename =  getFileNameByTime();

    if (createFolder("flights")) {
        m_filename = "flights/" + m_filename;
    }
    createFolder("img");
    createFolder("plates");
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
    char pipeline[256];
    std::string saveElement = "tee name=t t. ! queue ! mpegtsmux name=mux mux. ! filesink location="+m_filename+".mp4 ! ";
    sprintf(pipeline,m_ip.c_str(),saveElement.c_str());
    std::string pipelineSaveInclude(pipeline);
    if(QString::fromStdString(m_ip).contains("filesrc")){
        m_pipelineStr = pipelineSaveInclude + " ! appsink name=mysink sync=true async=true";
    }else{
        m_pipelineStr = pipelineSaveInclude + " ! appsink name=mysink sync=false async=false";
    }
    //    m_pipelineStr = "filesrc location=/home/pgcs-04/Videos/vt/vt5.mp4 ! decodebin ! appsink name=mysink sync=true async=true";
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

    GstAppSink *m_appsink = (GstAppSink *)gst_bin_get_by_name((GstBin *)m_pipeline, "mysink");

    if (!m_appsink) {
        g_print("Fail to get element \n");
        //        return false;
    }else{
        // drop
        gst_app_sink_set_drop(m_appsink, true);
        g_object_set(m_appsink, "emit-signals", TRUE, NULL);
        // add call back received video data
        GstAppSinkCallbacks cbs;
        memset(&cbs, 0, sizeof(GstAppSinkCallbacks));
        cbs.new_sample = wrapperOnNewSample;
        cbs.eos = wrapperOnEOS;
        cbs.new_preroll = wrapperOnNewPreroll;
        gst_app_sink_set_callbacks(m_appsink, &cbs, (void *)this, NULL);
    }

    // check end of stream
    m_bus = gst_pipeline_get_bus(GST_PIPELINE(m_pipeline));
    m_busWatchID = gst_bus_add_watch(m_bus, wrapperOnBusCall, (void *)this);
    gst_object_unref(m_bus);

    // add call back save meta to file
    GstAppSrc *klvsrc = (GstAppSrc *)gst_bin_get_by_name((GstBin *)m_pipeline, "klvsrc");

    if (!klvsrc) {
        g_print("Fail to get klvsrc \n");
        //        return false;
    }else{
        //        g_signal_connect (klvsrc, "seek-data", G_CALLBACK (seek_data), this);
        g_signal_connect (klvsrc, "need-data", G_CALLBACK (wrapNeedKlv), this);
        /* set the caps on the source */
        GstCaps *caps = gst_caps_new_simple ("meta/x-klv",
                                             "parsed", G_TYPE_BOOLEAN, TRUE,
                                             nullptr);
        gst_app_src_set_caps(GST_APP_SRC(klvsrc), caps);
        g_object_set(GST_APP_SRC(klvsrc), "format", GST_FORMAT_TIME, nullptr);
        //        g_signal_connect (klvsrc, "enough-data", G_CALLBACK (stop_feed), this);
    }

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
