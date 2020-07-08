#include "VFrameGrabber.h"
#include <gobject/gobject.h>
#include "Camera/VideoEngine/VideoEngineInterface.h"
VFrameGrabber::VFrameGrabber()
{
    gst_init(0, NULL);
    m_loop = g_main_loop_new(NULL, FALSE);
    m_gstFrameBuff = Cache::instance()->getGstFrameCache();
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
    while(!m_stop){
        initPipeline();
        GstStateChangeReturn result =
                gst_element_set_state(GST_ELEMENT(m_pipeline), GST_STATE_PLAYING);

        if (result != GST_STATE_CHANGE_SUCCESS) {
            printf("ReadingCam-gstreamer failed to set pipeline state to PLAYING "
                   "(error %u)\n",
                   result);
        }
        g_main_loop_run(m_loop);
        gst_element_set_state(GST_ELEMENT(m_pipeline), GST_STATE_NULL);
        gst_object_unref(m_pipeline);
        printf("Pipeline stopped");
    }
    return;
}

void VFrameGrabber::setSource(std::string _ip, int _port)
{
    m_ip = _ip;
    m_port = (uint16_t)_port;
    g_main_loop_quit(m_loop);
//    pause(true);
//    this->initPipeline();
//    pause(false);
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
gboolean VFrameGrabber::wrapNeedKlv(void* userPointer){
//    printf("%s\r\n",__func__);
    VFrameGrabber *itseft = (VFrameGrabber *)userPointer;
    return itseft->needKlv(userPointer);
}
void VFrameGrabber::wrapStartFeedKlv(GstElement * pipeline, guint size, void* userPointer){
//    printf("%s\r\n",__func__);
    VFrameGrabber *itseft = (VFrameGrabber *)userPointer;
    itseft->needKlv(userPointer);
//    g_idle_add ((GSourceFunc) wrapNeedKlv, userPointer);
}
gboolean VFrameGrabber::needKlv(void* userPointer)
{
    VFrameGrabber *itseft = (VFrameGrabber *)userPointer;
    GstAppSrc* appsrc = itseft->m_klvAppSrc;
    std::vector<uint8_t> klvData = VideoEngine::encodeMeta(m_gimbal);
    printf("%s [%d]\r\n",__func__,itseft->m_metaID);
    GstBuffer *buffer = gst_buffer_new_allocate(nullptr, klvData.size(), nullptr);
    GstMapInfo map;
    GstClock *clock;
    GstClockTime abs_time, base_time;

    gst_buffer_map (buffer, &map, GST_MAP_WRITE);
    memcpy(map.data, klvData.data(), klvData.size());
    gst_buffer_unmap (buffer, &map);
    GstClockTime gstDuration = GST_SECOND / m_metaPerSecond;
    GST_BUFFER_PTS (buffer) = (itseft->m_metaID + 1) * gstDuration;
    GST_BUFFER_DURATION (buffer) = gstDuration;
    GST_BUFFER_OFFSET(buffer) = itseft->m_metaID + 1;
    gst_app_src_push_buffer(GST_APP_SRC(appsrc), buffer);

    int ms = 1000 / m_metaPerSecond;
    struct timespec ts = { ms / 1000, (ms % 1000) * 1000 * 1000 };
    nanosleep(&ts, NULL);
    itseft->m_metaID +=1;
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
//    printf("ReadFrame %d - %d\r\n", m_currID, gst_buffer_get_size(gst_sample_get_buffer(sample)));
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
        g_signal_stop_emission_by_name(m_klvAppSrc, "need-data");
        break;
    }

    case GST_MESSAGE_ERROR: {
        gchar *debug;
        GError *error;
        gst_message_parse_error(_msg, &error, &debug);
        g_free(debug);
        g_printerr("\nError: %s", error->message);
        g_error_free(error);
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
//    if (m_pipeline != nullptr) {
//        gst_element_set_state(GST_ELEMENT(m_pipeline), GST_STATE_NULL);
//    }
    m_filename =  getFileNameByTime();

    if (createFolder("flights")) {
        m_filename = "flights/" + m_filename;
    }
    createFolder("img");
    createFolder("plates");
    std::string m_pipelineStr = m_ip + std::string(" ! appsink name=mysink async=true sync=")+
        (QString::fromStdString(m_ip).contains("filesrc")?std::string("true"):std::string("false"))+""
//        " t. ! queue ! mpegtsmux name=mux mux. ! filesink location="+m_filename+".mp4 "
//        " appsrc name=klvsrc ! mux. "
            ;
    printf("\nReading pipeline: %s\r\n", m_pipelineStr.data());
    m_pipeline = GST_PIPELINE(gst_parse_launch(m_pipelineStr.data(), &m_err));

    if ((m_err != NULL) || (!m_pipeline)) {
        g_print("gstreamer decoder failed to create pipeline\n");
        g_error_free(m_err);
        return false;
    } else {
        g_print("gstreamer decoder create pipeline success\n");
    }
    // check end of stream
    m_bus = gst_pipeline_get_bus(GST_PIPELINE(m_pipeline));
    m_busWatchID = gst_bus_add_watch(m_bus, wrapperOnBusCall, (void *)this);
    gst_object_unref(m_bus);

    GstAppSink *m_appsink = (GstAppSink *)gst_bin_get_by_name((GstBin *)m_pipeline, "mysink");

    if (m_appsink == nullptr) {
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

    // add call back save meta to file
    m_klvAppSrc = nullptr;
    m_klvAppSrc = (GstAppSrc *)gst_bin_get_by_name((GstBin *)m_pipeline, "klvsrc");
    if (m_klvAppSrc == nullptr) {
        g_print("Fail to get klvsrc \n");
    }else{
        gst_app_src_set_latency(m_klvAppSrc,m_metaPerSecond,30);
        g_signal_connect (m_klvAppSrc, "need-data", G_CALLBACK (wrapStartFeedKlv), (void *)this);
        /* set the caps on the source */
        GstCaps *caps = gst_caps_new_simple ("meta/x-klv",
                                             "parsed", G_TYPE_BOOLEAN, TRUE,
                                             nullptr);
        gst_app_src_set_caps(GST_APP_SRC(m_klvAppSrc), caps);
        g_object_set(GST_APP_SRC(m_klvAppSrc), "format", GST_FORMAT_TIME, nullptr);
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
    m_stop = true;
}
