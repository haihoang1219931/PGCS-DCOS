#include <gst/app/gstappsrc.h>
#include "CVVideoCapture.h"
#include "Camera/GimbalController/GimbalInterface.h"
#include "Camera/VideoEngine/VideoEngineInterface.h"
CVVideoCapture::CVVideoCapture(QObject *parent) : QObject(parent)
{
    gst_init(0, NULL);
}
CVVideoCapture::~CVVideoCapture()
{
    setStateRun(false);
    //    GError* err = NULL;
    //    GMainLoop *loop;
    //    GstPipeline *pipeline;
    //    GstElement *vsink;
    //    GstElement* mPipeline;
}

void CVVideoCapture::create_pipeline()
{
    /* Init gstreamer pipeline for receiver side */
    pipeline = GST_PIPELINE(gst_pipeline_new(nullptr));

    if (pipeline == NULL) {
        g_print("gst_pipeline_new failed\r\n");
    } else {
        g_print("gst_pipeline_new done\r\n");
    }
}
gint64 CVVideoCapture::getTotalTime()
{
    return m_totalTime;
}

gint64 CVVideoCapture::getPosCurrent()
{
    gint64 pos;
    gst_element_query_position(GST_ELEMENT(m_pipeline), GST_FORMAT_TIME, &pos);
    return pos;
}

void CVVideoCapture::setSpeed(float speed){
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
void CVVideoCapture::pause(bool pause){
    if(m_pipeline == NULL) return;
    if(pause){
        gst_element_set_state(GST_ELEMENT(m_pipeline), GST_STATE_PAUSED);
    }else{
        gst_element_set_state(GST_ELEMENT(m_pipeline), GST_STATE_PLAYING);
    }
}
void CVVideoCapture::goToPosition(float percent){
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
GstPadProbeReturn CVVideoCapture::wrap_pad_data_mod(GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
    //    g_print("wrap_pad_data_mod create \r\n");
    CVVideoCapture *itself = (CVVideoCapture *)user_data;
    //    g_print("wrap_pad_data_mod OK\n");
    itself->pad_data_mod(pad, info, NULL);
    //    g_print("wrap_pad_data_mod Run over here ..............\n");
    return GST_PAD_PROBE_OK;
}
GstFlowReturn CVVideoCapture::wrap_read_frame_buffer(GstAppSink *sink, gpointer user_data)
{
    //    g_print("wrap_read_frame_mod create \r\n");
    CVVideoCapture *itself = (CVVideoCapture *)user_data;
    //    g_print("wrap_read_frame_buffer ok\n");
    itself->read_frame_buffer(sink, NULL);
    //    g_print("wrap_read_frame_buffer Run over here ..............\n");
    return GST_FLOW_OK;
}
GstPadProbeReturn CVVideoCapture::pad_data_mod(GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
    if (m_stop == true) {
        return GST_PAD_PROBE_OK;
    }

    GstMapInfo map;
    guint8 *ptr;
    GstBuffer *buffer;
    gsize size;
    //    return GST_PAD_PROBE_OK;
    buffer = GST_PAD_PROBE_INFO_BUFFER(info);
    buffer = gst_buffer_make_writable(buffer);

    /* Making a buffer writable can fail (for example if it
     * cannot be copied and is used more than once)
     */
    if (buffer && !GST_IS_BUFFER(buffer)) {
        return GST_PAD_PROBE_OK;
    }

    m_time = GST_BUFFER_PTS(buffer);

    /* Mapping a buffer can fail (non-writable) */
    try {
        //        GstBuffer* buffer1 = gst_buffer_copy_region(buffer, GST_BUFFER_COPY_ALL ,0, BUFFER_SIZE*sizeof(guint8));
        GST_BUFFER_PTS(buffer) =  m_time;
        GST_PAD_PROBE_INFO_DATA(info) = buffer;
        guint8 meta[QUEUE_SIZE];

        if (gst_buffer_map(buffer, &map, (GstMapFlags)GST_MAP_READWRITE)) {
            ptr = (guint8 *) map.data;
            size = map.size;
            memcpy(meta, ptr, QUEUE_SIZE);
            gst_buffer_unmap(buffer, &map);
        }
    } catch (...) {
        std::runtime_error("Some thing went wrong! ");
    }

    return GST_PAD_PROBE_OK;
}

GstFlowReturn CVVideoCapture::read_frame_buffer(GstAppSink *sink, gpointer user_data)
{
    if (m_stop == true) {
        printf("stop read frame buffer\r\n");
        return GST_FLOW_OK;
    }

//    g_print("Run over here ..............\n");
    GstSample *sample;
    g_signal_emit_by_name(sink, "pull-sample", &sample);
    // test
    // test

    if (sample != NULL && GST_IS_SAMPLE(sample)) {
        int m_imageQueue_size = m_imageQueue->size();

        while (m_imageQueue_size > QUEUE_SIZE) {
            if (m_stop == true) {
                break;
            }

            //            printf("Pop videodata===========================\r\n");
            //            m_imageQueue->front().first = 1;
            GstSample *usedSample = m_imageQueue->front().second;

            if (GST_IS_SAMPLE(usedSample)) {
                gst_sample_unref(usedSample);
                //                printf("unref sample\r\n");
            }

            m_mutexCapture->lock();
            m_imageQueue->pop_front();
            m_mutexCapture->unlock();
            m_imageQueue_size --;
        }

        //        std::pair <int,GstSample*> data ;
        //        data = std::make_pair (0,sample);
        //        m_imageQueue->push_back(data);
        //        if(m_indexVideoData >= 4)
        if (GST_IS_SAMPLE(sample)) {
//            printf("push sample[%d]\r\n",m_frameCaptureID);
            m_mutexCapture->lock();
            m_imageQueue->push_back(std::make_pair(m_frameCaptureID, sample));
            m_mutexCapture->unlock();
            m_imageQueue_size ++;
            m_frameCaptureID++;

            if (m_frameCaptureID > MAX_FRAME_ID) {
                m_frameCaptureID = 0;
            }
        }
    }

    return GST_FLOW_OK;
}
gboolean CVVideoCapture::wrapNeedKlv(void* userPointer){
//    printf("%s\r\n",__func__);
    CVVideoCapture *itseft = (CVVideoCapture *)userPointer;
    return itseft->needKlv(userPointer);
}
void CVVideoCapture::wrapStartFeedKlv(GstElement * pipeline, guint size, void* userPointer){
//    printf("%s\r\n",__func__);
    CVVideoCapture *itseft = (CVVideoCapture *)userPointer;
    itseft->needKlv(userPointer);
//    g_idle_add ((GSourceFunc) wrapNeedKlv, userPointer);
}
gboolean CVVideoCapture::needKlv(void* userPointer)
{
    CVVideoCapture *itseft = (CVVideoCapture *)userPointer;
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
    int metaPerSecond = 5;
    GstClockTime gstDuration = GST_SECOND / 30;
    GST_BUFFER_PTS (buffer) = (itseft->m_metaID + 1) * gstDuration * metaPerSecond;
    GST_BUFFER_DURATION (buffer) = GST_SECOND / 30;
    gst_app_src_push_buffer(GST_APP_SRC(appsrc), buffer);
    itseft->m_metaID +=1;
    return true;
}

gboolean CVVideoCapture::gstreamer_pipeline_operate()
{
    loop = g_main_loop_new(NULL, FALSE);
    // launch pipeline
    std::string m_filename =  getFileNameByTime();
    std::string m_pipelineStr = m_source + std::string(" ! appsink name=mysink async=true sync=")+
        (QString::fromStdString(m_source).contains("filesrc")?std::string("true"):std::string("false"))+""
        " t. ! queue ! mpegtsmux name=mux mux. ! filesink location="+m_filename+".mp4 "
        " appsrc name=klvsrc ! mux. ";
    std::cout << m_pipelineStr.c_str() << std::endl;
    m_pipeline = gst_parse_launch(m_pipelineStr.c_str(), &err);

    if (err != NULL) {
        g_print("gstreamer decoder failed to create pipeline\n");
        g_error_free(err);
        return FALSE;
    } else {
        g_print("gstreamer decoder create pipeline success\n");
    }

    pipeline = GST_PIPELINE(m_pipeline);

    if (!pipeline) {
        printf("gstreamer failed to cast GstElement into GstPipeline\n");
        return FALSE;
    } else {
        g_print("gstreamer decoder create Gstpipeline success\n");
    }

    GstElement *m_sink = gst_bin_get_by_name((GstBin *)m_pipeline, "mysink");
    GstAppSink *m_appsink = (GstAppSink *)m_sink;

    if (!m_sink || !m_appsink) {
#ifdef DEBUG
        g_print("Fail to get element \n");
#endif
        return FALSE;
    }

    gst_app_sink_set_drop(m_appsink, true);
    g_object_set(m_appsink, "emit-signals", TRUE, NULL);
    //    gst_pipeline_use_clock(pipeline, nullptr);
    //    GstCaps *caps = gst_caps_from_string("video/x-raw, format=I420");
    //    gst_app_sink_set_caps(m_appsink, caps);
    //    gst_caps_unref(caps);
    // add call back received video data
    GstAppSinkCallbacks cbs;
    memset(&cbs, 0, sizeof(GstAppSinkCallbacks));
    cbs.new_sample = wrap_read_frame_buffer;
    gst_app_sink_set_callbacks(m_appsink, &cbs, (void *)this, NULL);
    // add call back save meta to file
    m_klvAppSrc = nullptr;
    m_klvAppSrc = (GstAppSrc *)gst_bin_get_by_name((GstBin *)m_pipeline, "klvsrc");
    if (m_klvAppSrc == nullptr) {
        g_print("Fail to get klvsrc \n");
    }else{
        gst_app_src_set_latency(m_klvAppSrc,5,30);
        g_signal_connect (m_klvAppSrc, "need-data", G_CALLBACK (wrapStartFeedKlv), (void *)this);
        /* set the caps on the source */
        GstCaps *caps = gst_caps_new_simple ("meta/x-klv",
                                             "parsed", G_TYPE_BOOLEAN, TRUE,
                                             nullptr);
        gst_app_src_set_caps(GST_APP_SRC(m_klvAppSrc), caps);
        g_object_set(GST_APP_SRC(m_klvAppSrc), "format", GST_FORMAT_TIME, nullptr);
    }

    const GstStateChangeReturn result = gst_element_set_state(m_pipeline, GST_STATE_PLAYING);

    if (result != GST_STATE_CHANGE_SUCCESS) {
        g_print("gstreamer failed to playing\n");
    }

    g_main_loop_run(loop);
    gst_element_set_state(GST_ELEMENT(m_pipeline), GST_STATE_NULL);
    //    g_object_unref(m_sink);
    //    g_object_unref(m_appsink);
    g_object_unref(m_pipeline);
    //    g_main_loop_unref(loop);
    //    g_object_unref(pipeline);
    printf("gstreamer setup done\n");
    return TRUE;
}
std::string CVVideoCapture::getFileNameByTime()
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
void CVVideoCapture::correctTimeLessThanTen(std::string &_inputStr, int _time)
{
    _inputStr += "_";

    if (_time < 10) {
        _inputStr += "0";
        _inputStr += std::to_string(_time);
    } else {
        _inputStr += std::to_string(_time);
    }
}
void CVVideoCapture::setSource(std::string source){
    m_source = source;
    if(m_pipeline != NULL){
//        setStateRun(false);
        gst_element_set_state(GST_ELEMENT(m_pipeline), GST_STATE_NULL);
        std::string m_filename =  getFileNameByTime();
        std::string m_pipelineStr = m_source + std::string(" ! appsink name=mysink async=true sync=")+
            (QString::fromStdString(m_source).contains("filesrc")?std::string("true"):std::string("false"))+""
            " t. ! queue ! mpegtsmux name=mux mux. ! filesink location="+m_filename+".mp4 "
            " appsrc name=klvsrc ! mux. "
                ;
        std::cout << m_pipelineStr.c_str() << std::endl;
        GError *err = nullptr;
        m_pipeline = gst_parse_launch(m_pipelineStr.c_str(), &err);
        if( err != NULL )
        {
#ifdef DEBUG
            g_print("gstreamer decoder failed to reset filesrc\n");
#endif
            g_error_free(err);
        }else{
            #ifdef DEBUG
            g_print("gstreamer decoder reset filesrc success\n");
#endif
        }
        pipeline = GST_PIPELINE(m_pipeline);

        if( !pipeline )
        {
            #ifdef DEBUG
            printf("gstreamer failed to cast GstElement into GstPipeline\n");
#endif
        }else{
            #ifdef DEBUG
            g_print("gstreamer decoder create Gstpipeline success\n");
#endif
        }
        GstElement *m_sink = gst_bin_get_by_name((GstBin*)m_pipeline, "mysink");
        GstAppSink *m_appsink = (GstAppSink *)m_sink;
        if(!m_sink || !m_appsink)
        {
    #ifdef DEBUG
            g_print("Fail to get element \n");
    #endif
        }
        // drop
        gst_app_sink_set_drop(m_appsink, true);
        g_object_set(m_appsink, "emit-signals", TRUE, NULL);
        // check end of stream
//        m_bus = gst_pipeline_get_bus (GST_PIPELINE(mPipeline));
//        m_bus_watch_id = gst_bus_add_watch (m_bus, wrap_bus_call, (void*)this);
//        gst_object_unref (m_bus);
        // add call back received video data
        GstAppSinkCallbacks cbs;
        memset(&cbs, 0, sizeof(GstAppSinkCallbacks));
        cbs.new_sample = wrap_read_frame_buffer;
        gst_app_sink_set_callbacks(m_appsink, &cbs, (void*)this, NULL);
        // add call back received meta data
        // add call back save meta to file
        m_klvAppSrc = nullptr;

        m_klvAppSrc = (GstAppSrc *)gst_bin_get_by_name((GstBin *)m_pipeline, "klvsrc");
        if (m_klvAppSrc == nullptr) {
            g_print("Fail to get klvsrc \n");
        }else{
            gst_app_src_set_latency(m_klvAppSrc,5,30);
            g_signal_connect (m_klvAppSrc, "need-data", G_CALLBACK (wrapStartFeedKlv), (void *)this);
            /* set the caps on the source */
            GstCaps *caps = gst_caps_new_simple ("meta/x-klv",
                                                 "parsed", G_TYPE_BOOLEAN, TRUE,
                                                 nullptr);
            gst_app_src_set_caps(GST_APP_SRC(m_klvAppSrc), caps);
            g_object_set(GST_APP_SRC(m_klvAppSrc), "format", GST_FORMAT_TIME, nullptr);
        }
        const GstStateChangeReturn result = gst_element_set_state(m_pipeline, GST_STATE_PLAYING);
        if(result != GST_STATE_CHANGE_SUCCESS)
        {
            #ifdef DEBUG
            g_print("gstreamer failed to playing\n");
#endif
        }
    }else{

    }
}
void CVVideoCapture::setStateRun(bool running)
{
    //    printf(" CVVideoCapture::setStateRun\r\n");
    m_stop = !running;

    if (m_stop == true) {
        //        gst_element_set_state(GST_ELEMENT(mPipeline), GST_STATE_NULL);
        if (loop != NULL &&  g_main_loop_is_running(loop) == TRUE) {
            //            printf("Set video capture state to null\r\n");
            g_main_loop_quit(loop);
        }
    }
}

void CVVideoCapture::doWork()
{
    CVVideoCapture::create_pipeline();

    if (CVVideoCapture::gstreamer_pipeline_operate()) {
        g_print("Pipeline running successfully . . .\n");
    } else {
        g_print("Running Error!");
    }

    Q_EMIT stopped();
}
void CVVideoCapture::msleep(int ms)
{
#ifdef __linux__
    //linux code goes here
    struct timespec ts = { ms / 1000, (ms % 1000) * 1000 * 1000 };
    nanosleep(&ts, NULL);
#elif _WIN32
    // windows code goes here
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
#else
#endif
}
void CVVideoCapture::stop()
{
    m_stop = true;
    printf("Stopping capture thread\r\n");
    gst_element_set_state(GST_ELEMENT(m_pipeline), GST_STATE_NULL);
}
