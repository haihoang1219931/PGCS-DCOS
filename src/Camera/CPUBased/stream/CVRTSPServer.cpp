#include "CVRTSPServer.h"

CVRTSPServer::CVRTSPServer(QObject *parent) : QObject(parent)
{
    gst_init(nullptr,nullptr);
}

CVRTSPServer::~CVRTSPServer()
{

}
GstFlowReturn
CVRTSPServer::wrap_need_data (GstElement * appsrc, guint unused, gpointer user_data){
    CVRTSPServer * itself = static_cast<CVRTSPServer*>(user_data);
    return itself->need_data(appsrc,unused,nullptr);
}
void
CVRTSPServer::wrap_enough_data (GstElement * appsrc, guint unused, gpointer user_data){
    CVRTSPServer * itself = static_cast<CVRTSPServer*>(user_data);
    itself->enough_data(appsrc,unused,nullptr);
}
gboolean CVRTSPServer::wrap_seek_data (GstElement* object,
                                         guint64 arg0,
                                           gpointer user_data){
    CVRTSPServer * itself = static_cast<CVRTSPServer*>(user_data);
    itself->seek_data(object,arg0,nullptr);
}
void
CVRTSPServer::enough_data (GstElement * appsrc, guint unused, gpointer user_data)
{
    Q_UNUSED(appsrc);
    Q_UNUSED(unused);
    Q_UNUSED(user_data);
    printf("Enough Data\r\n");
}

GstFlowReturn
CVRTSPServer::need_data (GstElement * appsrc, guint unused, gpointer user_data)
{
    Q_UNUSED(unused);
    Q_UNUSED(user_data);
    GstBuffer *buffer;
    guint size;
    GstFlowReturn ret;
    GstSample *sample;
    GstMapInfo map;
    std::pair <int,GstSample*> data;
//    printf("m_imageQueue->size() = %d\r\n",m_imageQueue->size());
    if(m_imageQueue->size() > 0){
        data = m_imageQueue->back();
    }else {
        return GST_FLOW_FLUSHING;
    }
    sample = data.second;
    size = static_cast<guint>(m_width * m_height * 3 / 2) ;
    buffer = gst_sample_get_buffer (sample);
    /* check buffer size */
    gst_buffer_map(buffer, &map, GST_MAP_READ);
//    printf("frame size[%dx%d] %d\r\n",m_width,m_height,map.size);
    if(map.size != size){
        gint32 width,height;
        GstCaps *caps;
        gboolean res;
        GstBuffer *buf;
        GstStructure *str;
        bool error = false;
        caps = gst_sample_get_caps (sample);
        if(!GST_IS_CAPS(caps)) {
            error = true;
        }
        str = gst_caps_get_structure(caps, 0);
        if(!GST_IS_STRUCTURE(str)) {
            error = true;
        }
        res = gst_structure_get_int (str, "width", &width);
        res |= gst_structure_get_int (str, "height", &height);
        if (!res || width == 0 || height == 0)
        {
            g_print ("could not get snapshot dimension\n");
            error = true;
        }
        if(!error){
//            printf("picYV12[%dx%d]\r\n",width,height);
            cv::Mat picYV12 = cv::Mat(height * 3 / 2 ,  map.size / height /3*2, CV_8UC1, map.data);
            cv::Mat picFHD;
            cv::resize(picYV12,picFHD,cv::Size(m_width ,  m_height * 3 / 2));
//            printf("create buff %d\r\n",size);
            buf = gst_buffer_new_allocate (NULL, size, NULL);
            gst_buffer_memset (buf, 0, 0x0, m_width*m_height*3/2);
            gst_buffer_fill(buf,0,picFHD.data,size);
            buf->pts = static_cast<GstClockTime>(timestamp);
            buf->duration = gst_util_uint64_scale_int (1, GST_SECOND, m_fps);
            timestamp += buf->duration;
            g_signal_emit_by_name (appsrc, "push-buffer", buf, &ret);
            gst_buffer_unref (buf);
        }
    }else{
        buffer->pts = static_cast<GstClockTime>(timestamp);
        buffer->duration = gst_util_uint64_scale_int (1, GST_SECOND, m_fps);
        timestamp += buffer->duration;
        g_signal_emit_by_name (appsrc, "push-buffer", buffer, &ret);
//        printf("Push data return %d\r\n",ret);
    }
    gst_buffer_unmap(buffer, &map);
    /* increment the timestamp every 1/30 second */

    return ret;
}
gboolean CVRTSPServer::seek_data (GstElement* object,
                                         guint64 arg0,
                        gpointer user_data){
    printf("User Data\r\n");
}
void
CVRTSPServer::wrap_media_configure (GstRTSPMediaFactory * factory, GstRTSPMedia * media,
                                    gpointer user_data){
    printf("wrap_media_configure user_data %p\r\n",user_data);
    CVRTSPServer * itself = static_cast<CVRTSPServer*>(user_data);
    itself->media_configure(factory,media,nullptr);
}
void
CVRTSPServer::media_configure (GstRTSPMediaFactory * factory, GstRTSPMedia * media,
    gpointer user_data)
{
    Q_UNUSED(factory);
    Q_UNUSED(user_data);
    printf("media_configure media = %p\r\n",media);
    this->media = media;
    element = gst_rtsp_media_get_element (media);
    appsrc = gst_bin_get_by_name_recurse_up (GST_BIN (element), "othersrc");

    gst_util_set_object_arg (G_OBJECT (appsrc), "format", "time");
    /* configure the caps of the video */
    g_object_set (G_OBJECT (appsrc), "caps",
        gst_caps_new_simple ("video/x-raw",
          "format", G_TYPE_STRING, "I420",
          "width", G_TYPE_INT, m_width,
          "height", G_TYPE_INT, m_height,
          "framerate", GST_TYPE_FRACTION, m_fps, 1, NULL), NULL);
    /* install the callback that will be called when a buffer is needed */
    g_signal_connect (appsrc, "need-data", (GCallback) wrap_need_data, this);
    g_signal_connect (appsrc, "enough-data", (GCallback) wrap_enough_data, this);
    g_signal_connect (appsrc, "seek-data", (GCallback) wrap_seek_data, this);
//    gst_object_unref (appsrc);
//    gst_object_unref (element);
}
gboolean CVRTSPServer::gstreamer_pipeline_operate() {
    loop = g_main_loop_new (NULL, FALSE);

    server = gst_rtsp_server_new ();
    mounts = gst_rtsp_server_get_mount_points (server);
    factory = gst_rtsp_media_factory_new ();
    gst_rtsp_media_factory_set_launch (factory,
        "( appsrc name=othersrc ! videoconvert ! timeoverlay ! avenc_mpeg4 bitrate=4000000 ! rtpmp4vpay name=pay0 pt=96 )");

    printf("gstreamer_pipeline_operate user_data %p\r\n",this);
    g_signal_connect (factory, "media-configure", (GCallback) wrap_media_configure, this);
    gst_rtsp_media_factory_set_shared (factory, TRUE);
    gst_rtsp_mount_points_add_factory (mounts, "/stream", factory);
    g_object_unref (mounts);
    gst_rtsp_server_attach (server, NULL);

    g_print ("stream ready at rtsp://127.0.0.1:8554/stream\n");
    g_main_loop_run (loop);
}
void CVRTSPServer::changeResolution(int width,int height,int fps){
//    printf("changeResolution to (%dx%d)-%d\r\n",width,height,fps);
//    printf("changeResolution media = %p\r\n",media);
//    if(appsrc == nullptr) return;
    /* configure the caps of the video */
//    g_object_set (G_OBJECT (appsrc), "caps",
//        gst_caps_new_simple ("video/x-raw",
//          "format", G_TYPE_STRING, "I420",
//          "width", G_TYPE_INT, m_width,
//          "height", G_TYPE_INT, m_height,
//          "framerate", GST_TYPE_FRACTION, m_fps, 1, NULL), NULL);
//    g_signal_connect (factory, "media-configure", (GCallback) wrap_media_configure, this);
//    gst_rtsp_media_set_pipeline_state(media,GST_STATE_NULL);
//    g_object_set (G_OBJECT (appsrc), "caps",
//        gst_caps_new_simple ("video/x-raw",
//          "format", G_TYPE_STRING, "I420",
//          "width", G_TYPE_INT, m_width,
//          "height", G_TYPE_INT, m_height,
//          "framerate", GST_TYPE_FRACTION, m_fps, 1, NULL), NULL);
//    /* install the callback that will be called when a buffer is needed */
//    g_signal_connect (appsrc, "need-data", (GCallback) wrap_need_data, this);
//    g_signal_connect (appsrc, "enough-data", (GCallback) wrap_enough_data, this);
//    gst_rtsp_media_set_pipeline_state(media,GST_STATE_PLAYING);
}
void CVRTSPServer::pause(bool pause){
//     GstRTSPMedia * media = gst_rtsp
//    gst_rtsp_media_set_state()
}
void CVRTSPServer::doWork() {

    if(CVRTSPServer::gstreamer_pipeline_operate()) {
        g_print("Pipeline running successfully . . .\n");
    }else {
        g_print("Running Error!");
    }
    stopped();
}
void CVRTSPServer::setStateRun(bool running) {
//    printf(" CVVideoCapture::setStateRun\r\n");
    m_stop = !running;
    if(m_stop == true){
//        gst_element_set_state(GST_ELEMENT(mPipeline), GST_STATE_NULL);
        if(loop != NULL &&  g_main_loop_is_running(loop) == TRUE){
//            printf("Set video capture state to null\r\n");
            g_main_loop_quit(loop);
        }
    }
}
