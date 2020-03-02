#include "VRTSPServer.h"
VRTSPServer::VRTSPServer()
{
    gst_init(nullptr, nullptr);
}

VRTSPServer::~VRTSPServer()
{
    printf("Deconstruct VRTSPServer\r\n");
}

void VRTSPServer::run()
{
    printf("\n===========> start RTSP server");
    this->gstreamer_pipeline_operate();
    printf("\n===========> stop RTSP server");
}

GstFlowReturn VRTSPServer::wrap_need_data(GstElement *appsrc, guint unused, gpointer user_data)
{
    VRTSPServer *itself = static_cast<VRTSPServer *>(user_data);
    return itself->need_data(appsrc, unused, nullptr);
}
void VRTSPServer::wrap_enough_data(GstElement *appsrc, guint unused, gpointer user_data)
{
    VRTSPServer *itself = static_cast<VRTSPServer *>(user_data);
    itself->enough_data(appsrc, unused, nullptr);
}
gboolean VRTSPServer::wrap_seek_data(GstElement *object, guint64 arg0, gpointer user_data)
{
    VRTSPServer *itself = static_cast<VRTSPServer *>(user_data);
    itself->seek_data(object, arg0, nullptr);
}
void VRTSPServer::enough_data(GstElement *appsrc, guint unused, gpointer user_data)
{
    Q_UNUSED(appsrc);
    Q_UNUSED(unused);
    Q_UNUSED(user_data);
    printf("Enough Data\r\n");
}

GstFlowReturn VRTSPServer::need_data(GstElement *appsrc, guint unused, gpointer user_data)
{
    printf("\n====> run on need data");
    if (m_gstRTSPBuff->size() == 0) {
        return GstFlowReturn::GST_FLOW_CUSTOM_SUCCESS;
    }

    GstFrameCacheItem gstFrame = m_gstRTSPBuff->last();

//    while ((gstFrame.getIndex() == 0) || (gstFrame.getIndex() == m_currID)) {
    while ((gstFrame.getIndex() == -1) || (gstFrame.getIndex() == 0)) {
        gstFrame = m_gstRTSPBuff->last();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        printf("\nStill waitting for new data");
        continue;
    }

    m_currID = gstFrame.getIndex();
    //    unsigned char * streamData = gstFrame.getImageData();
    //    GstBuffer *gstBuffStream = gst_buffer_new_wrapped((gpointer) streamData, m_width*m_height*3/2);
    //    GstBuffer *img_stream = gst_buffer_copy(gstBuffStream);
    GstBuffer *img_stream = gstFrame.getGstBuffer();
    printf("\n===> RTSP Stream: %d", m_currID);
    GstFlowReturn ret;
    img_stream->pts = static_cast<GstClockTime>(timestamp);
    img_stream->duration = gst_util_uint64_scale_int(1, GST_SECOND, m_fps);
    timestamp += img_stream->duration;
    g_signal_emit_by_name(appsrc, "push-buffer", img_stream, &ret);
    return ret;
}
gboolean VRTSPServer::seek_data(GstElement *object,  guint64 arg0, gpointer user_data)
{
    printf("User Data\r\n");
}
void VRTSPServer::wrap_media_configure(GstRTSPMediaFactory *factory, GstRTSPMedia *media, gpointer user_data)
{
    printf("wrap_media_configure user_data %p\r\n", user_data);
    VRTSPServer *itself = static_cast<VRTSPServer *>(user_data);
    itself->media_configure(factory, media, nullptr);
}
void VRTSPServer::media_configure(GstRTSPMediaFactory *factory, GstRTSPMedia *media, gpointer user_data)
{
    Q_UNUSED(factory);
    Q_UNUSED(user_data);
    printf("media_configure media = %p\r\n", media);
    this->media = media;
    element = gst_rtsp_media_get_element(media);
    appsrc = gst_bin_get_by_name_recurse_up(GST_BIN(element), "othersrc");
    gst_util_set_object_arg(G_OBJECT(appsrc), "format", "time");
    /* configure the caps of the video */
    g_object_set(G_OBJECT(appsrc), "caps",
                 gst_caps_new_simple("video/x-raw",
                                     "format", G_TYPE_STRING, "BGRA",
                                     "width", G_TYPE_INT, m_width,
                                     "height", G_TYPE_INT, m_height,
                                     "framerate", GST_TYPE_FRACTION, m_fps, 1, NULL), NULL);
    /* install the callback that will be called when a buffer is needed */
    g_signal_connect(appsrc, "need-data", (GCallback) wrap_need_data, this);
    g_signal_connect(appsrc, "enough-data", (GCallback) wrap_enough_data, this);
    g_signal_connect(appsrc, "seek-data", (GCallback) wrap_seek_data, this);
    //    gst_object_unref (appsrc);
    //    gst_object_unref (element);
}
gboolean VRTSPServer::gstreamer_pipeline_operate()
{
    loop = g_main_loop_new(NULL, FALSE);
    server = gst_rtsp_server_new();
    mounts = gst_rtsp_server_get_mount_points(server);
    factory = gst_rtsp_media_factory_new();
    gst_rtsp_media_factory_set_launch(factory, "( appsrc name=othersrc ! videoconvert ! avenc_mpeg4 bitrate=4000000 ! rtpmp4vpay name=pay0 pt=96 )");
    //    gst_rtsp_media_factory_set_launch(factory, "( videotestsrc is-live=1 ! video/x-raw,format=BGRA,height=480,width=640,framerate=30/1 ! videoconvert ! avenc_mpeg4 bitrate=4000000 ! rtpmp4vpay name=pay0 pt=96 )");
    printf("gstreamer_pipeline_operate user_data %p\r\n", this);
    g_signal_connect(factory, "media-configure", (GCallback) wrap_media_configure, this);
    gst_rtsp_media_factory_set_shared(factory, TRUE);
    gst_rtsp_mount_points_add_factory(mounts, (const gchar *)m_streamMount.data(), factory);
    g_object_unref(mounts);
    gst_rtsp_server_attach(server, NULL);
    printf("stream ready at rtsp://127.0.0.1:8554%s -- [%d - %d]\n", m_streamMount.data(), m_width, m_height);
    g_main_loop_run(loop);
}
void VRTSPServer::setStateRun(bool running)
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

void VRTSPServer::setStreamMount(std::string _streamMount)
{
    m_streamMount = _streamMount;
}

void VRTSPServer::setStreamSize(int _width, int _height)
{
    m_width = _width;
    m_height = _height;
}
