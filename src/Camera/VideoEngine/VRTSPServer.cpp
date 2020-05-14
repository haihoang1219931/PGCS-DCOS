#include "VRTSPServer.h"
VRTSPServer::VRTSPServer()
{
    gst_init(nullptr, nullptr);
}

VRTSPServer::~VRTSPServer()
{

    printf("Deconstruct VRTSPServer\r");
}

void VRTSPServer::run()
{
    m_gstRTSPBuff = Cache::instance()->getGstRTSPCache();
    printf("===========> start RTSP server\n");
    this->gstreamer_pipeline_operate();
    printf("===========> stop RTSP server\n");
}

/// @brief Removes clients from the server, called by rtsp server
GstRTSPFilterResult VRTSPServer::wrap_clientFilter(GstRTSPServer* server, GstRTSPClient* client, gpointer user){
    VRTSPServer *itself = static_cast<VRTSPServer *>(user);
    itself->clientFilter(server,client,nullptr);
}
GstRTSPFilterResult VRTSPServer::clientFilter(GstRTSPServer* server, GstRTSPClient* client, gpointer user)
{
    return GST_RTSP_FILTER_REMOVE;
}
void VRTSPServer::wrap_clientClosed(GstRTSPClient* client, gpointer user){
    VRTSPServer *itself = static_cast<VRTSPServer *>(user);
    itself->clientClosed(client,nullptr);
}
void VRTSPServer::clientClosed(GstRTSPClient* client, gpointer user)
{
    g_print("client closed - count: %u\n", --m_clientCount);
}
void VRTSPServer::wrap_clientConnected(GstRTSPServer* server, GstRTSPClient* client, gpointer user){
    VRTSPServer *itself = static_cast<VRTSPServer *>(user);
    itself->clientConnected(server,client,nullptr);
}
void VRTSPServer::clientConnected(GstRTSPServer* server, GstRTSPClient* client, gpointer user)
{
    // hook the client close callback

    g_signal_connect(client, "closed", reinterpret_cast<GCallback>(wrap_clientClosed), this);

    g_print("client-connected -- count: %u\n", ++m_clientCount);
}

/// @brief Closes clients, called by the app
void VRTSPServer::wrap_closeClients(GstRTSPServer* server, gpointer user){
    VRTSPServer *itself = static_cast<VRTSPServer *>(user);
    itself->closeClients(server,nullptr);
}
void VRTSPServer::closeClients(GstRTSPServer* server,gpointer user)
{
    g_print("closing clients - count: %d...\n", m_clientCount);

    if (0 < m_clientCount)
        gst_rtsp_server_client_filter(server, wrap_clientFilter, this);
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
    if (m_gstRTSPBuff->size() == 0) {
        return GstFlowReturn::GST_FLOW_CUSTOM_SUCCESS;
    }
//    printf("VRTSPServer::need_data\r\n");
    GstFrameCacheItem gstFrame = m_gstRTSPBuff->last();
    while ((gstFrame.getIndex() == -1) || (gstFrame.getIndex() == m_currID)) {
        gstFrame = m_gstRTSPBuff->last();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
    }

    m_currID = gstFrame.getIndex();
    GstFlowReturn ret = GstFlowReturn::GST_FLOW_OK;
//    if(m_currID > 0)
    {
        GstBuffer *img_stream = gstFrame.getGstBuffer();
        if(img_stream != nullptr){
            gsize bufSize = gst_buffer_get_size(img_stream);
//            printf("===> RTSP Stream: [%d] -%d\r\n",gstFrame.getIndex(),bufSize);
            if(static_cast<int>(bufSize) >= (m_width * m_height * 3/2))
            img_stream->pts = static_cast<GstClockTime>(timestamp);
            img_stream->duration = gst_util_uint64_scale_int(1, GST_SECOND, m_fps);
            timestamp += img_stream->duration;
            g_signal_emit_by_name(appsrc, "push-buffer", img_stream, &ret);
        }

    }
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
    printf("media_configure media = %p size(%d,%d)\r\n", media,m_width,m_height);
    this->media = media;
    element = gst_rtsp_media_get_element(media);
    appsrc = gst_bin_get_by_name_recurse_up(GST_BIN(element), "othersrc");
    gst_util_set_object_arg(G_OBJECT(appsrc), "format", "time");
    /* configure the caps of the video */
    g_object_set(G_OBJECT(appsrc), "caps",
                 gst_caps_new_simple("video/x-raw",
                                     "format", G_TYPE_STRING, "I420",
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
bool VRTSPServer::gstreamer_pipeline_operate()
{
    loop = g_main_loop_new(nullptr, FALSE);
    server = gst_rtsp_server_new();
    char cPort[16];
    sprintf(cPort,"%d",m_port);
    gst_rtsp_server_set_service(server,cPort);
    mounts = gst_rtsp_server_get_mount_points(server);
    factory = gst_rtsp_media_factory_new();
    gst_rtsp_media_factory_set_launch(factory, m_source.toStdString().c_str());
    //    gst_rtsp_media_factory_set_launch(factory, "( ximagesrc ! video/x-raw,framerate=30/1 ! videoconvert ! avenc_mpeg4 bitrate=4000000 ! rtpmp4vpay name=pay0 pt=96 )");
    //    gst_rtsp_media_factory_set_launch(factory, "( videotestsrc ! video/x-raw,width=352,height=288,framerate=15/1 ! x264enc ! rtph264pay name=pay0 pt=96 )");
    printf("gstreamer_pipeline_operate user_data %p\r\n", this);
    g_signal_connect(factory, "media-configure", (GCallback) wrap_media_configure, this);
    gst_rtsp_media_factory_set_shared(factory, TRUE);
    gst_rtsp_mount_points_add_factory(mounts, (const gchar *)m_streamMount.data(), factory);
    g_object_unref(mounts);
    m_rtspAttachID = gst_rtsp_server_attach(server, NULL);
    g_signal_connect(server, "client-connected", reinterpret_cast<GCallback>(wrap_clientConnected), this);
    printf("stream [%d] ready at rtsp://127.0.0.1:%s%s -- [%d - %d]\n",m_rtspAttachID,cPort, m_streamMount.data(), m_width, m_height);
    while(!m_stop){
        g_main_loop_run(loop);
        printf("Loop stopped\r\n");

        if (0 < m_clientCount)
        {
            g_print("closing all clients\n");
            wrap_closeClients(server,this);
        }
    }
    bool ret = g_source_remove (m_rtspAttachID);
    gst_rtsp_media_factory_set_eos_shutdown(factory,true);
    //    g_object_disconnect(server, "client-connected", reinterpret_cast<GCallback>(wrap_clientConnected), this);
    printf("Stop stream[%d] return %s\r\n",m_rtspAttachID,ret?"true":"false");
    if (G_IS_OBJECT(server))
    {
        g_print("Ref Count: %u\n", GST_OBJECT_REFCOUNT_VALUE(server));
        g_print("unref server\n");
        g_object_unref(server);
        server = nullptr;
    }
    if (G_IS_OBJECT(loop)){
        g_print("unref loop\n");
        g_object_unref(loop);
        loop = nullptr;
    }
}
void VRTSPServer::setStateRun(bool running)
{
    if(!running){
        if (nullptr != loop)
        {
            g_print("quitting main loop\n");
            g_main_loop_quit(loop);
        }
    }
}
void VRTSPServer::pause(bool pause){

}
void VRTSPServer::setStreamPort(int port){
    m_port = port;
}
void VRTSPServer::setStreamMount(QString _streamMount)
{
    m_streamMount = _streamMount.toStdString();
}

void VRTSPServer::setStreamSize(int _width, int _height)
{
    m_width = _width;
    m_height = _height;
}
