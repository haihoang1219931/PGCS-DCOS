#include "VSavingWorker.h"

VSavingWorker::VSavingWorker(std::string _mode)
{
    m_currID = 0;
    m_loop = g_main_loop_new(nullptr, FALSE);
    m_bitrate = 4000000;
    m_frameRate = 30;
    m_width = 1920;
    m_height = 1080;

    if (std::strcmp(_mode.data(), "EO") == 0) {
        m_sensorMode = "EO";
    } else {
        m_sensorMode = "IR";
    }
}

VSavingWorker::~VSavingWorker()
{
    delete m_appSrc;
    delete m_bus;
    delete m_loop;
    delete m_pipeline;
    delete  m_err;
}

void VSavingWorker::wrapperOnEnoughData(GstAppSrc *_appSrc, gpointer _uData)
{
    VSavingWorker *itself = (VSavingWorker *) _uData;
    return itself->onEnoughData(_appSrc, _uData);
}

void VSavingWorker::wrapperOnNeedData(GstAppSrc *_appSrc, guint _size, gpointer _uData)
{
    VSavingWorker *itself = (VSavingWorker *) _uData;
    return itself->onNeedData(_appSrc, _size, _uData);
}

gboolean VSavingWorker::wrapperOnSeekData(GstAppSrc *_appSrc, guint64 _offset, gpointer _uData)
{
    VSavingWorker *itself = (VSavingWorker *) _uData;
    return itself->onSeekData(_appSrc, _offset, _uData);
}

void VSavingWorker::onEnoughData(GstAppSrc *_appSrc, gpointer _uData)
{
}

void VSavingWorker::onNeedData(GstAppSrc *_appSrc, guint _size, gpointer _uData)
{
    GstFrameCacheItem gstBuff;
    gstBuff = m_buffVideoSaving->last();

    while ((gstBuff.getIndex() == -1) || (gstBuff.getIndex() == m_currID)) {
        gstBuff = m_buffVideoSaving->last();
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }

    m_currID = gstBuff.getIndex();
    GstBuffer *img_save = gst_buffer_copy(gstBuff.getGstBuffer());
    //    printf("\n===> Saving Video: sensor = %s  |  id = %d", (m_sensorMode == Status::SensorMode::EO) ? "EO" : "IR", m_currID);
    GstClockTime gstDuration = GST_SECOND / m_frameRate;
    GST_BUFFER_PTS(img_save) = (m_countFrame + 1) * gstDuration;
    GST_BUFFER_DTS(img_save) = (m_countFrame + 1) * gstDuration;
    GST_BUFFER_DURATION(img_save) = gstDuration;
    GST_BUFFER_OFFSET(img_save) = m_countFrame + 1;
    gst_app_src_push_buffer(_appSrc, img_save);
    m_countFrame++;
}

gboolean VSavingWorker::onSeekData(GstAppSrc *_appSrc, guint64 _offset, gpointer _uData)
{
    return TRUE;
}

bool VSavingWorker::initPipeline()
{
    m_filename =  getFileNameByTime();

    if (createFolder("flights")) {
        std::string sensor_name = (m_sensorMode == "EO") ? "eo_" : "ir_";
        m_filename = "flights/" + sensor_name + m_filename;
    }

    if (m_sensorMode == "EO") {
        m_buffVideoSaving = Cache::instance()->getGstEOSavingCache();
    } else {
        m_buffVideoSaving = Cache::instance()->getGstIRSavingCache();
    }

    //    m_pipeline_str = "appsrc name=mysrcSave ! video/x-raw,format=BGRA,width="
    //                     + std::to_string(m_width) + ",height=" + std::to_string(m_height) + " ! nvh265enc bitrate=" + std::to_string(m_bitrate)
    //                     + " ! h265parse ! matroskamux ! filesink location=" + m_filename  + ".mkv";
    m_pipeline_str = "appsrc name=mysrcSave ! video/x-raw,format=I420,width="
                     + std::to_string(m_width) + ",height=" + std::to_string(m_height) + " ! nvh264enc ! h264parse ! matroskamux ! filesink location=" + m_filename  + ".mkv";
    printf("\nReading pipeline: %s", m_pipeline_str.data());
    m_pipeline = GST_PIPELINE(gst_parse_launch(m_pipeline_str.data(), &m_err));

    if (!m_pipeline) {
        printf("\ngstreamer failed to cast GstElement into GstPipeline");
        return false;
    }

    m_appSrc = GST_APP_SRC(gst_bin_get_by_name(GST_BIN(m_pipeline), "mysrcSave"));
    std::string capStr = "video/x-raw,width=" + std::to_string(m_width) + ",height=" + std::to_string(m_height) + ",format=I420,framerate=" + std::to_string(m_frameRate) + "/1";
    GstCaps *caps =  gst_caps_from_string((const gchar *)capStr.c_str());
    gst_app_src_set_caps(m_appSrc, caps);
    gst_caps_unref(caps);

    if (!m_appSrc) {
        printf("\ngstreamer failed to retrieve AppSrc element from pipeline");
        return false;
    }

    GstAppSrcCallbacks cbs;
    cbs.need_data = wrapperOnNeedData;
    cbs.enough_data = wrapperOnEnoughData;
    cbs.seek_data = wrapperOnSeekData;
    gst_app_src_set_callbacks(GST_APP_SRC_CAST(m_appSrc), &cbs, this, nullptr);
    return true;
}

void VSavingWorker::run()
{
    //    std::this_thread::sleep_for(std::chrono::seconds(2));
    GstStateChangeReturn result = gst_element_set_state(GST_ELEMENT(m_pipeline), GST_STATE_PLAYING);

    if (result != GST_STATE_CHANGE_SUCCESS) {
        printf("\nVideoSaving-gstreamer failed to set pipeline state to PLAYING (error %u)", result);
    }

    g_main_loop_run(m_loop);
    gst_element_set_state(GST_ELEMENT(m_pipeline), GST_STATE_NULL);
    gst_object_unref(m_pipeline);
    return;
}

std::string VSavingWorker::getFileNameByTime()
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

void VSavingWorker::correctTimeLessThanTen(std::string &_inputStr, int _time)
{
    _inputStr += "_";

    if (_time < 10) {
        _inputStr += "0";
        _inputStr += std::to_string(_time);
    } else {
        _inputStr += std::to_string(_time);
    }
}

bool VSavingWorker::createFolder(std::string _folderName)
{
    if (!checkIfFolderExist(_folderName)) {
        const int dir_err = mkdir(_folderName.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

        if (-1 == dir_err) {
            return false;
        }
    }

    return true;
}

bool VSavingWorker::checkIfFolderExist(std::string _folderName)
{
    struct stat st;

    if (stat(_folderName.c_str(), &st) == 0) {
        return true;
    }

    return false;
}

void VSavingWorker::stopPipeline()
{
    if (m_loop != nullptr &&  g_main_loop_is_running(m_loop) == TRUE) {
        g_main_loop_quit(m_loop);
    }
}
