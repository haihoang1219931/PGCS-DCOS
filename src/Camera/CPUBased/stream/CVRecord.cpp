#include "CVRecord.h"

CVRecord::CVRecord(QObject *parent ): QObject(parent)
{

}
CVRecord::~CVRecord()
{

}

void CVRecord::capture(){
    std::string timestamp = Utils::get_time_stamp();
    std::string captureFile = m_logFolder+"/"+timestamp+".jpg";
    cv::imwrite(captureFile, m_img);
}
void CVRecord::doWork(){
    printf("CVRecord dowork started\r\n");
    int firstFrameCount = 0;

    while(m_stop == false){
        msleep(1000);
//        printf("=================================================\r\n");
        continue;
        auto startTotal = std::chrono::high_resolution_clock::now();
//        printf("Process time count = %d\r\n",count);
//        count ++;
        gboolean res;
        gint32 width,height;
        GstSample *sample;
        GstCaps *caps;
        GstBuffer *buf;
        GstStructure *str;
        GstMapInfo map;
        uint32_t rtpTime;
        std::pair <int,GstSample*> data;
        auto startGetData = std::chrono::high_resolution_clock::now();
//        printf("Process time count = %d\r\n",count);
//        m_mutexCapture->lock();
        if(m_imageQueue->size() > 0){
            firstFrameCount++;
            data = m_imageQueue->front();
        }else {
//            printf("m_imageQueue->size() <= 0\r\n");
           continue;
        }
//        std::pair <int,GstSample*> data = m_imageQueue->front();
//        m_mutexCapture->unlock();
//        printf("Process new Frame %d\r\n",data.first);
        if(data.first == 0){
            *m_frameID = data.first;
                sample = data.second;
//            printf("Process frame [%d]\r\n",data.first);
        }
        else if(data.first != *m_frameID)
        {

            *m_frameID = data.first;
                sample = data.second;
//            printf("Process frame [%d]\r\n",data.first);
        }
        else{
//            printf("Could not show frame\r\n");
            msleep(SLEEP_TIME);
            continue;
        }

        if(!GST_IS_SAMPLE (sample)){
//            m_flagNoSample = true;
//            g_print("Could not get sample\n");
            continue;
        }
        caps = gst_sample_get_caps (sample);
        if(!GST_IS_CAPS(caps)) {
//            g_print("Could not get cap\n");
            continue;
        }
        str = gst_caps_get_structure(caps, 0);
        if(!GST_IS_STRUCTURE(str)) {
//            g_print("Could not get structure\n");
            continue;
        }
        res = gst_structure_get_int (str, "width", &width);
        res |= gst_structure_get_int (str, "height", &height);
        if (!res || width == 0 || height == 0)
        {
            g_print ("could not get snapshot dimension\n");
            continue;
        }

        buf = gst_buffer_copy( gst_sample_get_buffer (sample) );

        if(!GST_IS_BUFFER(buf)) {
            g_print("Could not get buf\n");
            continue;
        }

        gst_buffer_map(buf, &map, GST_MAP_READ);
        cv::Mat picYV12 = cv::Mat(height * 3 / 2 ,  map.size / height /3*2, CV_8UC1, map.data);
        cv::cvtColor(picYV12, m_img, CV_YUV2RGB_YV12);
        if( firstFrameCount == 1 )
        {
            m_recorderOriginal = shared_ptr<EyePhoenix::VideoSaver>(
                        EyePhoenix::VideoSaver::Create(
                            gstEncodeType::GST_CODEC_MPEG,
                            m_img.cols, m_img.rows, 30,
                            (char*)(m_logFile+"_trungnd_original.mp4").c_str() )
                        );
            if( !m_recorderOriginal )
            {
                std::cout << "Somethings went wrong in saver initialization!!!" << std::endl;
            }
            if( !m_recorderOriginal->open() )
            {
                std::cout << "Somethings went wrong. Set saver to playing state failed" << std::endl;
            }

        }
        if(GST_IS_BUFFER(buf)){
            m_recorderOriginal->encodeFrame(buf, map.size);
        }
        gst_buffer_unmap(buf, &map);
        if(GST_IS_SAMPLE (sample)){
            gst_sample_unref (sample);
        }
        Q_EMIT processDone();
    }
    Q_EMIT stopped();
}
void CVRecord::msleep(int ms){
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
