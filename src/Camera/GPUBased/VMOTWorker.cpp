#include "VMOTWorker.h"

VMOTWorker::VMOTWorker() {
    m_mulTracker = new MultiTrack();
}

VMOTWorker::~VMOTWorker() {}


void VMOTWorker::run()
{
    std::chrono::high_resolution_clock::time_point start, stop;
    long sleepTime = 0;
    m_matImageBuff = Cache::instance()->getProcessImageCache();
    m_rbDetectedObjs = Cache::instance()->getDetectedObjectsCache();
    m_rbMOTObjs = Cache::instance()->getMOTCache();
    m_rbIPCEO = Cache::instance()->getMotionImageEOCache();
    m_rbIPCIR = Cache::instance()->getMotionImageIRCache();
    ProcessImageCacheItem processImgItem;
    int prevID = -1;
    float *h_gmeMat;
    float *d_gmeMat;

    while (m_running) {
        std::unique_lock<std::mutex> locker(m_mtx);

        while (!m_enOD) {
            m_cvEnMOT.wait_for(locker, std::chrono::seconds(2));
            //printf("\n===============>Still waitting for Enable Multi-Object Tracking");
        }

        start = std::chrono::high_resolution_clock::now();
        processImgItem = m_matImageBuff->at(m_matImageBuff->size() - 3);;//m_matImageBuff->last(); ?giapvn: Why -3?

        // Check if buffer is empty or there have no new image is pushed, waiting for 10ms and then go to the next loop.
        if ((processImgItem.getIndex() == -1) ||
            (processImgItem.getIndex() == m_currID)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // Get image from cache memory
        m_currID = processImgItem.getIndex();
        unsigned char *d_imageData = processImgItem.getDeviceImage();
        cv::Size imgSize = processImgItem.getImageSize();
        image_t input;
        input.c = 1;
        input.h = imgSize.height * 3 / 2;   // ?giapvn: Why x1.5?
        input.w = imgSize.width;
        input.data = (float *)d_imageData;
        //TODO: canculate GME
        h_gmeMat = (float *)processImgItem.getHostGMEMatrix();
        d_gmeMat = (float *)processImgItem.getDeviceGMEMatrix();
        cv::Mat trans = cv::Mat(3, 3, CV_32F, h_gmeMat);
        trans.convertTo(trans, CV_64F);

        //TODO: get Detected boxes from VODWorker
        auto detectionCachedItem = m_rbDetectedObjs->getElementById(m_currID);
        std::vector<bbox_t> detection_boxes;
        if (detectionCachedItem.getIndex() != -1 && detectionCachedItem.getIndex() == m_currID)
            detection_boxes = detectionCachedItem.getDetectedObjects();

        // TODO: get gpu_gray_resized from input I420
        cv::cuda::GpuMat gpu_i420_frame(cv::Size(input.w, input.h), CV_8UC1, input.data);
        cv::cuda::GpuMat gpu_gray = gpu_i420_frame(cv::Rect(0, 0, gpu_i420_frame.cols, gpu_i420_frame.rows * 2/3));;
        cv::Size multitrack_size(640, 480);                         // ?giapvn: Why declare multitrack_size for resizing?
        float dx = (float)multitrack_size.width  / gpu_gray.cols;
        float dy = (float)multitrack_size.height / gpu_gray.rows;

        cv::cuda::GpuMat gpu_gray_resized;
        cv::cuda::resize(gpu_gray, gpu_gray_resized, multitrack_size, cv::INTER_NEAREST);

        for(auto &b: detection_boxes)
        {
            b.x *= dx;
            b.y *= dy;
            b.w *= dx;
            b.h *= dy;
        }

        //TODO: Multitracking
        // Restore size
        std::vector<bbox_t> track_result = m_mulTracker->run(gpu_gray_resized, detection_boxes, trans);
        for(auto &b: track_result)
        {
            b.x /= dx;
            b.y /= dy;
            b.w /= dx;
            b.h /= dy;
        }

        //TODO: add track result
        DetectedObjectsCacheItem motCacheItem;
        motCacheItem.setIndex(m_currID);
        motCacheItem.setDetectedObjects(track_result);  /**< Push results to cache memory*/
        m_rbMOTObjs->add(motCacheItem);
        stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> timeSpan = stop - start;
        sleepTime = (long)(33333 - timeSpan.count());
        printf("\nVMOTWorker: process: %d | %d | %d | %d | Box Size = %d | %d", sleepTime,
               m_currID, processImgItem.getIndex(), detectionCachedItem.getIndex(), track_result.size(), detection_boxes.size());
        prevID = m_currID;

        std::this_thread::sleep_for(std::chrono::microseconds(sleepTime));
    }

}

void VMOTWorker::stop()
{
    m_running = false;
}

void VMOTWorker::enableMOT()
{
    m_enOD = true;
    m_cvEnMOT.notify_all();
}

void VMOTWorker::disableMOT()
{
    m_enOD = false;
}

bool VMOTWorker::isRunning()
{
    return m_enOD;
}
