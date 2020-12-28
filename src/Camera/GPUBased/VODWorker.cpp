#include "VODWorker.h"

#include "cuda_runtime_api.h"
VODWorker::VODWorker() {}

VODWorker::~VODWorker() {}

/**
 * @brief VODWorker::setDetector
 * @param _detector
 */
void VODWorker::setObjectClassifier(Detector *_detector)
{
    m_objectClassifier = _detector;
}
void VODWorker::setPlateIdentifier(PlateDetector* plateDetector)
{
    m_plateDetector = plateDetector;
}

void VODWorker::run()
{
    std::chrono::high_resolution_clock::time_point start, stop;
    long sleepTime = 0;
    m_matImageBuff = Cache::instance()->getProcessImageCache();         /**< */
    m_rbDetectedObjs = Cache::instance()->getDetectedObjectsCache();    /**< */
//    m_rbIPCEO = Cache::instance()->getMotionImageEOCache();             /**< */
//    m_rbIPCIR = Cache::instance()->getMotionImageIRCache();             /**< */

    cv::Mat proccImg;
    cv::Size imgSize;
    unsigned char *d_imageData;
    unsigned char *h_imageData;
    while (m_running) {
        std::unique_lock<std::mutex> locker(m_mtx);

        // Check for OD mode being enable every 2s.
        while (!m_enOD) {
            m_cvEnOD.wait_for(locker, std::chrono::seconds(2));
        }
        //For the OD mode is enable
        start = std::chrono::high_resolution_clock::now();
        ProcessImageCacheItem& processImgItem = m_matImageBuff->last();

        // Check if buffer is empty or there have no new image goto buffer, the thread sleep for 10ms and then go to the next loop
        if ((processImgItem.getIndex() == -1) ||
            (processImgItem.getIndex() == m_currID)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        m_currID = processImgItem.getIndex();   // update the new image ID
        // Get I420 Data
        d_imageData = processImgItem.getDeviceImage();
        h_imageData = processImgItem.getHostImage();
        imgSize = processImgItem.getImageSize();
        proccImg = cv::Mat(imgSize.height * 3 / 2, imgSize.width, CV_8UC1, h_imageData);

        image_t input;
        input.c = 1;
        input.h = imgSize.height * 3 / 2;
        input.w = imgSize.width;
        input.data = (float *)d_imageData;
        std::vector<bbox_t> detection_boxes;

        // Do detect objects on I420 image on gpu.s
        cudaDeviceSynchronize();
        detection_boxes = m_objectClassifier->gpu_detect_I420(input, imgSize.width, imgSize.height, 0.2f, false);
        printf("Detect [%d] objects\r\n",detection_boxes.size());
        // TODO: send detection results to cache memory
        DetectedObjectsCacheItem detectedObjsItem;
        detectedObjsItem.setIndex(m_currID);
        detectedObjsItem.setDetectedObjects(detection_boxes);
        m_rbDetectedObjs->add(detectedObjsItem);
        stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> timeSpan = stop - start;
//        sleepTime = (long)(100000 - timeSpan.count());
        sleepTime = 30000;
        std::this_thread::sleep_for(std::chrono::microseconds(sleepTime));
    }
}

void VODWorker::stop()
{
    m_running = false;
}

void VODWorker::enableOD()
{
    m_enOD = true;
    m_cvEnOD.notify_all();
}

void VODWorker::disableOD()
{
    m_enOD = false;
}

bool VODWorker::isActive()
{
    return m_enOD;
}
