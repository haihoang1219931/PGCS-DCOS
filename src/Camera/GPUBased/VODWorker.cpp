#include "VODWorker.h"
#include "cuda_runtime_api.h"
VODWorker::VODWorker() {}

VODWorker::~VODWorker() {}

void VODWorker::setDetector(Detector *_detector)
{
    m_detector = _detector;
}

void VODWorker::run()
{
    std::chrono::high_resolution_clock::time_point start, stop;
    long sleepTime = 0;
    m_matImageBuff = Cache::instance()->getProcessImageCache();
    m_rbDetectedObjs = Cache::instance()->getDetectedObjectsCache();
    m_rbIPCEO = Cache::instance()->getMotionImageEOCache();
    m_rbIPCIR = Cache::instance()->getMotionImageIRCache();
    ProcessImageCacheItem processImgItem;
    cv::Mat proccImg;
    cv::Size imgSize;
    unsigned char *d_imageData;
    unsigned char *h_imageData;
    while (m_running) {
        std::unique_lock<std::mutex> locker(m_mtx);

        while (!m_enOD) {
            m_cvEnOD.wait_for(locker, std::chrono::seconds(2));
            //            printf("\n===============>Still waitting for Enable Object Detection");
        }

        start = std::chrono::high_resolution_clock::now();
        processImgItem = m_matImageBuff->last();

        if ((processImgItem.getIndex() == -1) ||
            (processImgItem.getIndex() == m_currID)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        m_currID = processImgItem.getIndex();
        // Get I420 Data
        d_imageData = processImgItem.getDeviceImage();
        h_imageData = processImgItem.getHostImage();
        imgSize = processImgItem.getImageSize();

        proccImg = cv::Mat(imgSize.height * 3 / 2, imgSize.width, CV_8UC1, h_imageData);
        cv::Mat grayImage = cv::Mat(imgSize.height, imgSize.width, CV_8UC1, h_imageData);
        //        cv::imwrite("img/i420_" + std::to_string(m_currID) + ".png", proccImg);
        //
        image_t input;
        input.c = 1;
        input.h = imgSize.height * 3 / 2;
        input.w = imgSize.width;
        input.data = (float *)d_imageData;
        std::vector<bbox_t> detection_boxes;

        // drop detection
//        if (m_currID % 2 == 0)
        cudaDeviceSynchronize();
        if(imgSize.width > 0 && imgSize.height >0){
            detection_boxes = m_detector->gpu_detect_I420(input, imgSize.width, imgSize.height, 0.2f, false);
            // TODO: send detection results
            DetectedObjectsCacheItem detectedObjsItem;
            detectedObjsItem.setIndex(m_currID);
            detectedObjsItem.setDetectedObjects(detection_boxes);
            m_rbDetectedObjs->add(detectedObjsItem);
        }
        stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> timeSpan = stop - start;
//        sleepTime = (long)(33333 - timeSpan.count());
        sleepTime = (long)(100000 - timeSpan.count());
//        printf("\nVODWorker process: %d | %d | Box Size = %d", sleepTime, m_currID, detection_boxes.size());
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
