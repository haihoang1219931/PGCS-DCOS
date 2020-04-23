#include "VSearchWorker.h"

VSearchWorker::VSearchWorker() {
    m_plateOCR = new PlateOCR();
}

VSearchWorker::~VSearchWorker() {}

/**
 * Add OCR module to the class.
 */
void VSearchWorker::setOCR(OCR *_OCR)
{
    m_plateOCR->setOCR(_OCR);
}

/**
 * Add LPD module to the class.
 */
void VSearchWorker::setPlateDetector(Detector *_plateDetector)
{
    m_plateOCR->setPlateDetector(_plateDetector);
}

void VSearchWorker::run()
{
    std::chrono::high_resolution_clock::time_point start, stop;
    long sleepTime = 0;
    m_matImageBuff = Cache::instance()->getProcessImageCache();     /**< */
    m_rbMOTObjs = Cache::instance()->getMOTCache();                 /**< */
    m_rbSearchObjs = Cache::instance()->getSearchCache();           /**< */
    m_rbIPCEO = Cache::instance()->getMotionImageEOCache();         /**< */
    m_rbIPCIR = Cache::instance()->getMotionImageIRCache();         /**< */
    ProcessImageCacheItem processImgItem;                           /**< */
    DetectedObjectsCacheItem motCacheItem;                          /**< */
    int prevID = -1;

    cv::Size imgSize;
    unsigned char *d_imageData;
    unsigned char *h_imageData;
    cv::Mat proccImg;               /**< store I420 image */
    cv::Mat grayImage;              /**< store gray image */
    cv::Mat bgrImg;                  /**< store BGR image for classifying plate color */
    std::vector<bbox_t> motBoxs;    /**< */

    while (m_running) {
        std::unique_lock<std::mutex> locker(m_mtx);

        // Check continuously OD mode being enable after every 2s.
        while (!m_enOD) {
            m_cvEnMOT.wait_for(locker, std::chrono::seconds(2));
        }
        // For OD mode is enable
        start = std::chrono::high_resolution_clock::now();
        motCacheItem = m_rbMOTObjs->last();     /**< */

        // If buffer is empty or there have no new frame, go to the next loop
        if ((motCacheItem.getIndex() == -1) ||
            (motCacheItem.getIndex() == m_currID)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // Get data from cache memory
        m_currID = motCacheItem.getIndex();
        motBoxs = motCacheItem.getDetectedObjects();

        processImgItem = m_matImageBuff->getElementById(m_currID);  // Get image by id

        if(processImgItem.getIndex() != m_currID) continue;

        d_imageData = processImgItem.getDeviceImage();
        h_imageData = processImgItem.getHostImage();
        imgSize = processImgItem.getImageSize();
        image_t input;
        input.c = 1;
        input.h = imgSize.height * 3 / 2;
        input.w = imgSize.width;
        input.data = (float *)d_imageData;
        proccImg = cv::Mat(imgSize.height * 3 / 2, imgSize.width, CV_8UC1, h_imageData);
        cv::cvtColor(proccImg, bgrImg, CV_YUV2BGR_I420);
        grayImage = cv::Mat(imgSize.height, imgSize.width, CV_8UC1, h_imageData);

        // TODO: OCR
        m_plateOCR->run(motBoxs, input, grayImage, bgrImg, 1);

        // TODO: search
//        printf("%s motBoxs size = %d\r\n",__func__,motBoxs.size());
        std::vector<std::string> platesForSearch;
        std::vector<std::string> foundPlate;
        for(auto car: motBoxs)
        {
            if ( car.track_info.stringinfo.empty() ) continue;
//			std::cout << "\n   Plate: " << car.track_info.stringinfo << std::endl;
            /*for(auto truth_plate: platesForSearch)
            {
                std::string curPlate = car.track_info.stringinfo;
                if(PlateOCR::get_strings_correlation(curPlate, truth_plate) >= 0.7f)
                {
                    if (std::find(foundPlate.begin(), foundPlate.end(), curPlate) == foundPlate.end())
                    {
                        foundPlate.push_back(curPlate);
                    }
                }
            }*/
            std::string curPlate = car.track_info.stringinfo;
//            std::cout << PlateOCR::get_strings_correlation(curPlate, std::string("30E-57066")) << std::endl;
            if(PlateOCR::get_strings_correlation(curPlate, std::string("30E-57066")) >= 0.4f)
            {
                printf("OK\n");
//                exit(0);
            }

        }

        //TODO: add search result
        DetectedObjectsCacheItem detectedObjsItem;
        detectedObjsItem.setIndex(m_currID);
        detectedObjsItem.setDetectedObjects(motBoxs);
        m_rbSearchObjs->add(detectedObjsItem);
        stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> timeSpan = stop - start;
        sleepTime = (long)(33333 - timeSpan.count());
        printf("\nVSearchWorker: process: %d | %d | %d | Box Size = %d", sleepTime, m_currID, processImgItem.getIndex(), foundPlate.size());
        prevID = m_currID;

        if (sleepTime > 1000) {
            std::this_thread::sleep_for(std::chrono::microseconds(1000));
        }
    }
}

void VSearchWorker::stop()
{
    m_running = false;
}

void VSearchWorker::enableSearch()
{
    m_enOD = true;
    m_cvEnMOT.notify_all();
}

void VSearchWorker::disableSearch()
{
    m_enOD = false;
}

bool VSearchWorker::isRunning()
{
    return m_enOD;
}
