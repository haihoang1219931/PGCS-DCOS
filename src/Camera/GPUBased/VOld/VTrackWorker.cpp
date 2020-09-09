#include "VTrackWorker.h"

VTrackWorker::VTrackWorker()
{
    this->init();
    //    m_stop = false;
    m_tracker = new ITrack(FEATURE_HOG, KERNEL_GAUSSIAN);
    m_clickTrack = new ClickTrack();
}

VTrackWorker::~VTrackWorker()
{
    delete m_tracker;
}

void VTrackWorker::init()
{
}

void VTrackWorker::changeTrackSize(float _trackSize)
{
    m_trackSize = (int)_trackSize;
}

void VTrackWorker::run()
{
    std::chrono::high_resolution_clock::time_point start, stop;
    long sleepTime = 0;
    m_matImageBuff = Cache::instance()->getProcessImageCache();
    m_rbTrackResEO = Cache::instance()->getEOTrackingCache();
    m_rbTrackResIR = Cache::instance()->getIRTrackingCache();
    m_rbSystem = Cache::instance()->getSystemStatusCache();
    m_rbIPCEO = Cache::instance()->getMotionImageEOCache();
    m_rbIPCIR = Cache::instance()->getMotionImageIRCache();
    ProcessImageCacheItem processImgItem;
    cv::Mat proccImg;
    cv::Size imgSize;
    cv::Mat stabMatrix;
    unsigned char *d_imageData;
    unsigned char *h_imageData;
    float *h_stabMat;
    //    float *d_stabMat;
    cv::Mat grayTrackObject;
    image_t input;


    while (m_running) {
        std::unique_lock<std::mutex> locker(m_mtx);

        while (!m_hasNewTrack) {
            m_cvHasNewTrack.wait_for(locker, std::chrono::seconds(2));
            printf("\n=====================>Still waitting for new Track");
        }

        m_hasNewTrack = false;
        m_hasNewMode = false;
        printf("\nHas new track %d - [%f, %f, %f, %f]", m_trackPoint.getIndex(),
               m_trackPoint.getPx(), m_trackPoint.getPy(), m_trackPoint.getWidth(),
               m_trackPoint.getHeight());
        index_type frameID = m_trackPoint.getIndex();

        if (m_matImageBuff == nullptr) continue;

        if (m_matImageBuff->size() == 0) continue;

        if (m_steerEn) {
            frameID = m_matImageBuff->last().getIndex();
        }

        processImgItem = m_matImageBuff->getElementById(frameID);

        if (processImgItem.getIndex() != frameID) {
            processImgItem = m_matImageBuff->last();
        }

        // TODO: Init Tracker
        m_currID = processImgItem.getIndex();
        d_imageData = processImgItem.getDeviceImage();
        h_imageData = processImgItem.getHostImage();
        imgSize = processImgItem.getImageSize();
        stabMatrix = processImgItem.getStabMatrix();
        h_stabMat = processImgItem.getHostStabMatrix();
        proccImg = cv::Mat(imgSize.height, imgSize.width, CV_8UC1, h_imageData);
        // Inverst stab
        int x, y;
        x = m_trackPoint.getPx() / m_trackPoint.getWidth() * imgSize.width;
        y = m_trackPoint.getPy() / m_trackPoint.getHeight() * imgSize.height;

        if (m_enStab) {
            int xStab, yStab;
            int m_cropRatio = 1;
            xStab = x + proccImg.cols * (1 - m_cropRatio) / 2;
            yStab = y + proccImg.rows * (1 - m_cropRatio) / 2;
            cv::Mat stabMatrixInvert;
            cv::invertAffineTransform(stabMatrix, stabMatrixInvert);
            cv::Mat pointBeforeStab(3, 1, CV_64FC1, cv::Scalar::all(0));
            cv::Mat pointInStab(3, 1, CV_64FC1);
            pointInStab.at<double>(0, 0) = (double)xStab;
            pointInStab.at<double>(1, 0) = (double)yStab;
            pointInStab.at<double>(2, 0) = 1;
            pointBeforeStab = stabMatrixInvert * pointInStab;
            x = (int)pointBeforeStab.at<double>(0, 0);
            y = (int)pointBeforeStab.at<double>(1, 0);
        }

        cv::Rect roiOpen;
        // Check roi size for track to avoid coredump fault
        int trackSize = m_trackSize;

        if (m_steerEn) {
            trackSize = 400;
        }

        if (proccImg.rows >= trackSize && proccImg.cols >= trackSize) {
            roiOpen.x       = ((x - trackSize / 2) < 0) ? 0 : (x - trackSize / 2);
            roiOpen.y       = ((y - trackSize / 2) < 0) ? 0 : (y - trackSize / 2);
            roiOpen.width   = ((roiOpen.x + trackSize) >= proccImg.cols) ? (proccImg.cols - roiOpen.x) : (trackSize);
            roiOpen.height  = ((roiOpen.y + trackSize) >= proccImg.rows) ? (proccImg.rows - roiOpen.y) : (trackSize);

            if (roiOpen.x > 0 && roiOpen.y > 0 &&
                roiOpen.x + roiOpen.width <= proccImg.cols &&
                roiOpen.y + roiOpen.height <= proccImg.rows) {
                m_tracker->initTrack(proccImg, roiOpen);
            } else {
                continue;
            }
        } else {
            continue;
        }

        while (true) {
            start = std::chrono::high_resolution_clock::now();
            // DO TRACK
            processImgItem = m_matImageBuff->last();

            if ((processImgItem.getIndex() == -1) ||
                (processImgItem.getIndex() == m_currID)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            m_currID = processImgItem.getIndex();
            h_imageData = processImgItem.getHostImage();
            d_imageData = processImgItem.getDeviceImage();
            imgSize = processImgItem.getImageSize();
            proccImg = cv::Mat(imgSize.height, imgSize.width, CV_8UC1, h_imageData);

            input.c = 1;
            input.h = imgSize.height * 3 / 2;
            input.w = imgSize.width;

            if (m_tracker->isInitialized()) {
                m_tracker->performTrack(proccImg);

                if (m_tracker->trackStatus() == TRACK_LOST) {
                    printf("Track lost\r\n");
                    Q_EMIT objectLost();
                    m_tracker->resetTrack();
                    m_trackEn = false;
                    break;
                } else {
                    cv::Rect objectPosition = m_tracker->getPosition();

                    if (m_steerEn) {
                        this->drawSteeringCenter(proccImg, 400, objectPosition.x + objectPosition.width / 2.0,
                                                 objectPosition.y + objectPosition.height / 2.0, cv::Scalar(255, 0, 0));
                    } else {
                        this->drawObjectBoundary(
                            proccImg, objectPosition, cv::Scalar(255, 255, 255));
                    }

                    //                    printf("\n====> DO TRACK [%d, %d, %d, %d]", objectPosition.x, objectPosition.y, objectPosition.width, objectPosition.height);
                    Q_EMIT determinedTrackObjected(m_currID, objectPosition.x, objectPosition.y,
                                                   objectPosition.width, objectPosition.height,
                                                   imgSize.width, imgSize.height);
                    // TODO : Detect Plate
//                    // check if objectPosition is inside proccImg
                    if (    (objectPosition.x > 0)
                         && (objectPosition.y > 0)
                         && ((objectPosition.x + objectPosition.width ) < proccImg.cols)
                         && ((objectPosition.y + objectPosition.height) < proccImg.rows))
                    {
                        printf("\n===========how to get here");
                        grayTrackObject = proccImg(objectPosition).clone();
//                        cv::imwrite("img/grayTrackObject" + std::to_string(m_currID) + ".png", grayTrackObject);
                        std::string plateNumber = "";
                        input.data = (float *)d_imageData;
                        int status = m_clickTrack->updateNewImage_I420(input, proccImg, objectPosition);
                        if (status == PLATE_SUCCESS){
                            plateNumber = m_clickTrack->getPlateNumber_I420();
                            printf("\n====>Plate Recognize: %s\n", plateNumber.c_str());
                        }else{
                            printf("\n====>Cannot Recognize Plate\n");
                        }
                    }

                }
            } else {
                m_tracker->resetTrack();
                m_trackEn = false;
                break;
            }

            if (m_hasNewMode || m_hasNewTrack || !m_running) {
                m_tracker->resetTrack();
                m_trackEn = false;
                break;
            }

            stop = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::micro> timeSpan = stop - start;
            sleepTime = (long)(33333 - timeSpan.count());
            std::this_thread::sleep_for(std::chrono::microseconds(sleepTime));
        }
    }
}

void VTrackWorker::stop()
{
    m_running = false;
}
void VTrackWorker::hasNewTrack(int _id, double _px, double _py, double _w,
                               double _h, bool _enSteer, bool _enStab)
{
    m_trackPoint = XPoint(_id, _px, _py, _w, _h);
    m_hasNewTrack = true;
    m_hasNewMode = true;
    m_steerEn = _enSteer;
    m_trackEn = !_enSteer;
    m_enStab = _enStab;
    m_cvHasNewTrack.notify_all();
}

void VTrackWorker::hasNewMode()
{
    m_hasNewMode = true;
}

void VTrackWorker::disSteer()
{
    m_steerEn = false;
}
bool VTrackWorker::isAcive()
{
    return m_trackEn;
}


void VTrackWorker::drawObjectBoundary(cv::Mat &_img, cv::Rect _objBoundary,
                                      cv::Scalar _color)
{
    cv::rectangle(_img, _objBoundary, _color, 2);
}

void VTrackWorker::drawSteeringCenter(cv::Mat &_img, int _wBoundary,
                                      int _centerX, int _centerY,
                                      cv::Scalar _color)
{
    _centerX -= _wBoundary / 2;
    _centerY -= _wBoundary / 2;
    cv::line(_img, cv::Point(_centerX, _centerY),
             cv::Point(_centerX + _wBoundary / 4, _centerY), _color, 2);
    cv::line(_img, cv::Point(_centerX, _centerY),
             cv::Point(_centerX, _centerY + _wBoundary / 4), _color, 2);
    cv::line(_img, cv::Point(_centerX + _wBoundary, _centerY),
             cv::Point(_centerX + 3 * _wBoundary / 4, _centerY), _color, 2);
    cv::line(_img, cv::Point(_centerX + _wBoundary, _centerY),
             cv::Point(_centerX + _wBoundary, _centerY + _wBoundary / 4), _color,
             2);
    cv::line(_img, cv::Point(_centerX, _centerY + _wBoundary),
             cv::Point(_centerX, _centerY + 3 * _wBoundary / 4), _color, 2);
    cv::line(_img, cv::Point(_centerX, _centerY + _wBoundary),
             cv::Point(_centerX + _wBoundary / 4, _centerY + _wBoundary), _color,
             2);
    cv::line(_img, cv::Point(_centerX + _wBoundary, _centerY + _wBoundary),
             cv::Point(_centerX + _wBoundary, _centerY + 3 * _wBoundary / 4),
             _color, 2);
    cv::line(_img, cv::Point(_centerX + _wBoundary, _centerY + _wBoundary),
             cv::Point(_centerX + 3 * _wBoundary / 4, _centerY + _wBoundary),
             _color, 2);
}

void VTrackWorker::setClicktrackDetector(Detector *_detector)
{
    m_clickTrack->setDetector(_detector);
}