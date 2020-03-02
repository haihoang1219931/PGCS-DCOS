#include "VTrackWorker.h"

VTrackWorker::VTrackWorker()
{
    this->init();
    //    m_stop = false;
}

VTrackWorker::~VTrackWorker() {}

void VTrackWorker::init()
{
    m_multiDetector = new MultilevelDetector();
}

void VTrackWorker::run()
{
    std::chrono::high_resolution_clock::time_point start, stop;
    long sleepTime = 0;
    m_matImageBuff = Cache::instance()->getProcessImageCache();
    m_rbDetectedObjs = Cache::instance()->getDetectedObjectsCache();
    m_rbTrackResEO = Cache::instance()->getEOTrackingCache();
    m_rbTrackResIR = Cache::instance()->getIRTrackingCache();
    m_rbSystem = Cache::instance()->getSystemStatusCache();
    m_rbIPCEO = Cache::instance()->getMotionImageEOCache();
    m_rbIPCIR = Cache::instance()->getMotionImageIRCache();
    ProcessImageCacheItem processImgItem;
    cv::Mat proccImg;
    image_t input;
    cv::Size imgSize;
    unsigned char *d_imageData;
    unsigned char *h_imageData;

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
        Eye::SystemStatus systemStatus = m_rbSystem->getElementById(frameID);
        processImgItem = m_matImageBuff->getElementById(frameID);
        int trackSize = (int)m_rbSystem->last().getIPCStatus().getTrackSize();

        if (processImgItem.getIndex() == frameID) {
            d_imageData = processImgItem.getDeviceImage();
            h_imageData = processImgItem.getHostImage();
            proccImg = cv::Mat(imgSize.height, imgSize.width, CV_8UC4, h_imageData);
            imgSize = processImgItem.getImageSize();
            input.c = 4;
            input.h = imgSize.height;
            input.w = imgSize.width;
            input.data = (float *)d_imageData;
            bbox_t objTrack;
            cv::Point trackPoint(m_trackPoint.getPx() / m_trackPoint.getWidth() * (double)imgSize.width, m_trackPoint.getPy() / m_trackPoint.getHeight() * (double)imgSize.height);
            bool isResultValid = m_multiDetector->detect(input, trackPoint, trackSize, objTrack);

            if (false) {
                printf("\nFound track Obj [%d, %d, %d, %d]", objTrack.x, objTrack.y, objTrack.w, objTrack.h);
                Q_EMIT determinedTrackObjected(
                    frameID, (double)(objTrack.x + objTrack.w / 2) / (double)(imgSize.width / 2),
                    (objTrack.y + objTrack.h / 2) / (double)(imgSize.height / 2), (double)imgSize.width,
                    (double)imgSize.height, objTrack.w / (double)(imgSize.width / 2), objTrack.h / (double)(imgSize.height / 2));
                printf("\n=================> Found object %d - [%f, %f, %f, %f, %f, %f] - [%d, %d]",
                       frameID, (double)(objTrack.x + objTrack.w / 2) / (double)(imgSize.width / 2),
                       (objTrack.y + objTrack.h / 2) / (double)(imgSize.height / 2), (double)imgSize.width,
                       (double)imgSize.height, objTrack.w / (double)(imgSize.width / 2), objTrack.h / (double)(imgSize.height / 2), imgSize.width, imgSize.height);
            } else {
                Q_EMIT determinedTrackObjected(
                    frameID, m_trackPoint.getPx() / (double)(m_trackPoint.getWidth() / 2),
                    m_trackPoint.getPy() / (double)(m_trackPoint.getHeight() / 2),
                    (double)imgSize.width, (double)imgSize.height, 0.0, 0.0);
                printf("\n=================> Default object %d - [%f, %f, %f, %f, %f, %f] - [%d - %d]",
                       frameID, m_trackPoint.getPx() / (double)(m_trackPoint.getWidth() / 2), m_trackPoint.getPy() / (double)(m_trackPoint.getHeight() / 2),
                       (double)imgSize.width, (double)imgSize.height, 0.0, 0.0,  imgSize.width, imgSize.height);
            }
        } else {
            printf("\n************Cannot found clicked frame in cache - %d | %d | %d | %d ******",
                   processImgItem.getIndex(), frameID,
                   m_matImageBuff->last().getIndex(), m_matImageBuff->size());
            continue;
        }

        //        // TODO: wait for tracking object response for OCR
        //        printf("\nwait for tracking object response for OCR");
        TrackResponse trackRes;
        Eye::MotionImage motionImage;
        Eye::MData stabData;

        while (true) {
            start = std::chrono::high_resolution_clock::now();
            processImgItem = m_matImageBuff->last();
            int processID = processImgItem.getIndex();

            if (m_hasNewTrack || m_hasNewMode) {
                m_recognizor.m_stop = false;
                m_recognizor.m_counter = 0;
                m_recognizor.m_codeVector.clear();
                m_recognizor.m_result = "";
            }

            if (processID != 0) {
                trackRes = m_rbTrackResEO->getElementById(processID);
                systemStatus = m_rbSystem->getElementById(processID);
                motionImage = m_rbIPCEO->getElementById(processID);

                if ((trackRes.getIndex() == processID) &&
                    (systemStatus.getIndex() == processID) &&
                    (motionImage.getIndex() == processID)) {
                    d_imageData = processImgItem.getDeviceImage();
                    h_imageData = processImgItem.getHostImage();
                    proccImg = cv::Mat(imgSize.height, imgSize.width, CV_8UC4, h_imageData);
                    imgSize = processImgItem.getImageSize();
                    input.c = 4;
                    input.h = imgSize.height;
                    input.w = imgSize.width;
                    input.data = (float *)d_imageData;
                    //TODO: Detect Plate
                    trackRes.setTrackResponse(trackRes.getIndex(),
                                              trackRes.getPx() * (double)imgSize.width / 2.0, trackRes.getPy() * (double)imgSize.height / 2.0,
                                              (double)imgSize.width, (double)imgSize.height,
                                              trackRes.getObjWidth() * (double)imgSize.width / 2.0, trackRes.getObjHeight() * (double)imgSize.height / 2.0);
                    cv::Rect roi(trackRes.getPx() + trackRes.getWidth() / 2.0f - trackRes.getObjWidth() / 2.f, trackRes.getPy() + trackRes.getHeight() / 2.0f - trackRes.getObjHeight() / 2.f, trackRes.getObjWidth(), trackRes.getObjHeight());
                    std::vector<bbox_t> plateBoxes = m_plateOCR->getPlateBoxes(input, roi);
                    //                    printf("\n===> Rect %d - [%f, %f, %f, %f, %f, %f] - %d", processID, trackRes.getPx(), trackRes.getPy(), trackRes.getObjWidth(), trackRes.getObjHeight(), trackRes.getWidth(), trackRes.getHeight(), plateBoxes.size());
                    std::vector<int> params;
                    params.push_back(cv::IMWRITE_PNG_COMPRESSION);
                    params.push_back(0);
                    if(m_recognizor.m_stop == false)
                    {
                        for(uint i = 0; i < plateBoxes.size(); i++)
                        {
                            cv::Mat tmpPlate(proccImg,
                                             cv::Rect(plateBoxes.at(i).x - (int)(0.1 * plateBoxes[i].w),
                                                      plateBoxes.at(i).y - (int)(0.15 * plateBoxes[i].h),
                                                      (int)(1.2 * plateBoxes.at(i).w), (int)(1.3 * plateBoxes.at(i).h)));
                            if(!tmpPlate.empty())
                            {
//                                auto s1 = std::chrono::high_resolution_clock::now();
                                cv::Mat plate;
                                cv::cvtColor(tmpPlate, plate, CV_BGRA2RGB);
                                cv::cvtColor(plate, plate, CV_RGB2HSV);
                                std::vector<cv::Mat> channels;
                                cv::split(plate, channels);
                                cv::Mat grayPlate = channels[2].clone();
//                                cv::imwrite("/home/pgcs-04/datnt/PGCS-DCOS/build/img/" + std::to_string(m_c) + ".png", grayPlate, params);
//                                m_c++;
                                plate.release();
                                channels.clear();
                                int plateType = plateBoxes[i].obj_id;
                                int sign = -1;
                                std::vector<cv::Mat> chars;
                                if(!grayPlate.empty())
                                    chars = preprocess(grayPlate, plateType, &sign);
                                if(chars.size() > 6)
                                {
                                    std::string code = m_recognizor.recognize(chars, sign);
                                    int cc = 0;
                                    for(uint l = 0; l < code.size(); l++)
                                    {
                                        if(code[l] != '_')
                                            cc++;
                                    }
                                    if(cc > 7)
                                    {
                                        cv::putText(proccImg, code, cv::Point(roi.x, roi.y - 2), CV_FONT_HERSHEY_SIMPLEX, 3.f, cv::Scalar(0, 0, 255), 3);
                                        m_recognizor.combine(code);
                                        m_recognizor.m_counter++;
                                        if(m_recognizor.m_counter > 4)
                                        {
                                            m_recognizor.m_result = m_recognizor.findBest();
                                            m_recognizor.m_stop = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    else
                    {
                        cv::putText(proccImg, m_recognizor.m_result, cv::Point(roi.x, roi.y - 2), CV_FONT_HERSHEY_SIMPLEX, 3.f, cv::Scalar(0, 0, 255), 3);
                    }
                }
            }

            if (m_hasNewTrack || m_hasNewMode) {
                m_trackEn = false;
                m_recognizor.m_stop = false;
                m_recognizor.m_counter = 0;
                m_recognizor.m_codeVector.clear();
                m_recognizor.m_result = "";
                break;
            }

            m_trackEn = true;
            stop = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::micro> timeSpan = stop - start;
            sleepTime = (long)(33333 - timeSpan.count());
            std::this_thread::sleep_for(std::chrono::microseconds(sleepTime));
        }
    }
}

void VTrackWorker::setPlateOCR(PlateOCR *_plateOCR)
{
    m_plateOCR = _plateOCR;
}
void VTrackWorker::stop()
{
    m_running = false;
}
void VTrackWorker::hasNewTrack(int _id, double _px, double _py, double _w,
                               double _h)
{
    m_trackPoint = XPoint(_id, _px, _py, _w, _h);
    m_hasNewTrack = true;
    m_hasNewMode = true;
    m_cvHasNewTrack.notify_all();
}
void VTrackWorker::hasNewMode()
{
    m_hasNewMode = true;
}
bool VTrackWorker::isRunning()
{
    return m_trackEn;
}
int VTrackWorker::findNearestObject(const std::vector<bbox_t> &_resultVec,
                                    const cv::Point &_point,
                                    double &_minDistance)
{
    double minD = 100000000;
    int indexMin = -1;

    for (int i = 0; i < _resultVec.size(); i++) {
        cv::Point center;
        bbox_t curBox = _resultVec[i];
        center.x = curBox.x + curBox.w / 2;
        center.y = curBox.y + curBox.h / 2;
        // find the object which has center closest to pointClick;
        double d = (_point.x - center.x) * (_point.x - center.x);
        d += (_point.y - center.y) * (_point.y - center.y);

        if (d <= minD) {
            minD = d;
            indexMin = i;
        }
    }

    _minDistance = minD;
    return indexMin;
}
