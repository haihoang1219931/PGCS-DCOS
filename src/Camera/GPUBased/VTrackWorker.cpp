#include "VTrackWorker.h"
// plate detection
#include "Clicktrack/clicktrack.h"
#include "OD/yolo_v2_class.hpp"
#include "src/Camera/GimbalController/GimbalInterface.h"
VTrackWorker::VTrackWorker()
{
    this->init();
    //    m_stop = false;
#ifdef TRACK_DANDO
    m_tracker = new ITrack(FEATURE_HOG, KERNEL_GAUSSIAN);
#else
    m_tracker = new Tracker();
#endif
    m_clickTrack = new ClickTrack();
    m_currID = -1;
    m_mutexCommand = new QMutex();
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
    m_trackSizePrev = m_trackSize;
}
void VTrackWorker::setClick(float x, float y,float width,float height){
    if(m_grayFrame.cols <= 0 || m_grayFrame.rows <= 0
            || fabs(width) < 20 || fabs(height) < 20){
        return;
    }
    if(!m_clickSet){
        m_clickSet = true;
        m_clickPoint.x = x/width*static_cast<float>(m_grayFrame.cols);
        m_clickPoint.y = y/height*static_cast<float>(m_grayFrame.rows);
        printf("%s at (%f,%f) / (%f,%f) = (%f,%f)\r\n",__func__,
               x,y,width,height,
               m_clickPoint.x,m_clickPoint.y);
    }
}
void VTrackWorker::moveImage(float panRate,float tiltRate,float zoomRate, float alpha){    
    if(m_grayFrame.cols <= 0 || m_grayFrame.rows <= 0){
        return;
    }
    m_moveZoomRate = -zoomRate / maxAxis;
    //    m_rotationAlpha = alpha;
    m_movePanRate = panRate / maxAxis * (m_grayFrame.cols/2) / 10;
    m_moveTiltRate = tiltRate / maxAxis * (m_grayFrame.rows/2) / 10;
    joystickData temp;

    temp.frameID = m_frameID;
    temp.panRate = m_movePanRate;
    temp.tiltRate = m_moveTiltRate;
    temp.zoomRate = m_moveZoomRate;
    //    printf("panRate,tiltRate = %f,%f\r\n",panRate,tiltRate);
    m_mutexCommand->lock();
    if(fabs(panRate) < deadZone &&
            fabs(tiltRate) < deadZone &&
            fabs(zoomRate) < deadZone){
        temp.panRate = 0;
        temp.tiltRate = 0;
        temp.zoomRate = 0;
        m_jsQueue.clear();
        m_zoomDir = (m_r - m_zoomStart)/fabs(m_r - m_zoomStart);
        Q_EMIT zoomTargetChangeStopped(m_r);
        startRollbackZoom();
        m_zoomStart = m_gimbal->context()->m_zoom[m_gimbal->context()->m_sensorID];
    }else{
        m_jsQueue.push_back(temp);
        while(m_jsQueue.size() > 5){
            m_jsQueue.pop_front();
        }

    }
    m_mutexCommand->unlock();
    //    if(m_trackEnable){
    //        setTrack(static_cast<int>(m_movePanRate/m_digitalZoomMax)+m_img.cols/2,
    //                 static_cast<int>(m_moveTiltRate/m_digitalZoomMax)+m_img.rows/2);
    //    }
}
cv::Mat VTrackWorker::createPtzMatrix(float w, float h, float dx, float dy,float r,float alpha){
    cv::Mat ptzMatrix = cv::Mat(3, 3, CV_64FC1,cv::Scalar::all(0));
    ptzMatrix.at<double>(0, 0) = static_cast<double>(r*cos(alpha));
    ptzMatrix.at<double>(0, 1) = static_cast<double>(-r*sin(alpha));
    ptzMatrix.at<double>(0, 2) = static_cast<double>(-r*dx*cos(alpha) + r*dy*sin(alpha) + w/2);
    ptzMatrix.at<double>(1, 0) = static_cast<double>(r*sin(alpha));
    ptzMatrix.at<double>(1, 1) = static_cast<double>(r*cos(alpha));
    ptzMatrix.at<double>(1, 2) = static_cast<double>(-r*dx*sin(alpha)-r*dy*cos(alpha)+h/2);
    ptzMatrix.at<double>(2, 0) = 0;
    ptzMatrix.at<double>(2, 1) = 0;
    ptzMatrix.at<double>(2, 2) = 1;
    return ptzMatrix;
}
void VTrackWorker::createRoiKeypoints(
        cv::Mat &grayImg,cv::Mat &imgResult,vector<cv::Point2f>& listPoints,
        KEYPOINT_TYPE type,int pointPerDimension,int dimensionSize,
        int dx, int dy){
    listPoints.clear();
    cv::Rect rectBound = cv::Rect(dx-dimensionSize/2,
                                  dy-dimensionSize/2,
                                  dimensionSize,dimensionSize);
    switch (static_cast<int>(type)) {
    case static_cast<int>(KEYPOINT_TYPE::CONTOURS):
    {
        cv::Mat binaryFrame;
        cv::adaptiveThreshold(grayImg,binaryFrame,255,cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 12);
        vector<vector<cv::Point>> contours;
        vector<cv::Vec4i> hierarchy;
        cv::RNG rng(12345);
        findContours( binaryFrame(rectBound), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );
        for( std::size_t i = 0; i< contours.size(); i++ )
        {
            cv::Scalar color = cv::Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
            drawContours( imgResult, contours, static_cast<int>(i), color, 5, 8, hierarchy, 0, cv::Point() );
            for( std::size_t j = 0; j< contours[i].size(); j++ ){
                cv::Point2f point(rectBound.x + contours[i][j].x,
                                  rectBound.y + contours[i][j].y);
                listPoints.push_back(
                            point);
            }
        }
    }
        break;
    case static_cast<int>(KEYPOINT_TYPE::GOOD_FEATURES):
    {
        cv::goodFeaturesToTrack(grayImg(rectBound),listPoints,200, 0.01, 30);
        for( std::size_t i = 0; i< listPoints.size(); i++ ){
            listPoints[i] = cv::Point2f(rectBound.x + listPoints[i].x,
                                        rectBound.y + listPoints[i].y);
            cv::circle(imgResult,listPoints[i],5,cv::Scalar(100,100,0),5);
        }
    }
        break;
    default:
        for(int row = 0; row < pointPerDimension; row++){
            for(int col =0; col < pointPerDimension; col ++){
                cv::Point2f point(rectBound.x + rectBound.width/(pointPerDimension-1)*col,
                                  rectBound.y + rectBound.height/(pointPerDimension-1)*row);
                listPoints.push_back(
                            point);
                cv::circle(imgResult,point,5,cv::Scalar(255,100,0),5);
            }
        }
    }
}
void VTrackWorker::run()
{
    std::chrono::high_resolution_clock::time_point start, stop;
    long sleepTime = 0;
    m_matImageBuff = Cache::instance()->getProcessImageCache();
    m_matTrackBuff = Cache::instance()->getTrackImageCache();
    //    m_rbTrackResEO = Cache::instance()->getEOTrackingCache();
    //    m_rbTrackResIR = Cache::instance()->getIRTrackingCache();
    //    m_rbSystem = Cache::instance()->getSystemStatusCache();
    //    m_rbIPCEO = Cache::instance()->getMotionImageEOCache();
    //    m_rbIPCIR = Cache::instance()->getMotionImageIRCache();
    ProcessImageCacheItem processImgItem;

    cv::Size imgSize;
    unsigned char *h_i420Image;
    unsigned char *d_i420Image;
    float *h_stabMat;
    float *d_stabMat;
    float *h_gmeMat;
    float *d_gmeMat;
    for(unsigned int i=0; i< 10; i++){
        m_zoomRateCalculate[i] = 1;
    }
    while (m_running) {
        std::unique_lock<std::mutex> locker(m_mtx);

        processImgItem = m_matImageBuff->last();

        if ((processImgItem.getIndex() == -1) ||
                (processImgItem.getIndex() == m_currID)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        start = std::chrono::high_resolution_clock::now();
        if(m_r < m_digitalZoomMin) {
            m_r = m_digitalZoomMin;
            Q_EMIT zoomTargetChanged(m_r);
        }
        if(m_r > m_digitalZoomMax) {
            m_r = m_digitalZoomMax;
            Q_EMIT zoomTargetChanged(m_r);
        }
        m_currID = processImgItem.getIndex();
        d_i420Image = processImgItem.getDeviceImage();
        h_i420Image = processImgItem.getHostImage();
        imgSize = processImgItem.getImageSize();
        d_stabMat = processImgItem.getDeviceStabMatrix();
        h_gmeMat = processImgItem.getDeviceGMEMatrix();
        d_gmeMat = processImgItem.getHostGMEMatrix();

        m_i420Img = cv::Mat(imgSize.height * 3 / 2, imgSize.width, CV_8UC1, h_i420Image);
        if(m_grayFrame.cols > 0 && m_grayFrame.rows > 0){
            m_grayFramePrev = m_grayFrame.clone();
        }
        if(m_i420Img.cols <= 0 || m_i420Img.rows <= 0){
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        cv::cvtColor(m_i420Img, m_grayFrame, CV_YUV2GRAY_YV12);
        //TODO: Perfrom tracking
        float w = static_cast<float>(m_grayFrame.cols);
        float h = static_cast<float>(m_grayFrame.rows);
        if (m_dx < 0) m_dx = w/2;
        if (m_dy < 0) m_dy = h/2;
        // handle command
        //        printf("m_jsQueue.size() = %d\r\n",m_jsQueue.size());
        if(m_gimbal->context()->m_lockMode == "FREE"){
            m_trackEnable = false;
        }else if(m_gimbal->context()->m_lockMode == "TRACK" ||
                 m_gimbal->context()->m_lockMode == "VISUAL"){
            m_trackEnable = true;
            if(m_gimbal->context()->m_lockMode == "VISUAL"){
                if( h/2 > m_trackSize)
                    m_trackSize = h/2;
            }else{
                if(m_trackSizePrev != m_trackSize){
                    m_trackSize = m_trackSizePrev;
                }
            }
        }
        if(m_clickSet || m_jsQueue.size()>0){
            if(m_clickSet){
                m_clickSet=false;
                cv::Mat ptzMatrixInvert;
                cv::invert(m_ptzMatrix,ptzMatrixInvert);
                cv::Mat pointBeforeStab(3, 1, CV_64FC1,cv::Scalar::all(0));
                cv::Mat pointInStab(3, 1, CV_64FC1);
                pointInStab.at<double>(0,0) = static_cast<double>(m_clickPoint.x);
                pointInStab.at<double>(1,0) = static_cast<double>(m_clickPoint.y);
                pointInStab.at<double>(2,0) = 1;
                pointBeforeStab = ptzMatrixInvert*pointInStab;
                if(!m_stabEnable){
                    pointBeforeStab = pointInStab.clone();
                }

                cv::Point lockPoint(static_cast<int>(pointBeforeStab.at<double>(0,0)),
                                    static_cast<int>(pointBeforeStab.at<double>(1,0)));
                if(m_trackEnable){
                    cv::Rect trackRectTmp(lockPoint.x-m_trackSize/2,
                                          lockPoint.y-m_trackSize/2,
                                          m_trackSize,m_trackSize);
                    if(trackRectTmp.x > 0 && trackRectTmp.x + trackRectTmp.width < w &&
                            trackRectTmp.y > 0 && trackRectTmp.y + trackRectTmp.height < h){
                        if(m_tracker->isInitialized()){
                            m_tracker->resetTrack();
                        }
                        m_tracker->initTrack(m_grayFramePrev,trackRectTmp);
                    }
                }else{
                    m_dx = lockPoint.x;
                    m_dy = lockPoint.y;
                }
            }else if(m_jsQueue.size()>0){
                joystickData temp = m_jsQueue.front();
                temp.panRate = m_movePanRate;
                temp.tiltRate = m_moveTiltRate;
                if(fabs(m_moveZoomRate) >= deadZone/maxAxis){
                    //                    m_r+=m_moveZoomRate /3;
                    //                    if(m_r > m_digitalZoomMax){
                    //                        m_r = m_digitalZoomMax;
                    //                    }else if(m_r < m_digitalZoomMin){
                    //                        m_r = m_digitalZoomMin;
                    //                    }
                    Q_EMIT zoomTargetChanged(m_r);
                    Q_EMIT zoomTargetChangeStopped(m_r);
                }
                cv::Point lockPoint(static_cast<int>(m_dx +
                                                     (temp.panRate * cos(m_rotationAlpha) + temp.tiltRate * sin(m_rotationAlpha))/m_r),
                                    static_cast<int>(m_dy +
                                                     (-temp.panRate * sin(m_rotationAlpha) + temp.tiltRate * cos(m_rotationAlpha))/m_r)
                                    );
                if(m_trackEnable){
                    int trackSizeTmp = m_trackSize;
                    //                    if(trackSizeTmp > 200){
                    //                        trackSizeTmp = 200;
                    //                    }
                    cv::Rect trackRectTmp(lockPoint.x-trackSizeTmp/2,
                                          lockPoint.y-trackSizeTmp/2,
                                          trackSizeTmp,trackSizeTmp);
                    int deadSpace = 10;
                    if(trackRectTmp.x < deadSpace){
                        trackRectTmp.x = deadSpace;
                    }else if(trackRectTmp.x + trackRectTmp.width > w - deadSpace){
                        trackRectTmp.x = w - deadSpace - trackRectTmp.width;
                    }

                    if(trackRectTmp.y < deadSpace){
                        trackRectTmp.y = deadSpace;
                    }else if(trackRectTmp.y + trackRectTmp.height > h - deadSpace){
                        trackRectTmp.y = h - deadSpace - trackRectTmp.height;
                    }

                    if(m_tracker->isInitialized()){
                        m_tracker->resetTrack();
                    }
                    m_tracker->initTrack(m_grayFramePrev,trackRectTmp);
                }
                else{
                    m_dx = lockPoint.x;
                    m_dy = lockPoint.y;
                }
            }

        }
        if(m_trackEnable){
            if(m_tracker->isInitialized()){
                //                printf("Before performTrack\r\n");
                m_tracker->performTrack(m_grayFrame);
                //                printf("After performTrack\r\n");
                cv::Rect trackRect = m_tracker->getPosition();
                m_dx = trackRect.x+trackRect.width/2;
                m_dy = trackRect.y+trackRect.height/2;
                m_trackRect.x = static_cast<int>(
                            static_cast<float>(m_dx) - static_cast<float>(m_trackSize)/2);
                m_trackRect.y = static_cast<int>(
                            static_cast<float>(m_dy) - static_cast<float>(m_trackSize)/2);
                m_trackRect.width = m_trackSize;
                m_trackRect.height = m_trackSize;
                if(m_tracker->Get_State() == TRACK_INVISION || m_tracker->Get_State() == TRACK_OCCLUDED){
                    Q_EMIT trackStateFound(0,
                                           static_cast<double>(m_trackRect.x),
                                           static_cast<double>(m_trackRect.y),
                                           static_cast<double>(m_trackRect.width),
                                           static_cast<double>(m_trackRect.height),
                                           static_cast<double>(w),
                                           static_cast<double>(h));
                    printf("%s _px=%f _py=%f _oW=%f _oH=%f _w=%f _h=%f\r\n",
                           __func__,
                           static_cast<double>(m_trackRect.x),
                           static_cast<double>(m_trackRect.y),
                           static_cast<double>(m_trackRect.width),
                           static_cast<double>(m_trackRect.height),
                           static_cast<double>(w),
                           static_cast<double>(h));
                }else{
                    Q_EMIT objectLost();
                }
            }else{

            }
        }else{
            m_trackRect.x = m_dx - m_trackSize/2;
            m_trackRect.y = m_dy - m_trackSize/2;
            m_trackRect.width = m_trackSize;
            m_trackRect.height = m_trackSize;
        }

        //        for(int i=0; i< 1;i++){
        //            cv::Mat imgDebug = m_grayFrame.clone();
        //            cv::Mat imgDebugWarp;
        //            if(fabs(m_moveZoomRate) < deadZone/maxAxis && !m_stopRollBack[0]){
        //                if(m_grayFramePrev.cols > 0 && m_grayFramePrev.rows > 0){
        //                    vector<cv::Point2f> pointsPrevOF,pointsCurrentOF;
        //                    // found zoom changed here

        //                    // create set of keypoints from center
        //                    createRoiKeypoints(m_grayFrame,imgDebug,pointsCurrentOF,static_cast<KEYPOINT_TYPE>(i),2,200,m_dx,m_dy);
        //                    vector<uchar> status;
        //                    vector<float> err;
        //                    cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);
        //                    calcOpticalFlowPyrLK(m_grayFrame,m_grayFramePrev, pointsCurrentOF, pointsPrevOF, status, err, cv::Size(15,15), 2, criteria);
        //                    cv::Mat rigidT = cv::estimateRigidTransform(pointsPrevOF, pointsCurrentOF, true);
        //                    //#ifdef DEBUG
        //                    //                std::cout << "rigid " << rigidT << std::endl;
        //                    //#endif
        //                    if(rigidT.rows >= 2 && rigidT.cols >=2){
        //                        // compute scale and rotaion between start keypoints and current keypoints
        //                        float zoomRateCalculatePrev = m_zoomRateCalculate[i];
        //                        float zoomRateCalculateTemp = static_cast<float>(pow((
        //                                                                                 pow(rigidT.at<double>(0,0),2) +pow(rigidT.at<double>(0,1),2)
        //                                                                                 ),0.5
        //                                                                             ));
        //                        //                        printf("zoomRateCalculateTemp = %f zoomRateCalculatePrev=%f\r\n",
        //                        //                               zoomRateCalculateTemp,zoomRateCalculatePrev);
        //                        m_zoomRateCalculate[i] *= zoomRateCalculateTemp;
        //                        Q_EMIT zoomCalculateChanged(i+1,m_zoomRateCalculate[i]*m_zoomStart);
        //                    }
        //                    printf("process[%d] m_r=%f m_zoomRateCalculate[%d]=%f m_zoomStart=%f pointsPrevOF.size=%d\r\n",
        //                           i,m_r,i,m_zoomRateCalculate[i],m_zoomStart,pointsPrevOF.size());
        //                }
        //                if(m_zoomDir * (m_zoomRateCalculate[i]*m_zoomStart - m_r) > 0){
        //                    m_stopRollBack[i] = true;
        //                    m_zoomRateCalculate[i] = 1;
        //                    m_zoomStart = m_r;
        //                    Q_EMIT zoomCalculateChanged(i+1,m_zoomRateCalculate[i]*m_zoomStart);
        //                }
        //            }
        //        }
        //        m_ptzMatrix = createPtzMatrix(w,h,m_dx,m_dy,m_r/m_zoomRateCalculate[0]*m_zoomStart,0);
        if(m_gimbal->context()->m_sensorID == 0){
            m_ptzMatrix = createPtzMatrix(w,h,m_dx,m_dy,1,m_rotationAlpha);
        }else{
            m_ptzMatrix = createPtzMatrix(w,h,m_dx,m_dy,m_r,m_rotationAlpha);
        }
        //        std::cout << "hainh create m_ptzMatrix " << m_ptzMatrix << std::endl;
        // add data to display worker
        ProcessImageCacheItem processImgItem;
        processImgItem.setIndex(m_currID);
        processImgItem.setHostImage(h_i420Image);
        processImgItem.setDeviceImage(d_i420Image);
        processImgItem.setImageSize(imgSize);
        processImgItem.setHostStabMatrix(m_ptzMatrix);
        processImgItem.setDeviceStabMatrix(d_stabMat);
        processImgItem.setHostGMEMatrix(h_gmeMat);
        processImgItem.setDeviceGMEMatrix(d_gmeMat);
        processImgItem.setLockMode(m_gimbal->context()->m_lockMode);
        processImgItem.setTrackRect(m_trackRect);
        processImgItem.setTrackStatus(m_tracker->Get_State());
        processImgItem.setZoom(m_gimbal->context()->m_zoom[m_gimbal->context()->m_sensorID]);
        m_matTrackBuff->add(processImgItem);

        stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> timeSpan = stop - start;
        sleepTime = (long)(33333 - timeSpan.count());
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
        //printf("VTrackWorker: %d - [%d, %d] \r\n", m_currID, imgSize.width, imgSize.height);
        //std::cout << "timeSpan: " << timeSpan.count() <<std::endl;
    }
}

void VTrackWorker::stop()
{
    m_running = false;
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

void VTrackWorker::setOCR(OCR* _OCR)
{
    m_clickTrack->setOCR(_OCR);
}
