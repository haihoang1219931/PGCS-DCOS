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
    m_mutex = new QMutex();
    m_pauseCond = new QWaitCondition();
    m_mapColorMode["WHITE_HOT"] = 0;
    m_mapColorMode["COLOR"] = 1;
}

VTrackWorker::~VTrackWorker()
{
    delete m_tracker;
}

void VTrackWorker::init()
{
#ifdef _test_ORBSearcher_
    process.init();
#endif
}

void VTrackWorker::changeTrackSize(float _trackSize)
{
    m_trackSize = static_cast<int>(_trackSize/8)*8;
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
void VTrackWorker::setPowerLineDetect(bool enable){
#ifdef USE_LINE_DETECTOR
    m_powerLineDetectEnable = enable;
    if(enable){
        if(m_grayFrame.cols <= 0 || m_grayFrame.rows <= 0){
            return;
        }
        m_powerLineDetectRect.x = 50;
        m_powerLineDetectRect.y = m_grayFrame.rows / 2;
        m_powerLineDetectRect.width = m_grayFrame.cols - 100;
        m_powerLineDetectRect.height = m_grayFrame.rows / 2 - 50;
    }
#endif
}
void VTrackWorker::setPowerLineDetectRect(QRect rect){
#ifdef USE_LINE_DETECTOR
    m_powerLineDetectRect.x = rect.x();
    m_powerLineDetectRect.y = rect.y();
    m_powerLineDetectRect.width = rect.width();
    m_powerLineDetectRect.height = rect.height();
#endif
}
void VTrackWorker::setSensorColor(QString colorMode){
    m_colorMode = m_mapColorMode[colorMode];
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
    //    printf("panRate,tiltRate,zoomRate = %f,%f,%f\r\n",
    //           panRate,tiltRate,zoomRate);
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
void VTrackWorker::pause(bool _pause){
    if(_pause == true){
        m_mutex->lock();
        m_pause = true;
        m_mutex->unlock();
    }else{
        m_mutex->lock();
        m_pause = false;
        m_mutex->unlock();
        m_pauseCond->wakeAll();
    }
}
void VTrackWorker::run()
{
#ifdef DEBUG_TIMER
    clock_t start, stop;
    clock_t startFrame;
#endif
    long sleepTime = 0;
    m_matImageBuff = Cache::instance()->getProcessImageCache();
    m_matTrackBuff = Cache::instance()->getTrackImageCache();

#ifdef _test_ORBSearcher_
    cv::Mat bgrImg;
    cv::Mat h_i420Img;
    bool f_update = false;
    cv::Rect dbg_center_img_roi;
    std::vector<cv::Rect> init_set;
#endif
    cv::Size imgSize;
    unsigned char *h_i420Image;
    unsigned char *d_i420Image;
    float *d_stabMat;
    float *h_gmeMat;
    float *d_gmeMat;
    for(unsigned int i=0; i< 10; i++){
        m_zoomRateCalculate[i] = 1;
    }
    while (m_running) {
#ifdef DEBUG_TIMER
        clock_t lastFrame = startFrame;
        startFrame = clock();

        {
            clock_t timeSpan = startFrame - lastFrame;
            std::cout << "Total run ["<<m_currID <<"] ["<<((double)timeSpan)/CLOCKS_PER_SEC * 1000<< "]" << std::endl;
        }
#endif
        m_mutex->lock();
        if(m_pause)
            m_pauseCond->wait(m_mutex); // in this place, your thread will stop to execute until someone calls resume
        m_mutex->unlock();
#ifdef DEBUG_TIMER
        start = clock();
#endif
        ProcessImageCacheItem& processImgItem = m_matImageBuff->last();
#ifdef DEBUG_TIMER
        stop = clock();
        {
            clock_t timeSpan = stop - start;
            std::cout << "Track Get last frame ["<< m_currID <<"/" << processImgItem.getIndex()<<"] ["<<((double)timeSpan)/CLOCKS_PER_SEC * 1000<< "]" << std::endl;
        }
#endif
#ifdef DEBUG_TIMER
        start = clock();
#endif
//        if(processImgItem.getIndex() == -1 ||
//                processImgItem.getIndex() <= m_currID){
//            msleep(10);
//            processImgItem = m_matImageBuff->last();
//        }
//        if(processImgItem.getIndex() == -1 ||
//                processImgItem.getIndex() <= m_currID){
//            msleep(10);
//            processImgItem = m_matImageBuff->last();
//        }
        if(processImgItem.getIndex() == -1 ||
                processImgItem.getIndex() <= m_currID){
            msleep(5);
            continue;
        }

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
#ifdef _test_ORBSearcher_
        cv::Rect img_OSR(cv::Point(0,0),imgSize);
#endif
        d_stabMat = processImgItem.getDeviceStabMatrix();
        h_gmeMat = processImgItem.getDeviceGMEMatrix();
        d_gmeMat = processImgItem.getHostGMEMatrix();

        cv::Mat nv12Img = cv::Mat(imgSize.height * 3 / 2, imgSize.width, CV_8UC1, h_i420Image);
        m_i420Img = cv::Mat(imgSize.height * 3 / 2, imgSize.width, CV_8UC1, h_i420Image);
        if(m_grayFrame.cols > 0 && m_grayFrame.rows > 0){
            m_grayFramePrev = m_grayFrame.clone();
        }
        m_grayFrame = cv::Mat(imgSize.height , imgSize.width, CV_8UC1, h_i420Image);

        if(m_i420Img.cols <= 0 || m_i420Img.rows <= 0){
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            //            printf("m_i420Img size invalid\r\n");
            continue;
        }
#ifdef DEBUG_TIMER
        stop = clock();
        {
            clock_t timeSpan = stop - start;
            std::cout << "Get frame ["<< m_currID <<"] ["<<((double)timeSpan)/CLOCKS_PER_SEC * 1000<< "]" << std::endl;
        }
#endif

        image_t i420Img;
        i420Img.c = 1;
        i420Img.h = imgSize.height * 3/2;
        i420Img.w = imgSize.width;
        i420Img.data = (float *)d_i420Image;
#ifdef USE_LINE_DETECTOR
        if(m_plrEngine == nullptr){
            m_plrEngine = my_pli::createPlrEngine(m_grayFrame.size());
        }
#endif
#ifdef DEBUG_TIMER
        start = clock();
#endif
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
            if(!m_tracker->isInitialized()){
                setClick(w/2,h/2,w,h);
            }
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
                m_objectType = -1;
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
                    if(m_objectSearch){
                        /** --- Object detect
                         * @Editor: giapvn
                         */
                        // Create an area with maximum size of [512x512] or 4x m_trackSize expanded from click point for OD
                        cv::Rect g_roi;

                        g_roi.x       = ((lockPoint.x - m_trackSize * 2) < 0) ? 0 : (lockPoint.x - m_trackSize *2);
                        g_roi.y       = ((lockPoint.y - m_trackSize * 2) < 0) ? 0 : (lockPoint.y - m_trackSize*2);
                        g_roi.width   = ((g_roi.x + m_trackSize*4) >= w) ? (w - g_roi.x) : (m_trackSize*4);
                        g_roi.height  = ((g_roi.y + m_trackSize*4) >= h) ? (h - g_roi.y) : (m_trackSize*4);

                        if(g_roi.width > MAX_SIZE){
                            g_roi.x += (g_roi.width-MAX_SIZE)/2;
                            g_roi.width = MAX_SIZE;
                        }
                        if(g_roi.height > MAX_SIZE){
                            g_roi.y += (g_roi.height-MAX_SIZE)/2;
                            g_roi.height = MAX_SIZE;
                        }
                        // Detect objects inside the area
                        std::vector<bbox_t> detection_boxes = m_detector->gpu_detect_roi_I420(i420Img, g_roi, 0.2, false);

                        // If there is one object detected at least
                        if(detection_boxes.size() > 0)
                        {
                            printf("==============================\r\n");
                            float minDist = 1920.f;
                            int minIdx = -1;
                            for(size_t i = 0; i < detection_boxes.size(); i++)
                            {
                                if(detection_boxes[i].obj_id != 0 && detection_boxes[i].obj_id != 11)
                                {
                                    float dist = ((detection_boxes[i].x + detection_boxes[i].w/2 - lockPoint.x) * (detection_boxes[i].x + detection_boxes[i].w/2 - lockPoint.x) +
                                                  (detection_boxes[i].y + detection_boxes[i].h/2 - lockPoint.y) * (detection_boxes[i].y + detection_boxes[i].h/2 - lockPoint.y));
                                    if(dist < minDist)
                                    {
                                        minDist = dist;
                                        minIdx = int(i);
                                    }
                                }
                            }
                            // If there is no valid object detected
                            if(minIdx == -1)
                            {
                                if(m_gimbal->context()->m_lockMode != "VISUAL")
                                    m_gimbal->context()->m_lockMode = "VISUAL";
                            }
                            else
                            {
                                m_objectType = int(detection_boxes[minIdx].obj_id);
                                trackRectTmp = cv::Rect(int(detection_boxes[minIdx].x), int(detection_boxes[minIdx].y), int(detection_boxes[minIdx].w), int(detection_boxes[minIdx].h));
                                if(m_gimbal->context()->m_lockMode != "TRACK")
                                    m_gimbal->context()->m_lockMode = "TRACK";
                            }
                        }
                        // If there is no object detected then switch into Steering Mode at the click point
                        else
                        {
                            if(m_gimbal->context()->m_lockMode != "VISUAL")
                                m_gimbal->context()->m_lockMode = "VISUAL";
                        }
                        // Object detect ---
                    }
                    if(trackRectTmp.x > 0 && trackRectTmp.x + trackRectTmp.width < w &&
                            trackRectTmp.y > 0 && trackRectTmp.y + trackRectTmp.height < h){
                        if(m_tracker->isInitialized()){
                            m_tracker->resetTrack();
                        }
                        m_tracker->initTrack(m_grayFramePrev,trackRectTmp);
#ifdef _test_ORBSearcher_
                        if(m_objectSearch)
                        {
                            cv::cvtColor(m_i420Img,bgrImg,CV_YUV2BGR_I420);
                            init_set.clear();
                            init_set.push_back(trackRectTmp);
                            f_update = process.update(bgrImg,init_set);
                            printf("=========== Searching for id: %d\n", m_objectType);

                            if(f_update){
                                cout << "Init searcher success!" << endl;
                            }else{
                                cout << "Init searcher fail!" << endl;
                            }
                            m_searchCount = 0;
                        }
#endif
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
                    if(m_gimbal->context()->m_sensorID == 1){
                        m_zoomIR+=m_moveZoomRate /3;
                        if(m_zoomIR > m_gimbal->context()->m_digitalZoomMax[1]){
                            m_zoomIR = m_gimbal->context()->m_digitalZoomMax[1];
                        }else if(m_zoomIR < 1){
                            m_zoomIR = 1;
                        }
                        m_gimbal->context()->m_zoom[1]= m_zoomIR;
                        //                        m_gimbal->context()->m_hfov[1] = atanf(
                        //                                    tan(m_gimbal->context()->m_hfovMax[1]/2/180*M_PI)/m_zoomIR
                        //                                )/M_PI*180*2;
                        //                        printf("IR zoomMin[%f] zoomMax[%f] zoomRatio[%f] digitalZoomMax[%f]\r\n",
                        //                               m_gimbal->zoomMin(),
                        //                               m_gimbal->zoomMax(),
                        //                               m_gimbal->zoom(),
                        //                               m_gimbal->digitalZoomMax());
                    }

                    //                    Q_EMIT zoomTargetChanged(m_r);
                    //                    Q_EMIT zoomTargetChangeStopped(m_r);
                }
                cv::Point lockPoint(static_cast<int>(m_dx +
                                                     (temp.panRate * cos(m_rotationAlpha) + temp.tiltRate * sin(m_rotationAlpha))/m_r),
                                    static_cast<int>(m_dy +
                                                     (-temp.panRate * sin(m_rotationAlpha) + temp.tiltRate * cos(m_rotationAlpha))/m_r)
                                    );
                if(m_trackEnable){
                    int trackSizeTmp = m_trackSize;
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
                if(m_objectSearch){
                    m_trackRect = trackRect;
                }else{
                    m_trackRect.x = static_cast<int>(
                                static_cast<float>(m_dx) - static_cast<float>(m_trackSize)/2);
                    m_trackRect.y = static_cast<int>(
                                static_cast<float>(m_dy) - static_cast<float>(m_trackSize)/2);
                    m_trackRect.width = m_trackSize;
                    m_trackRect.height = m_trackSize;
                }
                if(m_tracker->getState() == TRACK_INVISION || m_tracker->getState() == TRACK_OCCLUDED){
                    if(m_gimbal->context()->m_lockMode == "TRACK"){
#ifdef _test_ORBSearcher_
                        if(m_objectSearch){

                            if(!f_update){
                                if(trackRect.width <= 0 || trackRect.height <= 0 || img_OSR.contains(trackRect.tl()) == false || img_OSR.contains(trackRect.br()) == false){
                                    cout << "Roi input wrong!" << endl;
                                }else{
                                    //                                cout << "Re try update!" << endl;
                                    init_set.clear();
                                    init_set.push_back(trackRect);
                                    cv::cvtColor(m_i420Img,bgrImg,CV_YUV2BGR_I420);
                                    f_update = process.update(bgrImg,init_set);
                                }

                            }
                        }
#endif
                    }

                    Q_EMIT trackStateFound(0,
                                           static_cast<double>(m_trackRect.x),
                                           static_cast<double>(m_trackRect.y),
                                           static_cast<double>(m_trackRect.width),
                                           static_cast<double>(m_trackRect.height),
                                           static_cast<double>(w),
                                           static_cast<double>(h));
                }else{
                    if(m_gimbal->context()->m_lockMode == "TRACK"){
                        m_searchCount++;
                        if(m_searchCount >= m_maxSearchCount){
                            Q_EMIT trackStateLost();
                        }else{

                            // --- Object detect
                            //                            cv::Rect g_roi;
                            //                            cv::Point lockPoint(m_dx,m_dy);
                            //                            g_roi.x       = ((lockPoint.x - m_trackSize * 2) < 0) ? 0 : (lockPoint.x - m_trackSize *2);
                            //                            g_roi.y       = ((lockPoint.y - m_trackSize * 2) < 0) ? 0 : (lockPoint.y - m_trackSize*2);
                            //                            g_roi.width   = ((g_roi.x + m_trackSize*4) >= w) ? (w - g_roi.x) : (m_trackSize*4);
                            //                            g_roi.height  = ((g_roi.y + m_trackSize*4) >= h) ? (h - g_roi.y) : (m_trackSize*4);

                            //                            if(g_roi.width > MAX_SIZE){
                            //                                g_roi.x += (g_roi.width-MAX_SIZE)/2;
                            //                                g_roi.width = MAX_SIZE;
                            //                            }
                            //                            if(g_roi.height > MAX_SIZE){
                            //                                g_roi.y += (g_roi.height-MAX_SIZE)/2;
                            //                                g_roi.height = MAX_SIZE;
                            //                            }
                            //                            std::vector<bbox_t> detection_boxes = m_detector->gpu_detect_roi_I420(i420Img, g_roi, 0.2, false);
                            std::vector<bbox_t> detection_boxes = m_detector->gpu_detect_I420(i420Img, int(w), int(h), 0.2f, false);
                            // Object detect ---
#ifdef _test_ORBSearcher_
                            if(m_objectSearch){
                                init_set.clear();
                                for(size_t i=0; i < detection_boxes.size(); i++){
                                    printf("=========== Searching for id: %d\n", detection_boxes[i].obj_id);
                                    if(m_objectType == 3 || m_objectType == 10)
                                    {
                                        if(detection_boxes[i].obj_id == 3 || detection_boxes[i].obj_id == 10)
                                            init_set.push_back(cv::Rect(int(detection_boxes[i].x),
                                                                        int(detection_boxes[i].y),
                                                                        int(detection_boxes[i].w),
                                                                        int(detection_boxes[i].h)));
                                    }
                                    else if(m_objectType == 1 || m_objectType == 2)
                                    {
                                        if(detection_boxes[i].obj_id == 1 || detection_boxes[i].obj_id == 2)
                                            init_set.push_back(cv::Rect(int(detection_boxes[i].x),
                                                                        int(detection_boxes[i].y),
                                                                        int(detection_boxes[i].w),
                                                                        int(detection_boxes[i].h)));
                                    }
                                    else if(m_objectType > 3 && m_objectType < 10)
                                    {
                                        if(detection_boxes[i].obj_id > 3 && detection_boxes[i].obj_id < 10)
                                            init_set.push_back(cv::Rect(int(detection_boxes[i].x),
                                                                        int(detection_boxes[i].y),
                                                                        int(detection_boxes[i].w),
                                                                        int(detection_boxes[i].h)));
                                    }
                                }
                                if(init_set.empty())
                                    continue;
                                cv::RotatedRect detectedObjBound;
                                cv::cvtColor(m_i420Img,bgrImg,CV_YUV2BGR_I420);
                                if(process.predict(bgrImg,init_set,detectedObjBound)){
                                    printf("Candidate object found!\n");
                                    cv::Rect suggestRs = detectedObjBound.boundingRect();
                                    if(m_tracker->isInitialized()){
                                        m_tracker->resetTrack();
                                    }
                                    m_tracker->initTrack(m_grayFrame,suggestRs);
                                }else{
                                    printf("Try to search object fail!\n");
                                }
                            }
#endif
                        }
                    }
                    else{
                        Q_EMIT trackStateLost();
                    }
                }
            }else{

            }
        }else{
            m_trackRect.x = m_dx - m_trackSize/2;
            m_trackRect.y = m_dy - m_trackSize/2;
            m_trackRect.width = m_trackSize;
            m_trackRect.height = m_trackSize;
        }
#ifdef USE_LINE_DETECTOR
        if(m_powerLineDetectEnable){
            m_plrEngine->init_track_plr(m_grayFrame,m_powerLineDetectRect,m_powerLineList,m_plrRR);
        }
#endif
        if(m_stabEnable){
            if(m_gimbal->context()->m_sensorID == 0){
                m_ptzMatrix = createPtzMatrix(w,h,m_dx,m_dy,1 * m_scale,m_rotationAlpha);
            }else{
                m_ptzMatrix = createPtzMatrix(w,h,m_dx,m_dy,m_zoomIR * m_scale,m_rotationAlpha);
            }
        }else{
            if(m_gimbal->context()->m_sensorID == 0){
                m_ptzMatrix = createPtzMatrix(w,h,w/2,h/2,1,m_rotationAlpha);
            }else{
                m_ptzMatrix = createPtzMatrix(w,h,w/2,h/2,m_zoomIR,m_rotationAlpha);
            }
        }
#ifdef DEBUG_TIMER
        stop = clock();
        {
            clock_t timeSpan = stop - start;
            std::cout << "Process ["<<((double)timeSpan)/CLOCKS_PER_SEC * 1000<< "]" << std::endl;
            sleepTime = (long)(33333 - std::chrono::duration<double, std::micro>(stop - start).count());
        }
        start = clock();
#endif
        // add data to display worker
        ProcessImageCacheItem resProcessImgItem;
        resProcessImgItem.setIndex(m_currID);
        resProcessImgItem.setHostImage(h_i420Image);
        resProcessImgItem.setDeviceImage(d_i420Image);
        resProcessImgItem.setImageSize(imgSize);
        resProcessImgItem.setHostStabMatrix(m_ptzMatrix);
        resProcessImgItem.setDeviceStabMatrix(d_stabMat);
        resProcessImgItem.setHostGMEMatrix(h_gmeMat);
        resProcessImgItem.setDeviceGMEMatrix(d_gmeMat);
        resProcessImgItem.setLockMode(m_gimbal->context()->m_mapLockMode[m_gimbal->context()->m_lockMode]);
        resProcessImgItem.setTrackRect(m_trackRect);
        resProcessImgItem.setTrackStatus(m_tracker->getState());
        resProcessImgItem.setZoom(m_gimbal->context()->m_zoom[m_gimbal->context()->m_sensorID]);
#ifdef USE_LINE_DETECTOR
        resProcessImgItem.setPowerlineDetectEnable(m_powerLineDetectEnable);
        resProcessImgItem.setPowerlineDetectRect(m_powerLineDetectRect);
        resProcessImgItem.setPowerLineList(m_powerLineList);
#endif
        resProcessImgItem.setSensorID(m_gimbal->context()->m_sensorID);
        resProcessImgItem.setColorMode(m_colorMode);
        m_matTrackBuff->add(resProcessImgItem);
#ifdef DEBUG_TIMER
        stop = clock();
        {
            clock_t timeSpan = stop - start;
            std::cout << "Push to buffer ["<<((double)timeSpan)/CLOCKS_PER_SEC * 1000<< "]" << std::endl;
            sleepTime = 33 - timeSpan;
        }
#endif
        msleep(1);
        //        printf("VTrackWorker: %d - [%d, %d] \r\n", m_currID, imgSize.width, imgSize.height);
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
void VTrackWorker::setObjDetector(Detector *_detector)
{
    m_detector = _detector;
}
void VTrackWorker::setClicktrackDetector(Detector *_detector)
{
    m_clickTrack->setDetector(_detector);
}

void VTrackWorker::setOCR(OCR* _OCR)
{
    m_clickTrack->setOCR(_OCR);
}
void VTrackWorker::setDetector(Detector* _detector){
    m_detector = _detector;
}
