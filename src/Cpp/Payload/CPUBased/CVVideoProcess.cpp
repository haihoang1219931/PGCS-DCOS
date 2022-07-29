#include "CVVideoProcess.h"
#include "Payload/VideoDisplay/ImageItem.h"
#include "Payload/VideoEngine/VideoEngineInterface.h"
#include "Payload/GimbalController/GimbalInterface.h"

CVVideoProcess::CVVideoProcess(QObject *parent): QObject(parent)
{
    m_gstRTSPBuff = Cache::instance()->getGstRTSPCache();
    m_buffVideoSaving = Cache::instance()->getGstEOSavingCache();
#ifdef TRACK_DANDO
    m_tracker = new ITrack(FEATURE_HOG, KERNEL_GAUSSIAN);
#else
    m_tracker = new Tracker();
#endif
    m_mutexCommand = new QMutex();
    m_mutex = new QMutex();
    m_pauseCond = new QWaitCondition();
    m_warpDataRender = std::vector<float>(16,0);
    m_warpDataRender[0*4+0] = 1;
    m_warpDataRender[1*4+1] = 1;
    m_warpDataRender[2*4+2] = 1;
    m_warpDataRender[3*4+3] = 1;
}
CVVideoProcess::~CVVideoProcess()
{
    delete m_stabilizer;
    delete m_tracker;
}
void CVVideoProcess::changeTrackSize(float _trackSize)
{
    m_trackSize = (int)_trackSize;
    m_trackSizePrev = m_trackSize;
}
void CVVideoProcess::setClick(float x, float y,float width,float height){
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
void CVVideoProcess::moveImage(float panRate,float tiltRate,float zoomRate, float alpha){
    if(m_grayFrame.cols <= 0 || m_grayFrame.rows <= 0){
        return;
    }
    m_moveZoomRate = -zoomRate / maxAxis;
    //    m_rotationAlpha = alpha;
    m_movePanRate = panRate / maxAxis * (m_grayFrame.cols/2) / 10;
    m_moveTiltRate = tiltRate / maxAxis * (m_grayFrame.rows/2) / 10;
    joystickData temp;
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
        //        Q_EMIT zoomTargetChangeStopped(m_r);
        //        startRollbackZoom();
        m_zoomStart = m_gimbal->context()->m_zoom[m_gimbal->context()->m_sensorID];
    }else{
        m_jsQueue.push_back(temp);
        while(m_jsQueue.size() > 5){
            m_jsQueue.pop_front();
        }

    }
    m_mutexCommand->unlock();
}
cv::Mat CVVideoProcess::createPtzMatrix(float w, float h, float dx, float dy,float r,float alpha){
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
void CVVideoProcess::capture()
{
    std::string timestamp = FileController::get_time_stamp();
    std::string captureFile = m_logFolder + "/" + timestamp + ".jpg";
    printf("Save file %s\r\n", captureFile.c_str());
    cv::imwrite(captureFile, m_img);
}
void CVVideoProcess::pause(bool _pause){
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
cv::Point CVVideoProcess::convertPoint(cv::Point originPoint, cv::Mat stabMatrix){
    cv::Point result(originPoint);
    if(stabMatrix.rows == 3 && stabMatrix.cols == 3){
        cv::Mat stabMatrixD(3, 3, CV_64FC1,cv::Scalar::all(0));
        stabMatrixD.at<double>(0,0) = static_cast<double>(stabMatrix.at<float>(0,0));
        stabMatrixD.at<double>(0,1) = static_cast<double>(stabMatrix.at<float>(0,1));
        stabMatrixD.at<double>(0,2) = static_cast<double>(stabMatrix.at<float>(0,2));
        stabMatrixD.at<double>(1,0) = static_cast<double>(stabMatrix.at<float>(1,0));
        stabMatrixD.at<double>(1,1) = static_cast<double>(stabMatrix.at<float>(1,1));
        stabMatrixD.at<double>(1,2) = static_cast<double>(stabMatrix.at<float>(1,2));
        stabMatrixD.at<double>(2,0) = static_cast<double>(stabMatrix.at<float>(2,0));
        stabMatrixD.at<double>(2,1) = static_cast<double>(stabMatrix.at<float>(2,1));
        stabMatrixD.at<double>(2,2) = static_cast<double>(stabMatrix.at<float>(2,2));
        cv::Mat pointBeforeStab(3, 1, CV_64FC1,cv::Scalar::all(0));
        pointBeforeStab.at<double>(0,0) = static_cast<double>(originPoint.x);
        pointBeforeStab.at<double>(1,0) = static_cast<double>(originPoint.y);
        pointBeforeStab.at<double>(2,0) = 1;
        cv::Mat pointInStab(3, 1, CV_64FC1);
        pointInStab = stabMatrixD*pointBeforeStab;
        result.x = pointInStab.at<double>(0,0);
        result.y = pointInStab.at<double>(1,0);
    }
    return result;
}
void CVVideoProcess::doWork()
{
    printf("CVVideoProcess dowork started\r\n");
    msleep(1000);
    int firstFrameCount = 0;
    cv::Mat imgYWarped;
    cv::Mat imgUWarped;
    cv::Mat imgVWarped;
    cv::Mat warpMatrix = cv::Mat(3,3,CV_32FC1);
    std::chrono::high_resolution_clock::time_point start, stop;
    int sleepTime = 0;
    while (m_stop == false) {
        m_mutex->lock();
        if(m_pause)
            m_pauseCond->wait(m_mutex); // in this place, your thread will stop to execute until someone calls resume
        m_mutex->unlock();
        gboolean res;
        gint32 width, height;
        GstSample *sample;
        GstCaps *caps;
        GstBuffer *buf;
        GstStructure *str;
        GstMapInfo map;
        std::pair <int, GstSample *> data;
        if (m_imageQueue->size() > 0) {
            firstFrameCount++;
            m_mutexCapture->lock();
            data = m_imageQueue->back();
            m_mutexCapture->unlock();
        } else {
            continue;
        }
        if (data.first == 0) {
            *m_frameID = data.first;
            sample = data.second;
        } else if (data.first != *m_frameID) {
            *m_frameID = data.first;
            sample = data.second;
        } else {
            msleep(SLEEP_TIME);
            continue;
        }

        if (!GST_IS_SAMPLE(sample)) {
            continue;
        }

        caps = gst_sample_get_caps(sample);

        if (!GST_IS_CAPS(caps)) {
            continue;
        }

        str = gst_caps_get_structure(caps, 0);

        if (!GST_IS_STRUCTURE(str)) {
            continue;
        }

        res = gst_structure_get_int(str, "width", &width);
        res |= gst_structure_get_int(str, "height", &height);

        if (!res || width == 0 || height == 0) {
            g_print("could not get snapshot dimension\n");
            continue;
        }

        buf = gst_buffer_copy(gst_sample_get_buffer(sample));

        if (!GST_IS_BUFFER(buf)) {
            g_print("Could not get buf\n");
            continue;
        }

        gst_buffer_map(buf, &map, GST_MAP_READ);
        //        printf("map.size=%d\r\n", map.size);
        m_imgI420 = cv::Mat(height * 3 / 2 ,  map.size / height / 3 * 2, CV_8UC1, map.data);
        if(m_imgI420Warped.rows != height * 3 / 2 || m_imgI420Warped.cols != map.size / height / 3 * 2){
            m_imgI420Warped = cv::Mat(height * 3 / 2 ,  map.size / height / 3 * 2, CV_8UC1);
            imgYWarped = cv::Mat(static_cast<int>(height), static_cast<int>(width), CV_8UC1, m_imgI420Warped.data);
            imgUWarped = cv::Mat(static_cast<int>(height/2), static_cast<int>(width/2), CV_8UC1, m_imgI420Warped.data + size_t(height * width));
            imgVWarped = cv::Mat(static_cast<int>(height/2), static_cast<int>(width/2), CV_8UC1, m_imgI420Warped.data + size_t(height * width * 5 / 4));
        }
        if(m_grayFrame.cols > 0 && m_grayFrame.rows > 0){
            m_grayFramePrev = m_grayFrame.clone();
        }
        m_grayFrame = cv::Mat(height,width,CV_8UC1,map.data);
        gst_buffer_unmap(buf, &map);
        gst_buffer_unref(buf);
        start = std::chrono::high_resolution_clock::now();        
        if(!m_gimbal->context()->m_processOnBoard){
            //TODO: Perfrom tracking
            if(static_cast<int>(m_w) != m_grayFrame.cols)
            {
                m_w = static_cast<float>(m_grayFrame.cols);
                m_dx = m_w / 2;
            }

            if(static_cast<int>(m_h) != m_grayFrame.rows)
            {
                m_h = static_cast<float>(m_grayFrame.rows);
                m_dy = m_h / 2;
            }
            // handle command
            if(m_gimbal->context()->m_lockMode == "FREE"){
                m_trackEnable = false;
            }else if(m_gimbal->context()->m_lockMode == "TRACK" ||
                     m_gimbal->context()->m_lockMode == "VISUAL"){
                m_trackEnable = true;
                if(!m_tracker->isInitialized()){
                    setClick(m_w/2,m_h/2,m_w,m_h);
                }
                if(m_gimbal->context()->m_lockMode == "VISUAL"){
                    if( m_h/2 > m_trackSize)
                        m_trackSize = m_h/2;
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
                        if(trackRectTmp.x > 0 && trackRectTmp.x + trackRectTmp.width < m_w &&
                                trackRectTmp.y > 0 && trackRectTmp.y + trackRectTmp.height < m_h){
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
                        if(m_gimbal->context()->m_sensorID == 1){
                            m_zoomIR+=m_moveZoomRate /3;
                            if(m_zoomIR > m_gimbal->context()->m_digitalZoomMax[1]){
                                m_zoomIR = m_gimbal->context()->m_digitalZoomMax[1];
                            }else if(m_zoomIR < 1){
                                m_zoomIR = 1;
                            }
                            m_gimbal->context()->m_zoom[1]= m_zoomIR;
                            Q_EMIT m_gimbal->zoomChanged();
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
                        }else if(trackRectTmp.x + trackRectTmp.width > m_w - deadSpace){
                            trackRectTmp.x = m_w - deadSpace - trackRectTmp.width;
                        }

                        if(trackRectTmp.y < deadSpace){
                            trackRectTmp.y = deadSpace;
                        }else if(trackRectTmp.y + trackRectTmp.height > m_h - deadSpace){
                            trackRectTmp.y = m_h - deadSpace - trackRectTmp.height;
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
                    if(m_tracker->getState() == TRACK_INVISION || m_tracker->getState() == TRACK_OCCLUDED){
                        Q_EMIT trackStateFound(0,
                                               static_cast<double>(m_trackRect.x),
                                               static_cast<double>(m_trackRect.y),
                                               static_cast<double>(m_trackRect.width),
                                               static_cast<double>(m_trackRect.height),
                                               static_cast<double>(m_w),
                                               static_cast<double>(m_h));
                    }else{
                        Q_EMIT trackStateLost();
                    }
                }else{

                }
            }else{
                m_trackRect.x = m_dx - m_trackSize/2;
                m_trackRect.y = m_dy - m_trackSize/2;
                m_trackRect.width = m_trackSize;
                m_trackRect.height = m_trackSize;
            }
            if(m_stabEnable){
                if(m_gimbal->context()->m_sensorID == 0){
                    m_ptzMatrix = createPtzMatrix(m_w,m_h,m_dx,m_dy,1 * m_scale,m_rotationAlpha);
                }else{
                    m_ptzMatrix = createPtzMatrix(m_w,m_h,m_dx,m_dy,m_zoomIR * m_scale,m_rotationAlpha);
                }
            }else{
                if(m_gimbal->context()->m_sensorID == 0){
                    m_ptzMatrix = createPtzMatrix(m_w,m_h,m_w/2,m_h/2,1,m_rotationAlpha);
                }else{
                    m_ptzMatrix = createPtzMatrix(m_w,m_h,m_w/2,m_h/2,m_zoomIR,m_rotationAlpha);
                }
            }

            warpMatrix.at<float>(0,0) = static_cast<float>(m_ptzMatrix.at<double>(0,0));
            warpMatrix.at<float>(0,1) = static_cast<float>(m_ptzMatrix.at<double>(0,1));
            warpMatrix.at<float>(0,2) = static_cast<float>(m_ptzMatrix.at<double>(0,2));
            warpMatrix.at<float>(1,0) = static_cast<float>(m_ptzMatrix.at<double>(1,0));
            warpMatrix.at<float>(1,1) = static_cast<float>(m_ptzMatrix.at<double>(1,1));
            warpMatrix.at<float>(1,2) = static_cast<float>(m_ptzMatrix.at<double>(1,2));

            cv::Mat imgY(static_cast<int>(height), static_cast<int>(width), CV_8UC1, m_imgI420.data);
            cv::Mat imgU(static_cast<int>(height/2), static_cast<int>(width/2), CV_8UC1, m_imgI420.data + size_t(height * width));
            cv::Mat imgV(static_cast<int>(height/2), static_cast<int>(width/2), CV_8UC1, m_imgI420.data + size_t(height * width * 5 / 4));
            cv::BorderTypes borderType = cv::BorderTypes::BORDER_CONSTANT;
            cv::Scalar color(100,100,100);
            double y,u,v;
            VideoEngine::convertRGB2YUV((double)color.val[0],(double)color.val[1],(double)color.val[2],y,u,v);
            cv::warpAffine(imgY,imgYWarped,warpMatrix(cv::Rect(0,0,3,2)),cv::Size(imgY.cols,imgY.rows),cv::INTER_NEAREST,borderType,cv::Scalar(y));

            warpMatrix.at<float>(0,2) = static_cast<float>(m_ptzMatrix.at<double>(0,2))/2;
            warpMatrix.at<float>(1,2) = static_cast<float>(m_ptzMatrix.at<double>(1,2))/2;

            cv::warpAffine(imgU,imgUWarped,warpMatrix(cv::Rect(0,0,3,2)),cv::Size(imgU.cols,imgU.rows),cv::INTER_NEAREST,borderType,cv::Scalar(u));
            cv::warpAffine(imgV,imgVWarped,warpMatrix(cv::Rect(0,0,3,2)),cv::Size(imgV.cols,imgV.rows),cv::INTER_NEAREST,borderType,cv::Scalar(v));
            // draw track
            warpMatrix.at<float>(0,2) = static_cast<float>(m_ptzMatrix.at<double>(0,2));
            warpMatrix.at<float>(1,2) = static_cast<float>(m_ptzMatrix.at<double>(1,2));
            cv::Rect trackRect = m_trackRect;
            cv::Point pointAfterStab = convertPoint(cv::Point(trackRect.x+trackRect.width/2,
                                                              trackRect.y+trackRect.height/2),
                                                    warpMatrix);
            trackRect.x = pointAfterStab.x - trackRect.width/2;
            trackRect.y = pointAfterStab.y - trackRect.height/2;
            cv::Scalar colorInvision(255,0,0);
            cv::Scalar colorOccluded(0,0,0);
            if(m_gimbal->context()->m_lockMode == "TRACK"){
                if(m_tracker->getState() == TRACK_INVISION){
                    VideoEngine::rectangle(imgYWarped,imgUWarped,imgVWarped,
                                           trackRect,colorInvision,2);
                }else if(m_tracker->getState() == TRACK_OCCLUDED){
                    VideoEngine::rectangle(imgYWarped,imgUWarped,imgVWarped,
                                           trackRect,colorOccluded,2);
                }else{

                }
            }else if(m_gimbal->context()->m_lockMode == "VISUAL"){
                if(m_tracker->getState() == TRACK_INVISION){
                    VideoEngine::drawSteeringCenter(imgYWarped,imgUWarped,imgVWarped,
                                       trackRect.width,
                                       static_cast<int>(trackRect.x + trackRect.width/2),
                                       static_cast<int>(trackRect.y + trackRect.height/2),
                                       colorInvision);
                }else if(m_tracker->getState() == TRACK_OCCLUDED){
                    VideoEngine::drawSteeringCenter(imgYWarped,imgUWarped,imgVWarped,
                                       trackRect.width,
                                       static_cast<int>(trackRect.x + trackRect.width/2),
                                       static_cast<int>(trackRect.y + trackRect.height/2),
                                       colorOccluded);
                }else{

                }
            }
        }
        else{
            m_imgI420.copyTo(m_imgI420Warped);
        }

        Q_EMIT readyDrawOnRenderID(0,m_imgI420Warped.data,width,height,m_warpDataRender.data(),nullptr);
        Q_EMIT readyDrawOnRenderID(1,m_imgI420Warped.data,width,height,m_warpDataRender.data(),nullptr);
        if(false)
        {
            if(m_sharedEnable && m_recordEnable){
                GstBuffer *rtspImage = gst_buffer_new();
                assert(rtspImage != NULL);
                GstMemory *gstMem = gst_allocator_alloc(NULL, m_imgI420Warped.u->size, NULL);
                assert(gstMem != NULL);
                gst_buffer_append_memory(rtspImage, gstMem);
                GstMapInfo mapT;
                gst_buffer_map(rtspImage, &mapT, GST_MAP_READ);
                memcpy((void *)mapT.data, m_imgI420Warped.data, m_imgI420Warped.u->size);
                gst_buffer_unmap(rtspImage , &mapT);
                //add to rtsp
                GstFrameCacheItem gstFrame;
                gstFrame.setIndex(*m_frameID);
                gstFrame.setGstBuffer(rtspImage);
                m_gstRTSPBuff->add(gstFrame);
//                printf("Adding rtsp frame m_gstRTSPBuff->size()=%d\r\n",m_gstRTSPBuff->size());
                //add to saving
                GstFrameCacheItem gstFrameSaving;
                gstFrameSaving.setIndex(*m_frameID);
                gstFrameSaving.setGstBuffer(gst_buffer_copy(rtspImage));
                m_buffVideoSaving->add(gstFrameSaving);
            }else if(m_sharedEnable && !m_recordEnable){
                GstBuffer *rtspImage = gst_buffer_new();
                assert(rtspImage != NULL);
                GstMemory *gstMem = gst_allocator_alloc(NULL, m_imgI420Warped.u->size, NULL);
                assert(gstMem != NULL);
                gst_buffer_append_memory(rtspImage, gstMem);
                GstMapInfo mapT;
                gst_buffer_map(rtspImage, &mapT, GST_MAP_READ);
                memcpy((void *)mapT.data, m_imgI420Warped.data, m_imgI420Warped.u->size);
                gst_buffer_unmap(rtspImage , &mapT);
                //add to rtsp
                GstFrameCacheItem gstFrame;
                gstFrame.setIndex(*m_frameID);
                gstFrame.setGstBuffer(rtspImage);
                m_gstRTSPBuff->add(gstFrame);
            }else if(!m_sharedEnable && m_recordEnable){
                GstBuffer *savingImage = gst_buffer_new();
                assert(savingImage != NULL);
                GstMemory *gstMem = gst_allocator_alloc(NULL, m_imgI420Warped.u->size, NULL);
                assert(gstMem != NULL);
                gst_buffer_append_memory(savingImage, gstMem);
                GstMapInfo mapT;
                gst_buffer_map(savingImage, &mapT, GST_MAP_READ);
                memcpy((void *)mapT.data, m_imgI420Warped.data, m_imgI420Warped.u->size);
                gst_buffer_unmap(savingImage , &mapT);
                //add to saving
                GstFrameCacheItem gstFrameSaving;
                gstFrameSaving.setIndex(*m_frameID);
                gstFrameSaving.setGstBuffer(savingImage);
                m_buffVideoSaving->add(gstFrameSaving);
            }else{

            }
        }
        stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> timeSpan = stop - start;
        sleepTime = 33333 - timeSpan.count();
//        std::cout << "timeSpan: " << timeSpan.count() <<std::endl;
        msleep(sleepTime/1000);
    }

    Q_EMIT stopped();
}

void CVVideoProcess::msleep(int ms)
{
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
