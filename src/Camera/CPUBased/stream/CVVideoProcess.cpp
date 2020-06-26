#include "CVVideoProcess.h"
#include "../../VideoDisplay/ImageItem.h"
Q_DECLARE_METATYPE(cv::Mat)
#include "src/Camera/GimbalController/GimbalInterface.h"

CVVideoProcess::CVVideoProcess(QObject *parent): QObject(parent)
{
    m_gstRTSPBuff = Cache::instance()->getGstRTSPCache();
    m_buffVideoSaving = Cache::instance()->getGstEOSavingCache();
    qRegisterMetaType< cv::Mat >("cv::Mat");
#ifdef TRACK_DANDO
    m_tracker = new ITrack(FEATURE_HOG, KERNEL_GAUSSIAN);
#else
    m_tracker = new Tracker();
#endif
    m_mutexCommand = new QMutex();
    m_mutex = new QMutex();
    m_pauseCond = new QWaitCondition();
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
    std::string timestamp = Utils::get_time_stamp();
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
        m_i420Img = cv::Mat(height * 3 / 2 ,  map.size / height / 3 * 2, CV_8UC1, map.data);
        if(!m_gimbal->context()->m_processOnBoard){
            cv::cvtColor(m_i420Img, m_img, CV_YUV2BGRA_I420);
            gst_buffer_unmap(buf, &map);
            gst_buffer_unref(buf);
            if (m_img.cols <= 0 || m_img.rows <=  0) {
                msleep(SLEEP_TIME);
                continue;
            }
            if(m_grayFrame.cols > 0 && m_grayFrame.rows > 0){
                m_grayFramePrev = m_grayFrame.clone();
            }
            if(m_i420Img.cols <= 0 || m_i420Img.rows <= 0){
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            cv::cvtColor(m_i420Img, m_grayFrame, CV_YUV2GRAY_I420);
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
            cv::Mat warpMatrix = cv::Mat(3,3,CV_32FC1);
            warpMatrix.at<float>(0,0) = static_cast<float>(m_ptzMatrix.at<double>(0,0));
            warpMatrix.at<float>(0,1) = static_cast<float>(m_ptzMatrix.at<double>(0,1));
            warpMatrix.at<float>(0,2) = static_cast<float>(m_ptzMatrix.at<double>(0,2));
            warpMatrix.at<float>(1,0) = static_cast<float>(m_ptzMatrix.at<double>(1,0));
            warpMatrix.at<float>(1,1) = static_cast<float>(m_ptzMatrix.at<double>(1,1));
            warpMatrix.at<float>(1,2) = static_cast<float>(m_ptzMatrix.at<double>(1,2));
            cv::warpAffine(m_img,*m_imgShow,warpMatrix(cv::Rect(0,0,3,2)),cv::Size(m_img.cols,m_img.rows),cv::INTER_LINEAR);

            // draw track
            cv::Rect trackRect = m_trackRect;
            cv::Point pointAfterStab = convertPoint(cv::Point(trackRect.x+trackRect.width/2,
                                                              trackRect.y+trackRect.height/2),
                                                    warpMatrix);
            trackRect.x = pointAfterStab.x - trackRect.width/2;
            trackRect.y = pointAfterStab.y - trackRect.height/2;
            cv::Scalar colorInvision(0,0,255,255);
            cv::Scalar colorOccluded(0,0,0,255);
            if(m_gimbal->context()->m_lockMode == "TRACK"){
                if(m_tracker->Get_State() == TRACK_INVISION){
                    cv::rectangle(*m_imgShow,trackRect,colorInvision,2);
                }else if(m_tracker->Get_State() == TRACK_OCCLUDED){
                    cv::rectangle(*m_imgShow,trackRect,colorOccluded,2);
                }else{

                }
            }else if(m_gimbal->context()->m_lockMode == "VISUAL"){
                if(m_tracker->Get_State() == TRACK_INVISION){
                    drawSteeringCenter(*m_imgShow,trackRect.width,
                                       static_cast<int>(trackRect.x + trackRect.width/2),
                                       static_cast<int>(trackRect.y + trackRect.height/2),
                                       colorInvision);
                }else if(m_tracker->Get_State() == TRACK_OCCLUDED){
                    drawSteeringCenter(*m_imgShow,trackRect.width,
                                       static_cast<int>(trackRect.x + trackRect.width/2),
                                       static_cast<int>(trackRect.y + trackRect.height/2),
                                       colorOccluded);
                }else{

                }
            }
        }
        else{
            cv::cvtColor(m_i420Img, *m_imgShow, CV_YUV2BGRA_I420);
            gst_buffer_unmap(buf, &map);
            gst_buffer_unref(buf);
        }
        cv::Mat _imgOtherFunc;
        cv::Mat _imgResize;
        cv::cvtColor(*m_imgShow, _imgOtherFunc, CV_BGRA2YUV_I420);
        if(m_sharedEnable && m_recordEnable){
            GstBuffer *rtspImage = gst_buffer_new();
            assert(rtspImage != NULL);
            GstMemory *gstMem = gst_allocator_alloc(NULL, _imgOtherFunc.u->size, NULL);
            assert(gstMem != NULL);
            gst_buffer_append_memory(rtspImage, gstMem);
            GstMapInfo mapT;
            gst_buffer_map(rtspImage, &mapT, GST_MAP_READ);
            memcpy((void *)mapT.data, _imgOtherFunc.data, _imgOtherFunc.u->size);
            gst_buffer_unmap(rtspImage , &mapT);
            //add to rtsp
            GstFrameCacheItem gstFrame;
            gstFrame.setIndex(*m_frameID);
            gstFrame.setGstBuffer(rtspImage);
            m_gstRTSPBuff->add(gstFrame);
            //            printf("Adding rtsp frame m_gstRTSPBuff->size()=%d\r\n",m_gstRTSPBuff->size());
            //add to saving
            GstFrameCacheItem gstFrameSaving;
            gstFrameSaving.setIndex(*m_frameID);
            gstFrameSaving.setGstBuffer(gst_buffer_copy(rtspImage));
            m_buffVideoSaving->add(gstFrameSaving);
        }else if(m_sharedEnable && !m_recordEnable){
            GstBuffer *rtspImage = gst_buffer_new();
            assert(rtspImage != NULL);
            GstMemory *gstMem = gst_allocator_alloc(NULL, _imgOtherFunc.u->size, NULL);
            assert(gstMem != NULL);
            gst_buffer_append_memory(rtspImage, gstMem);
            GstMapInfo mapT;
            gst_buffer_map(rtspImage, &mapT, GST_MAP_READ);
            memcpy((void *)mapT.data, _imgOtherFunc.data, _imgOtherFunc.u->size);
            gst_buffer_unmap(rtspImage , &mapT);
            //add to rtsp
            GstFrameCacheItem gstFrame;
            gstFrame.setIndex(*m_frameID);
            gstFrame.setGstBuffer(rtspImage);
            m_gstRTSPBuff->add(gstFrame);
        }else if(!m_sharedEnable && m_recordEnable){
            GstBuffer *savingImage = gst_buffer_new();
            assert(savingImage != NULL);
            GstMemory *gstMem = gst_allocator_alloc(NULL, _imgOtherFunc.u->size, NULL);
            assert(gstMem != NULL);
            gst_buffer_append_memory(savingImage, gstMem);
            GstMapInfo mapT;
            gst_buffer_map(savingImage, &mapT, GST_MAP_READ);
            memcpy((void *)mapT.data, _imgOtherFunc.data, _imgOtherFunc.u->size);
            gst_buffer_unmap(savingImage , &mapT);
            //add to saving
            GstFrameCacheItem gstFrameSaving;
            gstFrameSaving.setIndex(*m_frameID);
            gstFrameSaving.setGstBuffer(savingImage);
            m_buffVideoSaving->add(gstFrameSaving);
        }else{

        }
        Q_EMIT processDone();
        msleep(SLEEP_TIME);
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
void CVVideoProcess::drawSteeringCenter(cv::Mat &_img, int _wBoundary,
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
