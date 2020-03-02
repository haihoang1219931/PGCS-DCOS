#include "CVVideoProcess.h"

CVVideoProcess::CVVideoProcess(QObject *parent): QObject(parent)
{
    m_stabilizer = new stab_gcs_kiir::vtx_KIIRStabilizer();
    m_stabilizer->setMotionEstimnator(GOOD_FEATURE, RIGID_TRANSFORM);
    ClickTrackObj = new InitTracking();
    m_tracker = new ITrack(FEATURE_HOG, KERNEL_GAUSSIAN);
    thresh_tracker = new ThresholdingTracker();
    k_tracker = new KTrackers(KType::GAUSSIAN, KFeat::HLS, true);
    m_detector = new MovingDetector(cv::Size(m_detectSize, m_detectSize));
    GlobalTrackInited = false;
}
CVVideoProcess::~CVVideoProcess()
{
    delete m_stabilizer;
    delete ClickTrackObj;
    delete m_tracker;
    delete m_detector;
    delete thresh_tracker;
    delete k_tracker;
}
void CVVideoProcess::setTrackType(QString trackType)
{
    m_trackType = trackType;

    if (GlobalTrackInited) {
        if (trackType == "KCF")
            m_tracker->initTrack(grayFrame, object_position);

        if (trackType == "sKCF")
            k_tracker->initTrack(grayFrame, object_position);

        if (trackType == "Thresh")
            thresh_tracker->initTrack(grayFrame, object_position);
    }

    printf("Change tracktype to %s\r\n", m_trackType.toStdString().c_str());
}
void CVVideoProcess::setTrack(int _x, int _y)
{
    if (m_trackEnable == true) {
        if (m_setTrack == false) {
            m_setTrack = true;
            m_pointSetTrack.x = _x;
            m_pointSetTrack.y = _y;
            GlobalTrackInited = true;
        }
    }
}
void CVVideoProcess::capture()
{
    std::string timestamp = Utils::get_time_stamp();
    std::string captureFile = m_logFolder + "/" + timestamp + ".jpg";
    printf("Save file %s\r\n", captureFile.c_str());
    cv::imwrite(captureFile, m_img);
}
void CVVideoProcess::doWork()
{
    printf("CVVideoProcess dowork started\r\n");
    msleep(1000);
    int firstFrameCount = 0;

    while (m_stop == false) {
        //        printf("Process time\r\n");
        //        msleep(1000);
        //        continue;
        //        printf("Process time count = %d\r\n",count);
        //        count ++;
        gboolean res;
        gint32 width, height;
        GstSample *sample;
        GstCaps *caps;
        GstBuffer *buf;
        GstStructure *str;
        GstMapInfo map;
        std::pair <int, GstSample *> data;

        //        auto startGetData = std::chrono::high_resolution_clock::now();
        //        printf("Process time count = %d\r\n",count);
        if (m_imageQueue->size() > 0) {
            firstFrameCount++;
            m_mutexCapture->lock();
            data = m_imageQueue->back();
            m_mutexCapture->unlock();
        } else {
            //            printf("m_imageQueue->size() <= 0\r\n");
            continue;
        }

        //        std::pair <int,GstSample*> data = m_imageQueue->front();

        //        printf("Process new Frame %d\r\n",data.first);
        if (data.first == 0) {
            *m_frameID = data.first;
            sample = data.second;
            //            printf("Process frame [%d]\r\n",data.first);
        } else if (data.first != *m_frameID) {
            *m_frameID = data.first;
            sample = data.second;
            //            printf("Process frame [%d]\r\n",data.first);
        } else {
            //            printf("Could not show frame\r\n");
            msleep(SLEEP_TIME);
            continue;
        }

        //        printf("Process frame [%d]\r\n", data.first);

        if (!GST_IS_SAMPLE(sample)) {
            //            m_flagNoSample = true;
            //            g_print("Could not get sample\n");
            continue;
        }

        caps = gst_sample_get_caps(sample);

        if (!GST_IS_CAPS(caps)) {
            //            g_print("Could not get cap\n");
            continue;
        }

        str = gst_caps_get_structure(caps, 0);

        if (!GST_IS_STRUCTURE(str)) {
            //            g_print("Could not get structure\n");
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
        cv::Mat picYV12 = cv::Mat(height * 3 / 2 ,  map.size / height / 3 * 2, CV_8UC1, map.data);
        cv::cvtColor(picYV12, m_img, CV_YUV2RGB_YV12);
        gst_buffer_unmap(buf, &map);
        gst_buffer_unref(buf);

        //        printf("m_img %d-[%dx%d] channels = %d\r\n",
        //               m_img.depth(),
        //               m_img.rows,m_img.cols,
        //               m_img.channels());
        if (m_img.cols <= 0 || m_img.rows <=  0) {
            msleep(SLEEP_TIME);
            continue;
        }

        //        auto stopGetData = std::chrono::high_resolution_clock::now();
        //        double time_ms_getData = std::chrono::duration_cast<std::chrono::microseconds>(stopGetData - startGetData).count() / 1000.0f;
        //        std::cout << "---> Get Frame [" << (*m_frameID) << "] time: " << time_ms_getData << " ms" << std::endl;
        //        auto start = std::chrono::high_resolution_clock::now();
        //        m_imgStab = m_img.clone();
        if (m_usingIPC == true && m_stabEnable == true) {
            if (m_stabilizer->run(m_img, m_imgStab, NULL) != SUCCESS) {
                //                        printf("m_stabilizer->run failed\r\n");
                m_imgStab = m_img;
            } else {
                //                        printf("m_stabilizer->run success\r\n");
            }
        } else {
            //            printf("no stab\r\n");
            //            printf("m_img [%dx%d]-%d\r\n",m_img.cols,m_img.rows,m_img.channels());
            //            printf("m_imgStab [%dx%d]-%d\r\n",m_imgStab.cols,m_imgStab.rows,m_imgStab.channels());
            m_imgStab = m_img;
        }

        /** ===========================================================
         *      MINHBQ6: DETECT MOVING OBJECT TO TRACK
         * ============================================================
         */
        cv::Mat grayFramePrev = grayFrame.clone();
        cv::cvtColor(m_imgStab, grayFrame, cv::COLOR_RGB2GRAY);
        auto start = chrono::high_resolution_clock::now();

        if (grayFramePrev.rows < m_trackSize | grayFramePrev.cols < m_trackSize) {
            continue;
        }

        if (m_sensorTrack == "IR") {
            ClickTrackObj->UpdateMovingObject(m_imgStab);
            enhancedFrame = grayFrame.clone();
            //            enhancedFrame = contrastEnhance(grayFrame, 0.5);
        }

        if (m_usingIPC == true && m_detectEnable == true) {
            if (!m_imgStab.empty() && ClickTrackObj->processed_img.rows > 0 && ClickTrackObj->processed_img.cols > 0) {
                if (m_imgStab.cols > m_detectSize && m_imgStab.rows > m_detectSize) {
                    cv::Rect result;
                    //                    printf("---> detection start\n");
                    ClickTrackObj->Init(m_imgStab, cv::Point(grayFrame.cols / 2, grayFrame.rows / 2),  m_detectSize / 2);
                    ClickTrackObj->Run();
                    result = ClickTrackObj->object_position;
                    object_position = result;
                    //                bool detectResult = false;
                    //                    std::cout << "Location object is " << result << std::endl;
                    //                //                if (object_position.width > 0 && object_position.height > 0)
                    //                //                    detectResult = true;
                    //                    printf("---> detection stop\n");
                    //                bool detectResult = m_detector->process(m_imgStab, result);
                    //                    printf("Start init tracking \n");

                    if (result.width >= 0 && result.height > 0)
                        /*if (detectResult == true) */{
                        cv::rectangle(m_imgStab, result, cv::Scalar(255, 0, 255), 1, cv::LINE_AA);

                        // --------------original work-------------------------
                        if (m_trackType == "KCF") {
                            if (m_tracker->isInitialized()) {
                                while (m_tracker->isRunning());

                                m_tracker ->resetTrack();
                            }

                            m_tracker->initTrack(grayFrame, result);
                        }

                        if (m_trackType == "sKCF") {
                            if (k_tracker->isInititalized()) {
                                while (k_tracker->isRunning());

                                k_tracker->resetTrack();
                            }

                            k_tracker->initTrack(grayFrame, result);
                        }

                        //------------thresholding----------------
                        if (thresh_tracker->isInitialized()) {
                            while (thresh_tracker->isRunning());

                            thresh_tracker->resetTrack();
                        }

                        thresh_tracker->initTrack(grayFrame, result);
                        //                        printf("Check Init %d \n", thresh_tracker->isInitialized());
                        //----------------------------------------
                        m_detectEnable = false;
                        Q_EMIT detectObject();
                        continue;
                    }
                }
            }
        }

        /** ===========================================================
         *      MINHBQ6: TRACK OBJECT
         * ============================================================
         */
        if (m_usingIPC == true && m_trackEnable == true) {
            if (m_setTrack) {
                int x, y;
                int xStab, yStab;

                if (m_stabEnable == true) {
                    xStab = m_pointSetTrack.x + m_imgStab.cols * (1 - m_cropRatio) / 2;
                    yStab = m_pointSetTrack.y + m_imgStab.rows * (1 - m_cropRatio) / 2;
                } else {
                    xStab = m_pointSetTrack.x;
                    yStab = m_pointSetTrack.y;
                }

                printf("1: Init tracker at (%d,%d)\r\n", xStab, yStab);

                if (xStab + m_trackSize / 2 > m_imgStab.cols || yStab + m_trackSize / 2 > m_imgStab.rows ||
                    xStab - m_trackSize / 2 < 0 || yStab - m_trackSize / 2 < 0) {
                    printf("\n====>DCMN ahihi");
                    m_setTrack = false;
                    Q_EMIT trackInitSuccess(false, xStab, yStab, m_imgStab.cols, m_imgStab.rows);
                    continue;
                }

                x = xStab;
                y = yStab;

                if (m_tracker->isInitialized() == false) {
                } else {
                    while (m_tracker->isRunning());

                    m_tracker->resetTrack();
                }

                printf("2: Init tracker at (%d,%d)\r\n", x, y);
                cv::Rect roiOpen;
                roiOpen.x       = ((x - m_trackSize) < 0) ? 0 : (x - m_trackSize);
                roiOpen.y       = ((y - m_trackSize) < 0) ? 0 : (y - m_trackSize);
                roiOpen.width   = ((roiOpen.x + 2 * m_trackSize) >= grayFramePrev.cols) ? (grayFramePrev.cols - roiOpen.x) : (2 * m_trackSize);
                roiOpen.height  = ((roiOpen.y + 2 * m_trackSize) >= grayFramePrev.rows) ? (grayFramePrev.rows - roiOpen.y) : (2 * m_trackSize);
                cv::Mat tmpPatch = grayFramePrev(roiOpen);
                cv::Scalar meanVal, stdVal;
                cv::meanStdDev(tmpPatch, meanVal, stdVal);
                printf("mean = %f  std = %f  mean/std = %f\n", meanVal[0], stdVal[0], stdVal[0] / meanVal[0]);

                //        if( (stdVal[0] / meanVal[0]) > 0.002)

                if (!isTextureLess(tmpPatch)) {
                    cv::Mat patchOpen = m_imgStab(roiOpen).clone();
                    //----- Get adjusted image patch by saliency
                    printf("init thresh tracker inside setTrack function\n");
#ifdef MOVE_CLICK_TRACK
                    std::cout << "=============> 1\n";
                    ClickTrackObj->Init(m_imgStab, cv::Point(x, y), 60);
                    std::cout << "=============> 1.2\n";
                    ClickTrackObj->Run();
                    std::cout << "=============> 2\n";
                    bool initTrackResult = false;
                    initTrackResult = ClickTrackObj->foundObject;

                    if (m_trackType == "sKCF") {
                        k_tracker->initTrack(grayFramePrev, ClickTrackObj->object_position);
                    }

                    if (m_trackType == "Thresh") {
                        thresh_tracker->initTrack(grayFramePrev, ClickTrackObj->object_position);
                    }

                    if (m_trackType == "KCF") {
                        std::cout << "=============> 3\n";
                        m_tracker->initTrack(grayFramePrev, ClickTrackObj->object_position);
                        std::cout << "=============> 4\n";
                    }

                    Q_EMIT trackInitSuccess(initTrackResult, x, y, grayFramePrev.cols, grayFramePrev.rows);
#endif
#ifdef USE_SALIENCY
                    cv::Rect roiSaliency = saliency(patchOpen);
                    //            cv::Rect roiSaliency = computeThresMap( patchOpen );
                    roiSaliency.x += roiOpen.x;
                    roiSaliency.y += roiOpen.y;

                    if (roiSaliency.width <= 9) {
                        roiSaliency.width = 9;
                    }

                    if (roiSaliency.height <= 9) {
                        roiSaliency.height = 9;
                    }

                    if (roiSaliency.width >= 9 && roiSaliency.height >= 9) {
                        if (m_trackType == "sKCF") {
                            k_tracker->initTrack(m_imgStab, roiSaliency);
                        }

                        if (m_trackType == "Thresh") {
                            thresh_tracker->initTrack(grayFramePrev, roiSaliency);
                        } else {
                            m_tracker->initTrack(grayFramePrev, roiSaliency);
                        }
                    }

#endif
                } else {
                    Q_EMIT trackInitSuccess(false, x, y, grayFramePrev.cols, grayFramePrev.rows);
                }

                printf("Emit track init done\r\n");
                m_setTrack = false;
            }

            //            printf("Start tracking \n");
            // --------------original work------------------------------------
            if (m_trackType == "KCF") {
                if (m_tracker->isInitialized()) {
                    //
                    m_tracker->performTrack(enhancedFrame);
                    object_position = m_tracker->getPosition();

                    if (m_tracker->trackStatus() == TRACK_INVISION) {
                        cv::rectangle(m_imgStab, object_position, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
                        cv::Point centerTrack(object_position.x + object_position.width / 2,
                                              object_position.y + object_position.height / 2);
                        cv::Point centerFrame(m_imgStab.cols / 2, m_imgStab.rows / 2);
                        cv::line(m_imgStab, centerFrame, centerTrack, cv::Scalar(0, 255, 255), 2);
                        //                    Q_EMIT trackStateFound(centerTrack.x,centerTrack.y,grayFrame.cols,grayFrame.rows,
                        //                                         rectTrack.width,rectTrack.height);
                        //===== IR Zoom control
                        //                    printf("IRFOV: %f ,EOFOV: %f\r\n",m_irFOV,m_eoFOV);
                        std::string zoomDir = m_tracker->getZoomIR(m_irFOV);
                        float deltaIRFov = 0;

                        if (zoomDir == std::string("ZOOM_IN")) {
                            deltaIRFov = -1;
                        } else if (zoomDir == std::string("ZOOM_OUT")) {
                            deltaIRFov = 1;
                        } else {
                            deltaIRFov = 0;
                        }

                        {
                            if (object_position.width > 0 && object_position.height > 0) {
                                Q_EMIT trackStateFound(centerTrack.x, centerTrack.y, grayFrame.cols, grayFrame.rows,
                                                       object_position.width, object_position.height);
                            }
                        }
                    } else {
                        Q_EMIT trackStateLost();
                    }
                }
            }
            //---------------thresholding-------------------
            else if (m_trackType == "Thresh") {
                if (thresh_tracker->isInitialized()) {
                    thresh_tracker->performTrack(enhancedFrame);
                    object_position = thresh_tracker->getPosition();
                    //                    std::cout << "Position of object is " << object_position << std::endl;

                    if (object_position.width > 0 && object_position.height > 0) {
                        Q_EMIT trackStateFound(object_position.x + object_position.width / 2, object_position.y + object_position.height / 2,
                                               grayFrame.cols, grayFrame.rows,
                                               object_position.width, object_position.height);
                    }

                    cv::rectangle(m_imgStab, object_position, cv::Scalar(0, 255, 0));
                }
            } else {
                //---------------sKCF---------------------------
                if (k_tracker->isInititalized()) {
                    k_tracker->processFrame(enhancedFrame);
                    object_position = k_tracker->getPosition();
                    std::cout << "Position of object is " << object_position << std::endl;
                    cv::rectangle(m_imgStab, object_position, cv::Scalar(255, 255, 0));

                    if (object_position.width > 0 && object_position.height > 0) {
                        Q_EMIT trackStateFound(object_position.x + object_position.width / 2, object_position.y + object_position.height / 2,
                                               grayFrame.cols, grayFrame.rows,
                                               object_position.width, object_position.height);
                    }
                }
            }
        }

        auto end = chrono::high_resolution_clock::now();
        auto elapsed = chrono::duration_cast<chrono::milliseconds>(end - start);
        //        std::cout << "Process Time elapsed: " << elapsed.count() << std::endl;
        auto startShow = chrono::high_resolution_clock::now();

        if (m_usingIPC == true && m_detectEnable == true) {
            if (m_imgStab.rows > m_detectSize && m_imgStab.cols > m_detectSize) {
                cv::Point center(m_imgStab.cols / 2, m_imgStab.rows / 2);
                cv::Point rectUp(center.x - m_detectSize / 2, center.y - m_detectSize / 2);
                cv::rectangle(m_imgStab, cv::Rect(rectUp.x, rectUp.y, m_detectSize, m_detectSize), cv::Scalar(255, 255, 0), 2);
            }
        }

        if (m_usingIPC == true && m_stabEnable == true && m_cropRatio < 1) {
            cv::Rect bound(m_imgStab.cols * (1 - m_cropRatio) / 2,
                           m_imgStab.rows * (1 - m_cropRatio) / 2,
                           m_imgStab.cols * m_cropRatio,
                           m_imgStab.rows * m_cropRatio);
            cv::Mat disp_frame = m_imgStab(bound);
            //            m_mutexProcess->lock();
            cv::cvtColor(disp_frame, *m_imgShow, CV_BGR2BGRA);
            //            m_mutexProcess->unlock();
        } else {
            //            m_mutexProcess->lock();
            cv::cvtColor(m_imgStab, *m_imgShow, CV_BGR2BGRA);
            //            m_mutexProcess->unlock();
        }

        if ((m_streamWidth != m_imgShow->cols) || (m_streamHeight != m_imgShow->rows)) {
            m_streamHeight = m_imgShow->rows;
            m_streamWidth = m_imgShow->cols;
            Q_EMIT streamFrameSizeChanged(m_streamWidth, m_streamHeight);
        }

        // draw text
//        cv::putText(*m_imgShow, "Long Range Surveillance", cv::Point(30, 15), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 250), 1, CV_AA);
//        cv::putText(*m_imgShow, "HKVT-Viettel", cv::Point(70, 30), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 250), 1, CV_AA);

        //
        if(m_sharedEnable && m_recordEnable){
            GstBuffer *rtspImage = gst_buffer_new();
            assert(rtspImage != NULL);
            GstMemory *gstMem = gst_allocator_alloc(NULL, m_imgShow->u->size, NULL);
            assert(gstMem != NULL);
            gst_buffer_append_memory(rtspImage, gstMem);
            GstMapInfo mapT;
            gst_buffer_map(rtspImage, &mapT, GST_MAP_READ);
            memcpy((void *)mapT.data, m_imgShow->data, m_imgShow->u->size);
            gst_buffer_unmap(rtspImage , &mapT);

            //add to rtsp
            GstFrameCacheItem gstFrame;
            gstFrame.setIndex(*m_frameID);
            gstFrame.setGstBuffer(rtspImage);
            m_gstRTSPBuff->add(gstFrame);
            //add to saving
            GstFrameCacheItem gstFrameSaving;
            gstFrameSaving.setIndex(*m_frameID);
            gstFrameSaving.setGstBuffer(gst_buffer_copy(rtspImage));
            m_buffVideoSaving->add(gstFrameSaving);
        }else if(m_sharedEnable && !m_recordEnable){
            GstBuffer *rtspImage = gst_buffer_new();
            assert(rtspImage != NULL);
            GstMemory *gstMem = gst_allocator_alloc(NULL, m_imgShow->u->size, NULL);
            assert(gstMem != NULL);
            gst_buffer_append_memory(rtspImage, gstMem);
            GstMapInfo mapT;
            gst_buffer_map(rtspImage, &mapT, GST_MAP_READ);
            memcpy((void *)mapT.data, m_imgShow->data, m_imgShow->u->size);
            gst_buffer_unmap(rtspImage , &mapT);

            //add to rtsp
            GstFrameCacheItem gstFrame;
            gstFrame.setIndex(*m_frameID);
            gstFrame.setGstBuffer(rtspImage);
            m_gstRTSPBuff->add(gstFrame);
        }else if(!m_sharedEnable && m_recordEnable){
            GstBuffer *savingImage = gst_buffer_new();
            assert(savingImage != NULL);
            GstMemory *gstMem = gst_allocator_alloc(NULL, m_imgShow->u->size, NULL);
            assert(gstMem != NULL);
            gst_buffer_append_memory(savingImage, gstMem);
            GstMapInfo mapT;
            gst_buffer_map(savingImage, &mapT, GST_MAP_READ);
            memcpy((void *)mapT.data, m_imgShow->data, m_imgShow->u->size);
            gst_buffer_unmap(savingImage , &mapT);

            //add to saving
            GstFrameCacheItem gstFrameSaving;
            gstFrameSaving.setIndex(*m_frameID);
            gstFrameSaving.setGstBuffer(savingImage);
            m_buffVideoSaving->add(gstFrameSaving);
        }else{

        }

        auto endShow = chrono::high_resolution_clock::now();
        auto elapsedShow = chrono::duration_cast<chrono::milliseconds>(endShow - startShow);
        //            std::cout << "Show Time elapsed: " << elapsedShow.count() << std::endl;
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
