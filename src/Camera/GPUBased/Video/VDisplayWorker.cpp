#include "VDisplayWorker.h"

VDisplayWorker::VDisplayWorker(QObject *_parent) : QObject(_parent)
{
    init();
}

VDisplayWorker::~VDisplayWorker() {}

void VDisplayWorker::init()
{
    m_currID = 0;
    std::string names_file = "../src/Camera/GPUBased/Video/Multitracker/vehicle-weight/visdrone2019.names";
    m_objName = this->objects_names_from_file(names_file);
}

void VDisplayWorker::process()
{
    m_gstRTSPBuff = Cache::instance()->getGstRTSPCache();
    m_matImageBuff = Cache::instance()->getProcessImageCache();
    m_rbTrackResEO = Cache::instance()->getEOTrackingCache();
    m_rbXPointEO = Cache::instance()->getEOSteeringCache();
    m_rbTrackResIR = Cache::instance()->getIRTrackingCache();
    m_rbXPointIR = Cache::instance()->getIRSteeringCache();
    m_rbDetectedObjs = Cache::instance()->getDetectedObjectsCache();
    m_rbSystem = Cache::instance()->getSystemStatusCache();
    m_rbIPCEO = Cache::instance()->getMotionImageEOCache();
    m_rbIPCIR = Cache::instance()->getMotionImageIRCache();
    m_gstEOSavingBuff = Cache::instance()->getGstEOSavingCache();
    m_gstIRSavingBuff = Cache::instance()->getGstIRSavingCache();
    ProcessImageCacheItem processImage;
    QVideoFrame frame;
    std::chrono::high_resolution_clock::time_point start, stop;
    long sleepTime = 0;
    cv::Size img4KSize(3840, 2160);
//    FixedPinnedMemory fixedMemStream(5, 1920 * 1080 * 3/2);
    unsigned char *tempMem;
    cudaMalloc(&tempMem, 1920 * 1080 * 3/2);

    while (true) {
        start = std::chrono::high_resolution_clock::now();
        processImage = m_matImageBuff->at(m_matImageBuff->size() - 7);

        if ((processImage.getIndex() == 0) ||
            (processImage.getIndex() == m_currID)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        m_currID = processImage.getIndex();
        uchar *proccImgData = processImage.getHostImage();
        uchar *d_proccImgData = processImage.getDeviceImage();
        cv::Size imgSize = processImage.getImageSize();
        m_imgShow = cv::Mat(imgSize.height, imgSize.width, CV_8UC4, proccImgData);
        Eye::SystemStatus systemStatus = m_rbSystem->getElementById(m_currID);

        if (systemStatus.getIndex() == m_currID) {
            //            if ((Status::SensorMode)systemStatus.getIPCStatus().getSensorMode() == Status::SensorMode::EO)
            {
                //----------------------------- Draw EO object detected
                DetectedObjectsCacheItem detectedObjsItem = m_rbDetectedObjs->getElementById(m_currID);

                if (detectedObjsItem.getIndex() != 0) {
                    std::vector<bbox_t> listObj = detectedObjsItem.getDetectedObjects();
                    this->drawDetectedObjects(m_imgShow, listObj);
                }

                //----------------------------- Draw EO steering, tracking object
                if (Status::LockMode::LOCK_TRACK ==
                    (Status::LockMode)systemStatus.getIPCStatus().getLockMode()) {
                    Eye::TrackResponse trackRes = m_rbTrackResEO->getElementById(m_currID);

                    if (trackRes.getIndex() == m_currID) {
                        trackRes.setTrackResponse(trackRes.getIndex(),
                                                  trackRes.getPx() * (double)imgSize.width / 2.0, trackRes.getPy() * (double)imgSize.height / 2.0,
                                                  (double)imgSize.width, (double)imgSize.height,
                                                  trackRes.getObjWidth() * (double)imgSize.width / 2.0, trackRes.getObjHeight() * (double)imgSize.height / 2.0);
                        float trackCenterX =
                            (float)(trackRes.getPx() + trackRes.getWidth() / 2.0f),
                            trackCenterY =
                                (float)(trackRes.getPy() + trackRes.getHeight() / 2.0f);
                        //                        printf("\nTrackObj %d - [%f, %f, %f, %f, %f, %f] - [%d, %d]",
                        //                               m_currID, trackRes.getPx(), trackRes.getPy(), trackRes.getObjWidth(), trackRes.getObjHeight(),
                        //                               trackRes.getObjWidth(), trackRes.getHeight(), imgSize.width, imgSize.height);
                        this->drawObjectBoundary(
                            m_imgShow,
                            cv::Rect(trackCenterX - trackRes.getObjWidth() / 2,
                                     trackCenterY - trackRes.getObjHeight() / 2,
                                     trackRes.getObjWidth(), trackRes.getObjHeight()),
                            cv::Scalar(0, 255, 255));
                    }
                } else if (Status::LockMode::LOCK_VISUAL ==
                           (Status::LockMode)systemStatus.getIPCStatus().getLockMode()) {
                    Eye::XPoint xPoint = m_rbXPointEO->getElementById(m_currID);

//                    printf("\nSteering %d - %d - %d", xPoint.getIndex(), m_currID, m_rbXPointEO->last().getIndex());
                    if (xPoint.getIndex() == m_currID) {
                        xPoint.setLocation(xPoint.getPx() * (double)imgSize.width / 2.0, xPoint.getPy() * (double)imgSize.height / 2.0);
                        xPoint.setRegion((double)imgSize.width , (double)imgSize.height);
                        float steerCenterX =
                            (float)(xPoint.getPx() + xPoint.getWidth() / 2.0f),
                            steerCenterY =
                                (float)(xPoint.getPy() + xPoint.getHeight() / 2.0f);
//                        printf("\nSteeringPoint %d - [%f, %f, %f, %f,] - [%d, %d]",
//                               m_currID, xPoint.getPx(), xPoint.getPy(),
//                               xPoint.getWidth(), xPoint.getHeight(), imgSize.width, imgSize.height);
                        drawSteeringCenter(m_imgShow, 100, steerCenterX, steerCenterY,
                                           cv::Scalar(0, 255, 0));
                    }
                } else if (Status::LockMode::LOCK_OFF ==
                           (Status::LockMode)systemStatus.getIPCStatus().getLockMode()) {
                } else {
                }
            }
        }

        //------- Add RTSP server data
        if(imgSize.width != 1920 || imgSize.height != 1080){
            cv::Mat imgFullHD;
            cv::resize(m_imgShow, imgFullHD, cv::Size(1920, 1080));
            GstBuffer *gstBuffRTSP = gst_buffer_new( );
            assert( gstBuffRTSP != NULL );
            GstMemory *gstRTSPMem = gst_allocator_alloc( NULL, imgFullHD.rows * imgFullHD.cols * 4, NULL );
            assert( gstRTSPMem != NULL );
            gst_buffer_append_memory( gstBuffRTSP, gstRTSPMem );

            GstMapInfo mapRTSP;
            gst_buffer_map(gstBuffRTSP, &mapRTSP, GST_MAP_READ);
            memcpy(mapRTSP.data, imgFullHD.data, imgFullHD.rows * imgFullHD.cols * 4);
            gst_buffer_unmap(gstBuffRTSP ,&mapRTSP);

            GstFrameCacheItem gstRTSP;
            gstRTSP.setIndex(m_currID);
            gstRTSP.setGstBuffer(gstBuffRTSP);
            m_gstRTSPBuff->add(gstRTSP);

            if(m_enSaving){
                GstFrameCacheItem gstIRSaving;
                gstIRSaving.setIndex(m_currID);
                gstIRSaving.setGstBuffer(gst_buffer_copy(gstBuffRTSP));
                m_gstEOSavingBuff->add(gstIRSaving);
            }
        }else{
            GstBuffer *gstBuffRTSP = gst_buffer_new( );
            assert( gstBuffRTSP != NULL );
            GstMemory *gstRTSPMem = gst_allocator_alloc( NULL, imgSize.width * imgSize.height * 4, NULL );
            assert( gstRTSPMem != NULL );
            gst_buffer_append_memory( gstBuffRTSP, gstRTSPMem );

            GstMapInfo mapRTSP;
            gst_buffer_map(gstBuffRTSP, &mapRTSP, GST_MAP_READ);
            memcpy(mapRTSP.data, m_imgShow.data, imgSize.width * imgSize.height * 4);
            gst_buffer_unmap(gstBuffRTSP ,&mapRTSP);

            GstFrameCacheItem gstRTSP;
            gstRTSP.setIndex(m_currID);
            gstRTSP.setGstBuffer(gstBuffRTSP);
            m_gstRTSPBuff->add(gstRTSP);
            if(m_enSaving){
                GstFrameCacheItem gstEOSaving;
                gstEOSaving.setIndex(m_currID);
                gstEOSaving.setGstBuffer(gst_buffer_copy(gstBuffRTSP));
                m_gstEOSavingBuff->add(gstEOSaving);
            }
        }

        //-----------------------------
        start = std::chrono::high_resolution_clock::now();
        frame = QVideoFrame(QImage((uchar *)m_imgShow.data, m_imgShow.cols, m_imgShow.rows, QImage::Format_RGBA8888));
        Q_EMIT receivedFrame(m_currID, frame);
        //        Q_EMIT receivedFrame();
        stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> timeSpan = stop - start;
        sleepTime = (long)(33333 - timeSpan.count());
//        printf("\nDisplay Worker: %d | %d - [%d - %d]", m_currID, (long)(timeSpan.count()), imgSize.width, imgSize.height);

        if (sleepTime > 1000) {
            std::this_thread::sleep_for(std::chrono::microseconds(sleepTime));
        }
    }
}
void VDisplayWorker::drawDetectedObjects(
    cv::Mat &_img, const std::vector<bbox_t> &m_listObj)
{
    this->draw_boxes_center(_img, m_listObj, m_objName);
}
void VDisplayWorker::drawSteeringCenter(cv::Mat &_img, int _wBoundary,
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
void VDisplayWorker::drawObjectBoundary(cv::Mat &_img, cv::Rect _objBoundary,
                                        cv::Scalar _color)
{
    cv::rectangle(_img, _objBoundary, _color, 2);
}
void VDisplayWorker::drawCenter(cv::Mat &_img, int _r, int _centerX,
                                int _centerY, cv::Scalar _color)
{
    cv::line(_img, cv::Point(_centerX, _centerY - _r),
             cv::Point(_centerX - _r, _centerY), _color, 2);
    cv::line(_img, cv::Point(_centerX - _r, _centerY),
             cv::Point(_centerX, _centerY + _r), _color, 2);
    cv::line(_img, cv::Point(_centerX, _centerY + _r),
             cv::Point(_centerX + _r, _centerY), _color, 2);
    cv::line(_img, cv::Point(_centerX + _r, _centerY),
             cv::Point(_centerX, _centerY - _r), _color, 2);
}
std::vector<std::string>
VDisplayWorker::objects_names_from_file(std::string const filename)
{
    std::ifstream file(filename);
    std::vector<std::string> file_lines;

    if (!file.is_open()) {
        return file_lines;
    }

    for (std::string line; getline(file, line);) {
        file_lines.push_back(line);
    }

    std::cout << "object names loaded \n";
    return file_lines;
}

void VDisplayWorker::draw_boxes_center(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names, float fps)
{
    for (auto &i : result_vec) {
        bool isExisted = checkIDExisted(i.obj_id);

        if (!isExisted) {
            continue;
        }

        cv::Scalar color(0, 255, 0);
        //		cv::Scalar color = obj_id_to_color(1);
        cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 2);
        //        if (obj_names.size() > i.obj_id)
        {
            //            std::string obj_name = obj_names[i.obj_id];
            //            if (i.track_id > 0) obj_name += " - " + std::to_string(i.track_id);
            std::string obj_name = obj_names[i.obj_id];
            //			  std::string string_id(i.string_id);
            //            std::string string_id(i.track_info->stringinfo);
            //            if (string_id.empty()) {
            //                obj_name = std::string();
            //                //                obj_name = std::to_string(i.track_id);
            //            } else {
            //                obj_name = string_id;
            //            }
            cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
            int max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
            max_width = std::max(max_width, (int)i.w + 2);
            //max_width = std::max(max_width, 283);
            //            if (!obj_name.empty()) {
            cv::rectangle(mat_img, cv::Point2f(std::max((int)i.x - 1, 0), std::max((int)i.y - 35, 0)),
                          cv::Point2f(std::min((int)i.x + max_width, mat_img.cols - 1), std::min((int)i.y, mat_img.rows - 1)),
                          color, CV_FILLED, 8, 0);
            putText(mat_img, obj_name, cv::Point2f(i.x, i.y - 16), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0), 2);
            //            }
        }
    }

    if (fps >= 0) {
        std::string fps_str = "FPS: " + std::to_string(fps);
        putText(mat_img, fps_str, cv::Point2f(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(50, 255, 0), 2);
    }
}

index_type VDisplayWorker::readBarcode(const cv::Mat &_rgbImg)
{
    index_type res = 0;
    vector<decodedObject> decodedObjects;
    int zbarWidth = 130;
    int zbarHeight = 32;
    cv::Rect rectZBar(0, 0, zbarWidth, zbarHeight);
    cv::Mat zbarMat = _rgbImg(rectZBar);
    //  cv::imwrite("img / 1barcode_"+std::to_string(m_currID)+".png", zbarMat);
    ZbarLibs::decode(zbarMat, decodedObjects);

    if (decodedObjects.size() > 0) {
        std::string frameID =
            decodedObjects[0].data.substr(0, decodedObjects[0].data.length() - 1);
        res = (index_type)std::atoi(frameID.c_str());
        //    printf("\nBarcode str: % s | % d", decodedObjects.at(0).data.c_str(),
        //    res);
    }

    return res;
}


void VDisplayWorker::setListObjClassID(std::vector<int> _listObjClassID)
{
    m_listObjClassID = _listObjClassID;
}

void VDisplayWorker::setVideoSavingState(bool _state)
{
    m_enSaving = _state;
}

bool VDisplayWorker::checkIDExisted(int _idx)
{
    bool res = false;

    for (int i = 0; i < m_listObjClassID.size(); i++) {
        if (_idx == m_listObjClassID.at(i)) {
            res = true;
            break;
        }
    }

    return res;
}
