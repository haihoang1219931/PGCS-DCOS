#include "VDisplayWorker.h"
#include "../VideoEngine/VideoEngineInterface.h"
#include "Payload/Algorithms/tracker/mosse/tracker.h"
VDisplayWorker::VDisplayWorker(QObject *_parent) : QObject(_parent)
{
    qRegisterMetaType< cv::Mat >("cv::Mat");
    init();
}

VDisplayWorker::~VDisplayWorker() {}

void VDisplayWorker::init()
{
    m_currID = 0;
    std::string names_file = "../GPUBased/OD/yolo-setup/visdrone2019.names";
    m_objName = this->objects_names_from_file(names_file);
}

void VDisplayWorker::process()
{
    m_matImageBuff = Cache::instance()->getTrackImageCache();
    m_rbSearchObjs = Cache::instance()->getSearchCache();
    m_rbMOTObjs = Cache::instance()->getMOTCache();
    //    m_rbTrackResEO = Cache::instance()->getEOTrackingCache();
    //    m_rbXPointEO = Cache::instance()->getEOSteeringCache();
    //    m_rbTrackResIR = Cache::instance()->getIRTrackingCache();
    //    m_rbXPointIR = Cache::instance()->getIRSteeringCache();
    //    m_rbSystem = Cache::instance()->getSystemStatusCache();
    //    m_rbIPCEO = Cache::instance()->getMotionImageEOCache();
    //    m_rbIPCIR = Cache::instance()->getMotionImageIRCache();
    m_gstEOSavingBuff = Cache::instance()->getGstEOSavingCache();
    m_gstIRSavingBuff = Cache::instance()->getGstIRSavingCache();
    m_gstRTSPBuff = Cache::instance()->getGstRTSPCache();

    QVideoFrame frame;
    std::chrono::high_resolution_clock::time_point start, stop;
    long sleepTime = 0;
    FixedMemory fixedMemBRGA(5, 1920 * 1080 * 4);
    //    FixedMemory fixedMemI420Stab(5, 1920 * 1080 * 3 / 2);
    unsigned char *d_imageData;
    unsigned char *h_imageData;
    unsigned char *h_BRGAImage;
    unsigned char *d_BRGAImage;
    //    unsigned char *h_I420Image;
    //    unsigned char *d_I420Image;
    //    cudaMalloc(&d_I420Image, 1920 * 1080 * 3 / 2);
    cv::Mat i420Img;
    cv::Mat rgbaImg;
    cv::Size imgSize;
    cv::Mat stabMatrix;
    float* h_stabMat = nullptr;
    float *d_stabMat;
    cv::Scalar line_color(255,0,255,255);
    m_warpDataRender = std::vector<float>(16,0);
    m_warpDataRender[0*4+0] = 1;
    m_warpDataRender[1*4+1] = 1;
    m_warpDataRender[2*4+2] = 1;
    m_warpDataRender[3*4+3] = 1;
    float width = 0;
    float height = 0;
    cv::Mat imgYWarped;
    cv::Mat imgUWarped;
    cv::Mat imgVWarped;
    cv::Mat warpMatrix = cv::Mat(3,3,CV_32FC1);
    while (true) {
        start = std::chrono::high_resolution_clock::now();
        int index = m_matImageBuff->size() - 1;
        if(index < 0 || index >= m_matImageBuff->size()){
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
//        printf("process[%d/%d]\r\n",index,m_matImageBuff->size());
        ProcessImageCacheItem& processImgItem = m_matImageBuff->at(index);
        if ((processImgItem.getIndex() == -1) ||
                (processImgItem.getIndex() == m_currID)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        m_currID = processImgItem.getIndex();
        d_imageData = processImgItem.getDeviceImage();
        h_imageData = processImgItem.getHostImage();
        imgSize = processImgItem.getImageSize();

        if(imgSize.width <=0 || imgSize.height <=0){
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        width = imgSize.width;
        height = imgSize.height;
        // Warp I420 Image GPU
        h_stabMat = processImgItem.getHostStabMatrix();
        if(h_stabMat != nullptr)
            stabMatrix = cv::Mat(3,3,CV_32FC1,h_stabMat);
        //        d_stabMat = processImgItem.getDeviceStabMatrix();
        h_BRGAImage = fixedMemBRGA.getHeadHost();
        d_BRGAImage = fixedMemBRGA.getHeadDevice();
        //        assert(gpu_invWarpI420_V2(d_imageData, d_I420Image, d_stabMat, imgSize.width, imgSize.height, imgSize.width, imgSize.height) == cudaSuccess);
        //        assert(gpu_i420ToRGBA(d_I420Image, d_BRGAImage, imgSize.width, imgSize.height, 0, 0, imgSize.width, imgSize.height) == cudaSuccess);
        //        m_imgShow = cv::Mat(imgSize.height, imgSize.width, CV_8UC4, d_BRGAImage);
        /***Warp Image CPU*******/
//        printf("imgSize[%dx%d]\r\n",imgSize.width,imgSize.height);

        m_imgI420 = cv::Mat(imgSize.height * 3 / 2, imgSize.width, CV_8UC1, h_imageData);
        if(m_imgI420Warped.rows <= 0 || m_imgI420Warped.cols <= 0){
            m_imgI420Warped = m_imgI420.clone();
            imgYWarped = cv::Mat(static_cast<int>(height), static_cast<int>(width), CV_8UC1, m_imgI420Warped.data);
            imgUWarped = cv::Mat(static_cast<int>(height/2), static_cast<int>(width/2), CV_8UC1, m_imgI420Warped.data + size_t(height * width));
            imgVWarped = cv::Mat(static_cast<int>(height/2), static_cast<int>(width/2), CV_8UC1, m_imgI420Warped.data + size_t(height * width * 5 / 4));
        }
        m_imgGray = cv::Mat(imgSize.height, imgSize.width, CV_8UC1, h_imageData);
        // draw zoom
        char zoomText[100];
        sprintf(zoomText,"zoom: %.02f\r\n",processImgItem.getZoom());
//        cv::putText(m_imgShow,zoomText,cv::Point(100,100),
//                    cv::FONT_HERSHEY_COMPLEX, 1.2, cv::Scalar(0, 0, 0,255), 2);


        if(processImgItem.sensorID() == 1){
            if(processImgItem.colorMode() == 0){

            }else if(processImgItem.colorMode() == 0){
                cv::applyColorMap(m_imgGray,m_imgIRColor,cv::COLORMAP_HOT);
                cv::cvtColor(m_imgIRColor, m_imgI420, cv::COLOR_BGR2YUV_I420);
            }
        }
        if(m_captureSet){
            m_captureMutex.lock();
            m_captureSet = false;
            m_captureMutex.unlock();
            std::string timestamp = FileController::get_time_stamp();
            std::string captureFile = "flights/" + timestamp + ".jpg";
            printf("Save file %s\r\n", captureFile.c_str());
            cv::Mat imgSave;
            cv::cvtColor(m_imgI420, imgSave, cv::COLOR_YUV2BGR_I420);
            cv::imwrite(captureFile, imgSave);
        }

        if(stabMatrix.rows == 3 && stabMatrix.cols == 3 &&
                m_imgI420.cols > 0 && m_imgI420.rows > 0){
            warpMatrix.at<float>(0,0) = stabMatrix.at<float>(0,0);
            warpMatrix.at<float>(0,1) = stabMatrix.at<float>(0,1);
            warpMatrix.at<float>(0,2) = stabMatrix.at<float>(0,2);
            warpMatrix.at<float>(1,0) = stabMatrix.at<float>(1,0);
            warpMatrix.at<float>(1,1) = stabMatrix.at<float>(1,1);
            warpMatrix.at<float>(1,2) = stabMatrix.at<float>(1,2);
            cv::Mat imgY(static_cast<int>(height), static_cast<int>(width), CV_8UC1, m_imgI420.data);
            cv::Mat imgU(static_cast<int>(height/2), static_cast<int>(width/2), CV_8UC1, m_imgI420.data + size_t(height * width));
            cv::Mat imgV(static_cast<int>(height/2), static_cast<int>(width/2), CV_8UC1, m_imgI420.data + size_t(height * width * 5 / 4));
            cv::warpAffine(imgY,imgYWarped,warpMatrix(cv::Rect(0,0,3,2)),cv::Size(imgY.cols,imgY.rows),cv::INTER_LINEAR);
            warpMatrix.at<float>(0,2) = stabMatrix.at<float>(0,2)/2;
            warpMatrix.at<float>(1,2) = stabMatrix.at<float>(1,2)/2;
            cv::warpAffine(imgU,imgUWarped,warpMatrix(cv::Rect(0,0,3,2)),cv::Size(imgU.cols,imgU.rows),cv::INTER_LINEAR);
            cv::warpAffine(imgV,imgVWarped,warpMatrix(cv::Rect(0,0,3,2)),cv::Size(imgV.cols,imgV.rows),cv::INTER_LINEAR);
            if(m_enOD){
                if(m_countUpdateOD == 0){
                    //----------------------------- Draw EO object detected
                    DetectedObjectsCacheItem& detectedObjsItem = m_rbSearchObjs->last();

                    if(abs(
                        static_cast<int>(detectedObjsItem.getDetectedObjects().size() - 0)) > 1 | m_listObj.size() == 0)
                    m_listObj = detectedObjsItem.getDetectedObjects();
                    for(int i=0; i< m_listObj.size(); i++){
                        bbox_t objBeforeStab = m_listObj[i];
                        cv::Point pointAfterStab = convertPoint(cv::Point(objBeforeStab.x+objBeforeStab.w/2,
                                                                          objBeforeStab.y+objBeforeStab.h/2),
                                                                stabMatrix);
                        m_listObj[i].x = pointAfterStab.x - objBeforeStab.w/2;
                        m_listObj[i].y = pointAfterStab.y - objBeforeStab.w/2;
                    }
                }
                this->drawDetectedObjects(imgYWarped,imgUWarped,imgVWarped, m_listObj);
                m_countUpdateOD ++;
                if(m_countUpdateOD > 15){
                    m_countUpdateOD = 0;
                }
            }
            // draw track
            cv::Rect trackRect = processImgItem.trackRect();
            cv::Point pointAfterStab = convertPoint(cv::Point(trackRect.x+trackRect.width/2,
                                                                  trackRect.y+trackRect.height/2),
                                                        stabMatrix);
            trackRect.x = pointAfterStab.x - trackRect.width/2;
            trackRect.y = pointAfterStab.y - trackRect.height/2;
            cv::Scalar colorInvision(255,0,0);
            cv::Scalar colorOccluded(0,0,0);
            if(processImgItem.lockMode() == 1){
                if(processImgItem.trackStatus() == TRACK_INVISION){
                    VideoEngine::rectangle(imgYWarped,imgUWarped,imgVWarped,
                                           trackRect,colorInvision,2);
                }else if(processImgItem.trackStatus() == TRACK_OCCLUDED){
                    VideoEngine::rectangle(imgYWarped,imgUWarped,imgVWarped,
                                           trackRect,colorOccluded,2);
                }else{

                }
            }else if(processImgItem.lockMode() == 2){
                if(processImgItem.trackStatus() == TRACK_INVISION){
                    VideoEngine::drawSteeringCenter(imgYWarped,imgUWarped,imgVWarped,
                                        trackRect.width,
                                       static_cast<int>(trackRect.x + trackRect.width/2),
                                       static_cast<int>(trackRect.y + trackRect.height/2),
                                       colorInvision);
                }else if(processImgItem.trackStatus() == TRACK_OCCLUDED){
                    VideoEngine::drawSteeringCenter(imgYWarped,imgUWarped,imgVWarped,
                                        trackRect.width,
                                       static_cast<int>(trackRect.x + trackRect.width/2),
                                       static_cast<int>(trackRect.y + trackRect.height/2),
                                       colorOccluded);
                }else{

                }
            }
            // draw powerline
            if(processImgItem.powerlineDetectEnable()){
                cv::Rect rect = processImgItem.powerlineDetectRect();
                VideoEngine::rectangle(imgYWarped,imgUWarped,imgVWarped,
                                       rect,
                              cv::Scalar(0,255,0,255),2);
                vector<cv::Scalar> plr_lines = processImgItem.powerLineList();
                for(int i = 0;i < (int)plr_lines.size();i ++){
                    cv::Point2d pt1(plr_lines[i].val[0],plr_lines[i].val[1]);
                    {
                        cv::Point pointAfterStab = convertPoint(pt1,stabMatrix);
                        pt1 = pointAfterStab;
                    }
                    cv::Point2d pt2(plr_lines[i].val[2],plr_lines[i].val[3]);
                    {
                        cv::Point pointAfterStab = convertPoint(pt2,stabMatrix);
                        pt2 = pointAfterStab;
                    }
                    VideoEngine::line(imgYWarped,imgUWarped,imgVWarped,
                                      pt1,pt2,line_color,4,cv::LINE_AA);
                }
            }
            Q_EMIT readyDrawOnRenderID(0,m_imgI420Warped.data,width,height,m_warpDataRender.data(),nullptr);
            Q_EMIT readyDrawOnRenderID(1,m_imgI420Warped.data,width,height,m_warpDataRender.data(),nullptr);
            // Adding video saving and rtsp
            if(m_enShare &&
                    (m_imgI420Warped.cols > 0 && m_imgI420Warped.rows > 0)){
                GstBuffer *gstBuffRTSP = gst_buffer_new();
                assert(gstBuffRTSP != NULL);
                GstMemory *gstRTSPMem = gst_allocator_alloc(NULL, imgSize.width * imgSize.height * 3 / 2, NULL);
                assert(gstRTSPMem != NULL);
                gst_buffer_append_memory(gstBuffRTSP, gstRTSPMem);
                GstMapInfo mapRTSP;
                gst_buffer_map(gstBuffRTSP, &mapRTSP, GST_MAP_READ);
                memcpy(mapRTSP.data, m_imgI420Warped.data, m_imgI420Warped.cols * m_imgI420Warped.rows);
                gst_buffer_unmap(gstBuffRTSP , &mapRTSP);
                GstFrameCacheItem gstRTSP;
                gstRTSP.setIndex(m_currID);
                gstRTSP.setGstBuffer(gstBuffRTSP);
                m_gstRTSPBuff->add(gstRTSP);
            }
        }

        //-----------------------------
        fixedMemBRGA.notifyAddOne();
        //        fixedMemI420Stab.notifyAddOne();
        stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> timeSpan = stop - start;
        sleepTime = (long)(33333 - timeSpan.count());
//        printf("\nDisplay Worker: %d | %d - [%d - %d]", m_currID, (long)(timeSpan.count()), imgSize.width, imgSize.height);
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }
}
cv::Point VDisplayWorker::convertPoint(cv::Point originPoint, cv::Mat stabMatrix){
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

void VDisplayWorker::setVideoSavingState(bool _state)
{
    m_enSaving = _state;
}

void VDisplayWorker::capture(){
    m_captureMutex.lock();
    m_captureSet = true;
    m_captureMutex.unlock();
}

void VDisplayWorker::drawDetectedObjects(cv::Mat &imgY,cv::Mat &imgU,cv::Mat &imgV,
                                         const std::vector<bbox_t> &_listObj)
{
    unsigned int limitW = 30;
    unsigned int limitH = 60;
    for (auto b : _listObj) {
        cv::Rect rectObject = cv::Rect(
                    static_cast<int>(b.x),
                    static_cast<int>(b.y),
                    static_cast<int>(b.w),
                    static_cast<int>(b.h)
                    );
        if((b.w * b.h) < (limitW * limitH) ){
            continue;
        }
        VideoEngine::rectangle(imgY,imgU,imgV, rectObject, cv::Scalar(255, 0, 0,255), 2);
        {
            std::string obj_name;
            std::string string_id(b.track_info.stringinfo);
            if (string_id.empty())
                obj_name = std::string();
            else
                obj_name = /*std::to_string(i.track_id) + " - " + */ string_id;

            cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, nullptr);
            //			int max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
            //			max_width = std::max(max_width, (int)i.w + 2);
            int max_width = text_size.width;

            if (!obj_name.empty()){
                cv::Rect rectName(cv::Point2f(std::max((int)b.x - 1, 0), std::max((int)b.y - 35, 0)),
                                  cv::Point2f(std::min((int)b.x + max_width, imgY.cols - 1), std::min((int)b.y, imgY.rows - 1)));
                VideoEngine::rectangle(imgY,imgU,imgV,rectName,
                              cv::Scalar(255, 255, 255,255), CV_FILLED, 8, 0);
                VideoEngine::putText(imgY,imgU,imgV,
                            obj_name, cv::Point2f(b.x, b.y - 16), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0,255), 2);
                //                printf("obj_name = %s\r\n",obj_name.c_str());

                QString qstrObjName = QString::fromStdString(obj_name);
                qstrObjName.replace("/","-");
                std::string timestampt = FileController::get_time_stamp();
                bool writeLog = false;
                if(m_mapPlates.keys().contains(qstrObjName)){
                    if(!(m_mapPlates[qstrObjName] == QString::fromStdString(timestampt))){
                        writeLog = true;
                    }
                }else{
                    writeLog = true;
                }
                if(writeLog){
                    m_mapPlates[qstrObjName] = QString::fromStdString(timestampt);
                    std::string fileName = timestampt+"_"+qstrObjName.toStdString()+".jpg";
                    cv::imwrite("plates/"+fileName,imgY(rectObject));
                    std::string lineLog = timestampt+","+qstrObjName.toStdString()+","+fileName;
                    FileController::addLine("plates/plate_log.csv",lineLog);
                    if(m_plateLog != nullptr){
                        m_plateLog->appendLogFile("plates/plate_log.csv",QString::fromStdString(lineLog));
                    }
                }
            }
        }
    }
}

