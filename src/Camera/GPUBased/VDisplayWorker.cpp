#include "VDisplayWorker.h"
Q_DECLARE_METATYPE(cv::Mat)
#include "tracker/mosse/tracker.h"
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
    ProcessImageCacheItem processImgItem;
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
    while (true) {
        start = std::chrono::high_resolution_clock::now();
        int index = m_matImageBuff->size() - 1;
        if(index < 0 || index >= m_matImageBuff->size()){
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
//        printf("process[%d/%d]\r\n",index,m_matImageBuff->size());
        processImgItem = m_matImageBuff->at(index);
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
        m_imgShow = cv::Mat(imgSize.height * 3 / 2, imgSize.width, CV_8UC1, h_imageData);
        // draw zoom
        char zoomText[100];
        sprintf(zoomText,"zoom: %.02f\r\n",processImgItem.getZoom());
//        cv::putText(m_imgShow,zoomText,cv::Point(100,100),
//                    cv::FONT_HERSHEY_COMPLEX, 1.2, cv::Scalar(0, 0, 0,255), 2);

        cv::cvtColor(m_imgShow, m_imgShow, cv::COLOR_YUV2BGRA_I420);
        if(m_captureSet){
            m_captureMutex.lock();
            m_captureSet = false;
            m_captureMutex.unlock();
            std::string timestamp = FileController::get_time_stamp();
            std::string captureFile = "flights/" + timestamp + ".jpg";
            printf("Save file %s\r\n", captureFile.c_str());
            cv::imwrite(captureFile, m_imgShow);
        }

        if(stabMatrix.rows == 3 && stabMatrix.cols ==3 &&
                m_imgShow.cols > 0 && m_imgShow.rows > 0){
            cv::Mat warpMatrix = cv::Mat(2,3,CV_32FC1,h_stabMat);
            cv::warpAffine(m_imgShow, m_imgShow, warpMatrix, imgSize, cv::INTER_LINEAR);
            if(m_enOD){
                if(m_countUpdateOD == 0){
                    //----------------------------- Draw EO object detected
                    DetectedObjectsCacheItem detectedObjsItem = m_rbSearchObjs->last();

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
                this->drawDetectedObjects(m_imgShow, m_listObj);
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
            cv::Scalar colorInvision(0,0,255,255);
            cv::Scalar colorOccluded(0,0,0,255);
            if(processImgItem.lockMode() == "TRACK"){
                if(processImgItem.trackStatus() == TRACK_INVISION){
                    cv::rectangle(m_imgShow,trackRect,colorInvision,2);
                }else if(processImgItem.trackStatus() == TRACK_OCCLUDED){
                    cv::rectangle(m_imgShow,trackRect,colorOccluded,2);
                }else{

                }
            }else if(processImgItem.lockMode() == "VISUAL"){
                if(processImgItem.trackStatus() == TRACK_INVISION){
                    drawSteeringCenter(m_imgShow,trackRect.width,
                                       static_cast<int>(trackRect.x + trackRect.width/2),
                                       static_cast<int>(trackRect.y + trackRect.height/2),
                                       colorInvision);
                }else if(processImgItem.trackStatus() == TRACK_OCCLUDED){
                    drawSteeringCenter(m_imgShow,trackRect.width,
                                       static_cast<int>(trackRect.x + trackRect.width/2),
                                       static_cast<int>(trackRect.y + trackRect.height/2),
                                       colorOccluded);
                }else{

                }
            }
            // draw powerline
            if(processImgItem.powerlineDetectEnable()){
                cv::Rect rect = processImgItem.powerlineDetectRect();
                cv::rectangle(m_imgShow,rect,
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
                    cv::line(m_imgShow,pt1,pt2,line_color,4,cv::LINE_AA);
                }
            }
        }

        memcpy(h_BRGAImage, m_imgShow.data, imgSize.width * imgSize.height * 4);
        frame = QVideoFrame(QImage((uchar *)h_BRGAImage, m_imgShow.cols, m_imgShow.rows, QImage::Format_RGBA8888));
        Q_EMIT receivedFrame(m_currID, frame);
        Q_EMIT readyDrawOnViewerID(m_imgShow,0);
        // Adding video saving and rtsp
        if(m_enShare &&
                (m_imgShow.cols > 0 && m_imgShow.rows > 0)){
            cv::Mat imgRTSP;
            cv::cvtColor(m_imgShow, imgRTSP, cv::COLOR_BGRA2YUV_I420);
            GstBuffer *gstBuffRTSP = gst_buffer_new();
            assert(gstBuffRTSP != NULL);
            GstMemory *gstRTSPMem = gst_allocator_alloc(NULL, imgSize.width * imgSize.height * 3 / 2, NULL);
            assert(gstRTSPMem != NULL);
            gst_buffer_append_memory(gstBuffRTSP, gstRTSPMem);
            GstMapInfo mapRTSP;
            gst_buffer_map(gstBuffRTSP, &mapRTSP, GST_MAP_READ);
            memcpy(mapRTSP.data, imgRTSP.data, imgRTSP.cols * imgRTSP.rows);
            gst_buffer_unmap(gstBuffRTSP , &mapRTSP);
            GstFrameCacheItem gstRTSP;
            gstRTSP.setIndex(m_currID);
            gstRTSP.setGstBuffer(gstBuffRTSP);
            m_gstRTSPBuff->add(gstRTSP);
        }
        //-----------------------------
        fixedMemBRGA.notifyAddOne();
        //        fixedMemI420Stab.notifyAddOne();
        stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> timeSpan = stop - start;
        sleepTime = (long)(33333 - timeSpan.count());
//        printf("\nDisplay Worker: %d | %d - [%d - %d]", m_currID, (long)(timeSpan.count()), imgSize.width, imgSize.height);
        std::this_thread::sleep_for(std::chrono::microseconds(sleepTime));
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

void VDisplayWorker::drawDetectedObjects(cv::Mat &_img, const std::vector<bbox_t> &_listObj)
{
    cv::Mat tmpImg = _img.clone();
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
        cv::rectangle(_img, rectObject, cv::Scalar(255, 0, 0,255), 2);
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
                                  cv::Point2f(std::min((int)b.x + max_width, _img.cols - 1), std::min((int)b.y, _img.rows - 1)));
                cv::rectangle(_img,rectName,
                              cv::Scalar(255, 255, 255,255), CV_FILLED, 8, 0);
                cv::putText(_img, obj_name, cv::Point2f(b.x, b.y - 16), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0,255), 2);
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
                    cv::imwrite("plates/"+fileName,tmpImg(rectObject));
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

