#include "VDisplayWorker.h"

VDisplayWorker::VDisplayWorker(QObject *_parent) : QObject(_parent)
{
    init();
}

VDisplayWorker::~VDisplayWorker() {}

void VDisplayWorker::init()
{
    m_currID = 0;
    std::string names_file = "../Controller/Video/OD/yolo-setup/visdrone2019.names";
    m_objName = this->objects_names_from_file(names_file);
}

void VDisplayWorker::process()
{
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
    float *h_stabMat;
    float *d_stabMat;

    while (true) {
        start = std::chrono::high_resolution_clock::now();
        processImgItem = m_matImageBuff->at(m_matImageBuff->size() - 3);

        if ((processImgItem.getIndex() == -1) ||
            (processImgItem.getIndex() == m_currID)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        m_currID = processImgItem.getIndex();
        d_imageData = processImgItem.getDeviceImage();
        h_imageData = processImgItem.getHostImage();
        imgSize = processImgItem.getImageSize();
        stabMatrix = processImgItem.getStabMatrix();
        // Warp I420 Image GPU
        //        h_stabMat = processImgItem.getHostStabMatrix();
        //        d_stabMat = processImgItem.getDeviceStabMatrix();
        h_BRGAImage = fixedMemBRGA.getHeadHost();
        d_BRGAImage = fixedMemBRGA.getHeadDevice();
        //        assert(gpu_invWarpI420_V2(d_imageData, d_I420Image, d_stabMat, imgSize.width, imgSize.height, imgSize.width, imgSize.height) == cudaSuccess);
        //        assert(gpu_i420ToRGBA(d_I420Image, d_BRGAImage, imgSize.width, imgSize.height, 0, 0, imgSize.width, imgSize.height) == cudaSuccess);
        //        m_imgShow = cv::Mat(imgSize.height, imgSize.width, CV_8UC4, d_BRGAImage);
        /***Warp Image CPU*******/
        m_imgShow = cv::Mat(imgSize.height * 3 / 2, imgSize.width, CV_8UC1, h_imageData);
        cv::cvtColor(m_imgShow, m_imgShow, cv::COLOR_YUV2BGRA_I420);
        //----------------------------- Draw EO object detected
        DetectedObjectsCacheItem detectedObjsItem = m_rbDetectedObjs->getElementById(m_currID);

        if (detectedObjsItem.getIndex() != -1) {
            std::vector<bbox_t> listObj = detectedObjsItem.getDetectedObjects();
            this->drawDetectedObjects(m_imgShow, listObj);
        }

        if (m_enDigitalStab) {
            cv::warpAffine(m_imgShow, m_imgShow, stabMatrix, imgSize, cv::INTER_LINEAR);
        }

        memcpy(h_BRGAImage, m_imgShow.data, imgSize.width * imgSize.height * 4);
        // Adding video saving and rtsp
        GstBuffer *gstBuffRTSP = gst_buffer_new();
        assert(gstBuffRTSP != NULL);
        GstMemory *gstRTSPMem = gst_allocator_alloc(NULL, imgSize.width * imgSize.height * 3 / 2, NULL);
        assert(gstRTSPMem != NULL);
        gst_buffer_append_memory(gstBuffRTSP, gstRTSPMem);
        GstMapInfo mapRTSP;
        gst_buffer_map(gstBuffRTSP, &mapRTSP, GST_MAP_READ);
        memcpy(mapRTSP.data, h_imageData, imgSize.width * imgSize.height * 3 / 2);
        gst_buffer_unmap(gstBuffRTSP , &mapRTSP);
        GstFrameCacheItem gstRTSP;
        gstRTSP.setIndex(m_currID);
        gstRTSP.setGstBuffer(gstBuffRTSP);
        m_gstRTSPBuff->add(gstRTSP);

        if (m_enSaving) {
            GstFrameCacheItem gstEOSaving;
            gstEOSaving.setIndex(m_currID);
            gstEOSaving.setGstBuffer(gst_buffer_copy(gstBuffRTSP));
            m_gstEOSavingBuff->add(gstEOSaving);
        }

        //        frame = QVideoFrame(QImage((uchar *)m_imgShow.data, m_imgShow.cols, m_imgShow.rows, QImage::Format_RGBA8888));
        frame = QVideoFrame(QImage((uchar *)h_BRGAImage, m_imgShow.cols, m_imgShow.rows, QImage::Format_RGBA8888));
        Q_EMIT receivedFrame(m_currID, frame);
        //        Q_EMIT receivedFrame();
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

bool VDisplayWorker::isOnDigitalStab()
{
    return m_enDigitalStab;
}

int VDisplayWorker::setDigitalStab(bool _enStab)
{
    m_enDigitalStab = _enStab;
}

bool VDisplayWorker::getDigitalStab()
{
    return m_enDigitalStab;
}


void VDisplayWorker::drawDetectedObjects(cv::Mat &_img, const std::vector<bbox_t> &_listObj)
{
    for (auto b : _listObj) {
        cv::rectangle(_img, cv::Rect(b.x, b.y, b.w, b.h), cv::Scalar(255, 0, 0), 2);
    }
}
