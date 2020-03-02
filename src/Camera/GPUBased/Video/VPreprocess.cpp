#include "VPreprocess.h"

VPreprocess::VPreprocess()
{
    m_gstFrameBuff = Cache::instance()->getGstFrameCache();
    m_matImageBuff = Cache::instance()->getProcessImageCache();
    m_currID = 0;
}

VPreprocess::~VPreprocess() {}

void VPreprocess::run()
{
    GstFrameCacheItem gstFrame;
    std::chrono::high_resolution_clock::time_point start, stop;
    long sleepTime = 0;
    FixedPinnedMemory fixedMem(20, 1920 * 1080 * 4);
    cv::Mat brgaImg;

    while (true) {
        start = std::chrono::high_resolution_clock::now();
        gstFrame = m_gstFrameBuff->last();

        if ((gstFrame.getIndex() == -1) || (gstFrame.getIndex() == m_currID)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        m_currID = gstFrame.getIndex();
        GstBuffer *gstData = gstFrame.getGstBuffer();
        int dataSize = (int)gst_buffer_get_size(gstData);
        cv::Size imgSize = getImageSize(dataSize / 4);

        if (imgSize.width == 0 || imgSize.height == 0) {
            continue;
        }

        GstMapInfo map;
        gst_buffer_map(gstData, &map, GST_MAP_READ);
        brgaImg = cv::Mat(imgSize.height, imgSize.width, CV_8UC4, map.data);
        unsigned char *h_image = fixedMem.getHeadHost();
        unsigned char *d_image = fixedMem.getHeadDevice();
        ProcessImageCacheItem processImgItem;
        index_type frameID = this->readBarcode(brgaImg);
        processImgItem.setIndex(frameID);
        memcpy(h_image, map.data, imgSize.height * imgSize.width * 4);
        //        cudaMemcpy(d_image, map.data, imgSize.height * imgSize.width * 4, cudaMemcpyDeviceToDevice);
        processImgItem.setHostImage(h_image);
        processImgItem.setDeviceImage(d_image);
        processImgItem.setImageSize(imgSize);
        m_matImageBuff->add(processImgItem);
        fixedMem.notifyAddOne();
        gst_buffer_unmap(gstData, &map);
        stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> timeSpan = stop - start;
        sleepTime = (long)(33333 - timeSpan.count());
//        printf("\nPreProcess: %d | %d - [%d, %d]", m_currID, frameID, imgSize.width, imgSize.height);

        if (sleepTime > 1000) {
            std::this_thread::sleep_for(std::chrono::microseconds(1000));
        }
    }
}

cv::Size VPreprocess::getImageSize(int _dataSize)
{
    cv::Size res;

    switch (_dataSize) {
    case 1920*1080: {
        res.width = 1920;
        res.height = 1080;
        break;
    }

    case 640*480: {
        res.width = 640;
        res.height = 480;
        break;
    }

    case 1280*720: {
        res.width = 1280;
        res.height = 720;
        break;
    }

    case 720*576: {
        res.width = 720;
        res.height = 576;
        break;
    }

    case 3840*2160: {
        res.width = 3840;
        res.height = 2160;
        break;
    }

    default: {
        printf("\nERROR: Does NOT support this size of image");
        res.width = 0;
        res.height = 0;
        break;
    }
    }

    return res;
}

index_type VPreprocess::readBarcode(const cv::Mat &_rgbImg)
{
    index_type res = 0;
    vector<decodedObject> decodedObjects;
    int zbarWidth = 130;
    int zbarHeight = 32;
    cv::Rect rectZBar(0, 0, zbarWidth, zbarHeight);
    cv::Mat zbarMat = _rgbImg(rectZBar);
    cv::cvtColor(zbarMat, zbarMat, CV_BGRA2RGB);
    //    cv::imwrite("img/2barcode_" + std::to_string(m_currID) + ".png", zbarMat);
    ZbarLibs::decode(zbarMat, decodedObjects);

    if (decodedObjects.size() > 0) {
        std::string frameID =
            decodedObjects[0].data.substr(0, decodedObjects[0].data.length() - 1);
        res = (index_type)std::atoi(frameID.c_str());
        //    printf("\nBarcode str: %s | %d", decodedObjects.at(0).data.c_str(),
        //    res);
    }

    return res;
}
