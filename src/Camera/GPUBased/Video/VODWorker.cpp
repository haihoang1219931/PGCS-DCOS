#include "VODWorker.h"

VODWorker::VODWorker() {
	dropDetection = 0;
}

VODWorker::~VODWorker() {}

void VODWorker::setMultiTracker(MultiTrack *_mulTracker)
{
    m_mulTracker = _mulTracker;
}

void VODWorker::setPlateOCR(PlateOCR *_plateOCR)
{
	m_plateOCR = _plateOCR;
}

void VODWorker::run()
{
    std::chrono::high_resolution_clock::time_point start, stop;
    long sleepTime = 0;
    m_matImageBuff = Cache::instance()->getProcessImageCache();
    m_rbDetectedObjs = Cache::instance()->getDetectedObjectsCache();
    m_rbIPCEO = Cache::instance()->getMotionImageEOCache();
    m_rbIPCIR = Cache::instance()->getMotionImageIRCache();
    ProcessImageCacheItem processImgItem;
    int prevID = -1;

    while (m_running) {
        std::unique_lock<std::mutex> locker(m_mtx);

        while (!m_enOD) {
            m_cvEnOD.wait_for(locker, std::chrono::seconds(2));
            printf("\n===============>Still waitting for Enable Object Detection");
        }

        start = std::chrono::high_resolution_clock::now();
        processImgItem = m_matImageBuff->last();

        if ((processImgItem.getIndex() == 0) ||
            (processImgItem.getIndex() == m_currID)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        m_currID = processImgItem.getIndex();
        unsigned char *d_imageData = processImgItem.getDeviceImage();
        cv::Size imgSize = processImgItem.getImageSize();
        image_t input;
        input.c = 4;
        input.h = imgSize.height;
        input.w = imgSize.width;
        input.data = (float *)d_imageData;
        //TODO: canculate GME
		cv::Mat trans = cv::Mat::eye(3, 3, CV_64F);
		Eye::MotionImage motionImg = m_rbIPCEO->getElementById(m_currID);
		if(motionImg.getIndex() != -1){
			Eye::MData motionCtx = motionImg.getMotionContext();
			trans.at<double>(0, 0) = trans.at<double>(1, 1) = cos(motionCtx.getRot()) * motionCtx.getScale();
			trans.at<double>(0, 1) = sin( motionCtx.getRot() ) * motionCtx.getScale();
			trans.at<double>(1, 0) = -trans.at<double>(0, 1);
			trans.at<double>(0, 2) = motionCtx.getTx();
			trans.at<double>(1, 2) = motionCtx.getTy();
		}


		//TODO: Detection
		cv::cuda::GpuMat gpu_rgba_frame(cv::Size(input.w, input.h), CV_8UC4, input.data);
		std::vector<bbox_t> detection_boxes;
		if (++dropDetection % 4 == 0)
		{
			detection_boxes.clear();
			dropDetection = 0;
		}
		else
			detection_boxes = m_mulTracker->vehicle_detector->gpu_detect(input, input.w, input.h, 0.2f, false);

		//TODO: Multitracking
		std::vector<bbox_t> track_result = m_mulTracker->multitrack_detect(input, gpu_rgba_frame, detection_boxes, trans);
		// TODO: PlateOCR
        m_plateOCR->run(track_result, input, gpu_rgba_frame, 10);

        //TODO: add track result
        DetectedObjectsCacheItem detectedObjsItem;
        detectedObjsItem.setIndex(m_currID);
        detectedObjsItem.setDetectedObjects(track_result);
        m_rbDetectedObjs->add(detectedObjsItem);
        stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> timeSpan = stop - start;
        sleepTime = (long)(33333 - timeSpan.count());
//        printf("\nVODWorker: process: %d | %d | %d | Box Size = %d", sleepTime, m_currID, processImgItem.getIndex(), track_result.size());
        prevID = m_currID;

        if (sleepTime > 1000) {
            std::this_thread::sleep_for(std::chrono::microseconds(1000));
        }
    }
}

void VODWorker::stop()
{
    m_running = false;
}

void VODWorker::enableOD()
{
    m_enOD = true;
    m_cvEnOD.notify_all();
}

void VODWorker::disableOD()
{
    m_enOD = false;
}

bool VODWorker::isRunning()
{
    return m_enOD;
}
