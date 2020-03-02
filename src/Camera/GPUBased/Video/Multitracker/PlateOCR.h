#ifndef PLATEOCR_H
#define PLATEOCR_H

#include <mutex>
#include "yolo_v2_class.hpp"
#include "plate_utils.h"
#include "../OCR/preprocessing.h"
#include "../OCR/recognition.h"

class PlateOCR
{
public:
	PlateOCR();
	PlateOCR(std::string plate_cfg_file,
			 std::string plate_weight_file);
	~PlateOCR();

	void setPlateDetector(Detector * m_plate_detector);
	void run(std::vector<bbox_t> & track_vec, const image_t& frame, const cv::cuda::GpuMat& gpu_rgba_frame, int max_info_read);

	std::vector<bbox_t> getPlateBoxes(const image_t& frame, const cv::Rect& _roi);

private:
	std::string getPlateString(const image_t& frame, const cv::cuda::GpuMat& gpu_rgba_frame, const bbox_t& box);
	void contrastEnhance(cv::Mat &scr, cv::Mat &dst, int dist = 10);
	cv::Mat deskewImage(cv::Mat image);
	std::vector<uint> sort_indexes(std::vector<bbox_t> & track_vec);

public:
    OCR m_recognizer;
	int m_maxPlateDetect = 10;
	std::mutex m_mtx;
	Detector * m_plate_detector;
	std::map<int, std::string> data;
    cv::Rect m_regA = cv::Rect(324, 384, 1152, 648);
    int m_cter = 0;
};

#endif // PLATEOCR_H
