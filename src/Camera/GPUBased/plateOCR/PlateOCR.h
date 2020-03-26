#ifndef PLATEOCR_H
#define PLATEOCR_H

#include <mutex>
#include "../OD/yolo_v2_class.hpp"
#include "../Clicktrack/recognition.h"
#include "../Clicktrack/preprocessing.h"
#include "../../../Files/FileControler.h"
class PlateOCR
{
public:
	PlateOCR();
	~PlateOCR();

	void setPlateDetector(Detector * _plate_detector);
    void setOCR(OCR* _OCR);
	void run(std::vector<bbox_t> & track_vec, const image_t& frame, const cv::Mat& cpu_gray_frame, int max_info_read);

	std::vector<bbox_t> getPlateBoxes(const image_t& frame, const cv::Rect& _roi);

private:
	std::string getPlateString(const image_t& frame, const cv::Mat& cpu_gray_frame, const bbox_t& box);
	void contrastEnhance(cv::Mat &scr, cv::Mat &dst, int dist = 10);
	cv::Mat deskewImage(cv::Mat image);
	std::vector<uint> sort_indexes(std::vector<bbox_t> & track_vec);

public:
	int m_maxPlateDetect = 10;
	std::mutex m_mtx;
	Detector * m_plate_detector;
    std::map<int, std::vector<std::string>> data;

    OCR* m_OCR;

private:
    std::vector<int> wanted_class{4, 5, 6, 9, 10};
    std::string m_logFile;
	// helper functions
public:
	static float get_strings_correlation(const std::string& input, const std::string& truth);
    static std::string search_combine_plates(std::vector<std::string> cur_data);
};

#endif // PLATEOCR_H
