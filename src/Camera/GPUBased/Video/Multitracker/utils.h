#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <queue>
#include <fstream>
#include <thread>
#include <future>
#include <atomic>
#include <mutex>         // std::mutex, std::unique_lock
#include <cmath>

#include "yolo_v2_class.hpp"
#include "opencv2/opencv.hpp"

struct detection_data_t {
	cv::Mat cap_frame;
	std::shared_ptr<image_t> det_image;
	std::vector<bbox_t> result_vec;
	cv::Mat draw_frame;
	bool new_detection;
	uint64_t frame_id;
	bool exit_flag;
	cv::Mat zed_cloud;
	std::queue<cv::Mat> track_optflow_queue;
	detection_data_t() : exit_flag(false), new_detection(false) {}
};

template<typename T>
class send_one_replaceable_object_t {
public:
	const bool sync;
	std::atomic<T *> a_ptr;
public:

	void send(T const& _obj) {
		T *new_ptr = new T;
		*new_ptr = _obj;
		if (sync) {
			while (a_ptr.load()) std::this_thread::sleep_for(std::chrono::milliseconds(3));
		}
		std::unique_ptr<T> old_ptr(a_ptr.exchange(new_ptr));
	}

	T receive() {
		std::unique_ptr<T> ptr;
		do {
			while(!a_ptr.load()) std::this_thread::sleep_for(std::chrono::milliseconds(3));
			ptr.reset(a_ptr.exchange(NULL));
		} while (!ptr);
		T obj = *ptr;
		return obj;
	}

	bool is_object_present() {
		return (a_ptr.load() != NULL);
	}

	send_one_replaceable_object_t(bool _sync = true) : sync(_sync), a_ptr(NULL)
	{}
};

std::vector<std::string> objects_names_from_file(std::string const filename)
{
	std::ifstream file(filename);
	std::vector<std::string> file_lines;
	if (!file.is_open()) return file_lines;
	for(std::string line; getline(file, line);) file_lines.push_back(line);
	std::cout << "object names loaded \n";
	return file_lines;
}

static cv::Scalar obj_id_to_color(int obj_id) {
	int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };
	int const offset = obj_id * 123457 % 6;
	int const color_scale = 150 + (obj_id * 123457) % 100;
	cv::Scalar color(colors[offset][0], colors[offset][1], colors[offset][2]);
	color *= color_scale;
	return color;
}

void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names, int current_det_fps= -1)
{
	int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };

	for (auto &i : result_vec) {
		cv::Scalar color = obj_id_to_color(i.obj_id);
		cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 2);
		if (obj_names.size() > i.obj_id) {
			std::string obj_name = obj_names[i.obj_id];
			if (i.track_id > 0) obj_name += " - " + std::to_string(i.track_id);
			cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
			int max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
			max_width = std::max(max_width, (int)i.w + 2);
			//max_width = std::max(max_width, 283);

			cv::rectangle(mat_img, cv::Point2f(std::max((int)i.x - 1, 0), std::max((int)i.y - 35, 0)),
						  cv::Point2f(std::min((int)i.x + max_width, mat_img.cols - 1), std::min((int)i.y, mat_img.rows - 1)),
						  color, CV_FILLED, 8, 0);
			putText(mat_img, obj_name, cv::Point2f(i.x, i.y - 16), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0), 2);
		}
	}
	if (current_det_fps >= 0) {
		std::string fps_str = "FPS detection: " + std::to_string(current_det_fps);
		putText(mat_img, fps_str, cv::Point2f(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(50, 255, 0), 2);
	}
}

void draw_boxes_center(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names, float fps = -1)
{
	for (auto &i : result_vec) {
		cv::Scalar color = obj_id_to_color(i.obj_id);

		//		cv::Scalar color = obj_id_to_color(1);
		cv::rectangle(mat_img, cv::Rect(i.x-i.w/2, i.y-i.h/2, i.w, i.h), color, 2);
		//        if (obj_names.size() > i.obj_id)
		{
			//            std::string obj_name = obj_names[i.obj_id];
			//            if (i.track_id > 0) obj_name += " - " + std::to_string(i.track_id);
			std::string obj_name;
//			std::string string_id(i.string_id);
			std::string string_id(i.track_info->stringinfo);
			if (string_id.empty())
				obj_name = std::string();
//							obj_name = std::to_string(i.track_id);
			else
				obj_name = string_id;

			cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
			int max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
			max_width = std::max(max_width, (int)i.w + 2);
			//max_width = std::max(max_width, 283);

			if (!obj_name.empty()){

				cv::rectangle(mat_img, cv::Point2f(std::max((int)i.x - 1, 0), std::max((int)i.y - 35, 0)),
							  cv::Point2f(std::min((int)i.x + max_width, mat_img.cols - 1), std::min((int)i.y, mat_img.rows - 1)),
							  color, CV_FILLED, 8, 0);
				putText(mat_img, obj_name, cv::Point2f(i.x, i.y - 16), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0), 2);
			}
		}
	}
	if (fps >= 0) {
		std::string fps_str = "FPS: " + std::to_string(fps);
		putText(mat_img, fps_str, cv::Point2f(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(50, 255, 0), 2);
	}
}

void save_console_result(std::ofstream & ofs, std::vector<bbox_t> const result_vec, std::vector<std::string> const obj_names, int frame_id = -1) {
	for (auto &i : result_vec) {
		if (frame_id >= 0) ofs << frame_id << ",-1,";
		ofs << i.x << "," << i.y << "," << i.w << "," << i.h << "," << std::setprecision(3) << i.prob << ",-1,-1,-1" << std::endl;
		//        if (obj_names.size() > i.obj_id) std::cout << obj_names[i.obj_id] << " - ";
		//        std::cout << "obj_id = " << i.obj_id << ",  x = " << i.x << ", y = " << i.y
		//            << ", w = " << i.w << ", h = " << i.h
		//            << std::setprecision(3) << ", prob = " << i.prob << std::endl;
	}
}

#endif // UTILS_H
