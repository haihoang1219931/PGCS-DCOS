#ifndef IPSEARCH_UTILS_H
#define IPSEARCH_UTILS_H

#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <iomanip>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#include "ipsearch_stats.h"
#include <fstream>


/************************************************
 *
 *          constants - parameters
 *
 ***********************************************/
#define     NUM_DESCRIPTOR                  10       /**< maximum number of descriptors in database  */
#define     NUM_FLANN_DESCRIPTOR            100
#define     NUM_KEYPOINTS_PREDICT           700     /**< maximum number of keypoints while local search  */
#define     NUM_KEYPOINTS_GLOBAL_SEARCH     1800    /**< maximum number of keypoints while search full frame */
#define     NUM_KEYPOINTS_UPDATE            700     /**< maximum number of keypoints detected while update object database */
#define     SCALE_FACTOR                    1.2f    /**< pyramid decimation ratio when detected keypoints using ORB */
#define     SCALE_LEVEL                     8       /**< the number of pyramid levels when detected keypoints using ORB */
#define     EDGE_THRESHOLD                  31      /**< size of the border where the keypoints are not detected  */
#define     PATCH_SIZE                      31      /**< size of the patch used by the oriented brief (binary robust independent elementary features) descriptors */
#define     FAST_THRESHOLD                  20      /**< threshold of FAST detector features */
#define     RANSAC_THRESHOLD                5.0     /**< max error to classify as inliner */
#define     MATCH_RATIO                     0.9     /**< coff to make dicision pair of keypoints is good matches */
#define     WIDTH_REGION_2_COMPUTE_KP       400     /**< width of region in which detect and compute features when update and local search object */
#define     HEIGHT_REGION_2_COMPUTE_KP      400     /**< height of region in which detect and compute features when update and local search object */
#define     MIN_INLINER_KEYPOINTS           5       /**< minimum number of features to make decision object is found or not */
#define     ROTATE_ANGLE_RESOLUTION         30
#define     HARRIS_BLOCK_SIZE               9
#define     NUM_COLORS_NAME                 11
#define     FILE_DERICTORY                  "../DD11_w2c_fast.txt"




/************************************************
 *
 *          functions
 *
 ***********************************************/
namespace ip {

    namespace objsearch {

        /**
         * @brief drawBoudingBox - Mo ta muc dich cua ham
         * @param image     Mo ta bien
         * @param bb
         */
        extern void drawBoudingBox(cv::Mat image, std::vector<cv::Point2f> bb);

        /**
         * @brief drawBoundingCirle
         * @param image
         * @param center
         * @param radius
         */
        extern void drawBoundingCirle(cv::Mat image, cv::Point2f center, int radius);

        /**
         * @brief drawStatistics
         * @param image
         * @param stats
         * @param fps
         */
        extern void drawStatistics(cv::Mat image, const Stats& stats, double fps);

        /**
         * @brief Points
         * @param keypoints
         * @return
         */
        extern std::vector<cv::Point2f> Points(std::vector<cv::KeyPoint> keypoints);

        /**
         * @brief seclectROI
         * @param video_name
         * @param frame
         * @return      Mo ta return
         */
        extern cv::Rect2d seclectROI(const cv::String & video_name, const cv::Mat &frame);

        extern const int ColorNames[];

    }

}

#endif // IPSEARCH_UTILS_H
