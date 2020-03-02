#ifndef STRUCTURES_HPP
#define STRUCTURES_HPP

#include <vector>
#include <opencv2/opencv.hpp>

#define MIN_AREA_CHAR               30
#define MIN_WIDTH_CHAR              2
#define MIN_HEIGHT_CHAR             8
#define ANGLE_CENTROID_CHAR         22.0

#define MIN_RATIO_CHAR_SIZE         0.2
#define MAX_RATIO_CHAR_SIZE         1.2
#define MIN_RATIO_PLATE_SIZE        0.6
#define MAX_RATIO_PLATE_SIZE        5.8

#define ANGLE_BETWEEN_CHARS         12.0
#define DIFF_AREA_BETWEEN_CHARS     4.0
#define DIFF_WIDTH_BETWEEN_CHARS    0.4
#define DIFF_HEIGHT_BETWEEN_CHARS   0.3
#define IOU_CHARS                   0.4
#define IOU_PLATES                  0.2

// define a structure for storing possible characters.
struct possibleChar
{
public:
    possibleChar(std::vector<cv::Point> contour_points)
    {
        s_contour_points = contour_points;
        s_char_box = cv::boundingRect(s_contour_points);
        s_char_rot_box = cv::minAreaRect(s_contour_points);
        // Compute the clockwise angle of rotated box
        cv::Point2f rot_box_points[4];
        s_char_rot_box.points(rot_box_points);
        cv::Point2f edge1 = cv::Vec2f(rot_box_points[1].x, rot_box_points[1].y) - cv::Vec2f(rot_box_points[0].x, rot_box_points[0].y);
        cv::Point2f edge2 = cv::Vec2f(rot_box_points[3].x, rot_box_points[3].y) - cv::Vec2f(rot_box_points[0].x, rot_box_points[0].y);
        cv::Point2f unit_vector = cv::Vec2f(1.f, 0.f);
        cv::Point2f bottom_edge = cv::norm(edge1) < cv::norm(edge2) ? edge1 : edge2;

        float dot = unit_vector.x * bottom_edge.x + unit_vector.y * bottom_edge.y;
        float det = - unit_vector.x * bottom_edge.y + unit_vector.y * bottom_edge.x;

        s_char_angle = (float)(std::atan2(det, dot) * 180.f / CV_PI);
        //-----------------------------------------/-
        s_char_area = s_char_box.width * s_char_box.height;
        s_char_centerX = (2 * s_char_box.x + s_char_box.width) / 2;
        s_char_centerY = (2 * s_char_box.y + s_char_box.height) / 2;
        s_char_diagonal = (float)std::sqrt(s_char_box.width * s_char_box.width + s_char_box.height * s_char_box.height);
        s_char_ratio = (float)s_char_box.width / s_char_box.height;
    }

    std::vector<cv::Point> s_contour_points;    /**< contains points belonging to contour */
    cv::Rect s_char_box;                        /**< a rectangle box bounding a character */
    cv::RotatedRect s_char_rot_box;             /**< a minimal rectangle be able to bound the character */
    int s_char_area;                            /**< area of the rectangle bounding box of the character */
    int s_char_centerX;                         /**< x-coordinate of the bounding box */
    int s_char_centerY;                         /**< x-coordinate of the bounding box */
    float s_char_diagonal;                      /**< diagonal length of the bounding box */
    float s_char_ratio;                         /**< ratio of width by height of the bounding box */
    int s_char_index;                           /**< index is  assigned to the character */
    float s_char_angle;                         /**< incline of character (in degree) against horizontal plain */
};

// define a structure for saving possible plates.
struct possiblePlate
{
public:
    cv::Rect s_plate_box;                       /**< a rectangle box bounding a plate */
    cv::RotatedRect s_plate_rot_box;            /**< a minimal rectangle be able to bound the plate */
    cv::Mat s_plate_gray_img;                   /**< grayscale image of plate after detecting plate */
    cv::Mat s_plate_angle_point;                /**< contains 4 angle points of detected plate */
    int s_plate_num_chars;                      /**< number of characters in detected plates */
    int s_plate_id;                             /**< index is assigned to the detected plate */
    std::vector<std::string> s_plate_text;      /**< unused */
};

// Some structures that are used to support for recognizing.
struct counter
{
public:
    int s_index;
    int s_count;
};

struct groupPlate
{
public:
    cv::Mat s_plate_img;
    cv::RotatedRect s_rot_box;
    std::string s_text;
};

struct object
{
public:
    cv::Mat s_object_img;
    std::string s_code;
};
#endif // STRUCTURES_HPP
