#ifndef GME_SHORT_HPP
#define GME_SHORT_HPP

#include <iostream>
#include <opencv2/opencv.hpp>

namespace gme {

/********************************************
 *   d e f i n e s    &    m a c r o s
 *******************************************/
typedef     cv::Mat                     vtx_image;
typedef     cv::Mat                     vtx_mat;
typedef     std::vector<cv::Point2f>    vtx_points_vector;

#define     GOOD_FEATURE            1 << 0
#define     FAST_CORNER             1 << 1

#define     HOMOGRAPHY              1 << 0
#define     RIGID_TRANSFORM         1 << 1

#define     GENERIC_ERROR           -1000
#define     SUCCESS                 0
#define     BAD_TRANSFORM           -1

#define     MAX_CORNERS             500u
#define     MIN_CORNERS             100u
#define     VERTICA_BLKS            3u
#define     HORIZON_BLKS            4u

#define     GOODFEATURE_QUALITY     0.005f
#define     GOODFEATURE_MIN_DIS     20.0f
#define     GOODFEATURE_BLKSIZE     3
#define     FASTCORNERS_MIN_DIS     35
#define     RANSAC_INLIER_THRESHOLD 2.0f
#define     MIN_INLIER              15
#define     MIN_INLIER_RATIO        0.1f
#define     MIN_EIGEN_VALUE         1e-4f


/********************************************
 *          m a i n    c l a s s
 *******************************************/
class GMEstimator
{
    vtx_image _prev_gray, _curr_gray;
    vtx_points_vector _prev_pts, _curr_pts;
    uint _keypoint_type;
    uint _motion_mode;

    vtx_mat _last_trans;

public:
    GMEstimator( uint keypoint_type = GOOD_FEATURE, uint motion_mode = RIGID_TRANSFORM )
    {
        _keypoint_type = keypoint_type;
        _motion_mode   = motion_mode;
        _last_trans    = vtx_mat::eye( 3, 3, CV_64F );
    }
    ~GMEstimator() { }

private:

    void ageDelay();

public:

    int run( const vtx_image &rgb_input, vtx_mat &trans, bool backward_motion = true );
};



/********************************************
 *          f u n c t i o n s
 *******************************************/
extern int vtx_extractKeypoints( vtx_image &image, vtx_points_vector &pts, uint keypoint_type, uint max_pts_num );

extern int  vtx_estimate_transform( uint mode, vtx_points_vector &pts_1, vtx_points_vector &pts_2, vtx_mat &trans );

extern int vtx_appendVector( vtx_points_vector &dst, vtx_points_vector &src, cv::Point2f shift );

}


#endif // GME_SHORT_HPP
