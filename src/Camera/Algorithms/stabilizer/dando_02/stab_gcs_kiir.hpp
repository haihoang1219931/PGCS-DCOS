#ifndef STAB_GCS_KIIR_HPP
#define STAB_GCS_KIIR_HPP

#include <opencv2/videostab.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <algorithm>
#include <vector>
#include <numeric>
#include <iostream>
#define PROCESS_RELEASE

namespace stab_gcs_kiir
{


    /********************************************
     *   d e f i n e s    &    m a c r o s
     *******************************************/
    typedef     cv::Mat                     vtx_image;
    typedef     cv::Mat                     vtx_mat;
    typedef     std::vector<cv::Point2f>    vtx_points_vector;


#define     PSTD            4e-3
#define     CSTD            0.25f

#define     GOOD_FEATURE            1 << 0
#define     FAST_CORNER             1 << 1

#define     HOMOGRAPHY              1 << 0
#define     RIGID_TRANSFORM         1 << 1

#define     MAX_CORNERS             500u
#define     MIN_CORNERS             100u
#define     MIN_CORNER_PER_BLK      50u     // MIN_CORNER_PER_BLK = MIN_CORNERS / 2
#define     VERTICA_BLKS            7u
#define     HORIZON_BLKS            8u

#define     GOODFEATURE_QUALITY     0.005f
#define     GOODFEATURE_MIN_DIS     20.0f
#define     GOODFEATURE_BLKSIZE     3
#define     FASTCORNERS_MIN_DIS     25
#define     RANSAC_INLIER_THRESHOLD 2.0f
#define     MIN_INLIER              15
#define     MIN_INLIER_RATIO        0.1f
#define     MIN_EIGEN_VALUE         1e-4f

#define     IIR_MAX_DX              60
#define     IIR_MAX_DY              60
#define     IIR_MAX_DA              0.3491f         // 20 degree
#define     IIR_CORRECTION_D        0.95f
#define     IIR_CORRECTION_A        0.95f
#define     IIR_CORRECTION_BIAS     -0.01f
#define     IIR_POWER_FACTOR        4

#define     GENERIC_ERROR           -1000
#define     SUCCESS                 0
#define     BAD_TRANSFORM           -1

#define     SIGMA_X                 3.0f



    /********************************************
     *   s t r u c t s    &    c l a s s e s
     *******************************************/
    //!
    //! \brief The TransformParam struct
    //!
    struct TransformParam {
        TransformParam() {}
        TransformParam(double _dx, double _dy, double _da)
        {
            dx = _dx;
            dy = _dy;
            da = _da;
        }

        double dx;
        double dy;
        double da; // angle
    };




    //!
    //! \brief The Trajectory struct
    //!
    struct Trajectory {
        Trajectory() {}
        Trajectory(double _x, double _y, double _a)
        {
            x = _x;
            y = _y;
            a = _a;
        }
        // "+"
        friend Trajectory operator+(const Trajectory &c1, const Trajectory  &c2)
        {
            return Trajectory(c1.x + c2.x, c1.y + c2.y, c1.a + c2.a);
        }
        //"-"
        friend Trajectory operator-(const Trajectory &c1, const Trajectory  &c2)
        {
            return Trajectory(c1.x - c2.x, c1.y - c2.y, c1.a - c2.a);
        }
        //"*"
        friend Trajectory operator*(const Trajectory &c1, const Trajectory  &c2)
        {
            return Trajectory(c1.x * c2.x, c1.y * c2.y, c1.a * c2.a);
        }
        //"/"
        friend Trajectory operator/(const Trajectory &c1, const Trajectory  &c2)
        {
            return Trajectory(c1.x / c2.x, c1.y / c2.y, c1.a / c2.a);
        }
        //"="
        Trajectory operator =(const Trajectory &rx)
        {
            x = rx.x;
            y = rx.y;
            a = rx.a;
            return Trajectory(x, y, a);
        }

        double x;
        double y;
        double a; // angle
    };



    class vtx_KIIRStabilizer
    {
            // GME attributes
            vtx_image           _curr_gray, _prev_gray;
            vtx_points_vector   _curr_pts, _prev_pts;
            vtx_points_vector   _curr_pts_lk, _prev_pts_lk;
            uint                _key_type, _motion_mode;
            vtx_mat             _T;

            // KIIR stabilizer
            double _a, _x, _y;
            Trajectory c_X;      //posteriori state estimate
            Trajectory c_X_;     //priori estimate
            Trajectory c_P;      // posteriori estimate error covariance
            Trajectory c_P_;     // priori estimate error covariance
            Trajectory c_K;      //gain
            Trajectory c_z;      //actual measurement

            Trajectory c_Q;      // process noise covariance
            Trajectory c_R;      // measurement noise covariance

            vtx_mat    _acc_T;
            double     _last_correction;

        public:
            vtx_KIIRStabilizer()
            {
                // intial guesses
                c_X = Trajectory(0, 0, 0);    //Initial estimate,  set 0
                c_P = Trajectory(1, 1, 1);     //set error variance,set 1
                _a   = 0;
                _x   = 0;
                _y   = 0;
                // config noises
                c_Q = Trajectory(PSTD, PSTD, PSTD);
                c_R = Trajectory(CSTD, CSTD, CSTD);
                // initialize accumulate transformation
                _T      = vtx_mat::eye(3, 3, CV_64F);
                _acc_T  = vtx_mat::eye(3, 3, CV_64F);
                _last_correction = 1.0f;
            }
            ~vtx_KIIRStabilizer() { }

        private:

            void ageDelay();

        public:

            void setMotionEstimnator(const uint key_type, const uint motion_mode);

            int run(const vtx_image &input_rgb, cv::Mat &_stab_data, float *gmeData);

    };




    /*******************************************************
     *      f u n c t i o n    d e c l a r a t i o n
     ******************************************************/
    extern int vtx_extractKeypoints(vtx_image &image, vtx_points_vector &pts, uint keypoint_type, uint max_pts_num);

    extern int  vtx_estimate_transform(uint mode, vtx_points_vector &pts_1, vtx_points_vector &pts_2, vtx_mat &trans);

    extern int vtx_appendVector(vtx_points_vector &dst, vtx_points_vector &src, cv::Point2f shift);

    extern void vtx_iirSmoothHomo(vtx_mat &homo_mat);

    extern void vtx_homoDecomposition(vtx_mat &homo_mat, double *dx, double *dy, double *da, vtx_mat &P);

    extern void vtx_homoDecomposition_v2(vtx_mat &homo_mat, double *dx, double *dy, double *da, vtx_mat &P);

    extern void vtx_homoComposition(vtx_mat &homo_mat, const double dx, const double dy, const double da, const vtx_mat &P);

    extern void vtx_homoComposition_v2(vtx_mat &homo_mat, const double dx, const double dy, const double da, const vtx_mat &P);

    extern void vtx_overlapConstrain(vtx_mat &homo_mat);

    extern void vtx_homographyFilter(vtx_mat &homo_mat, const cv::Size frame_size);


    extern void vtx_enhancement(const vtx_mat &input, vtx_mat &output);

    extern void vtx_certify_model(vtx_mat &model, std::vector<cv::Point2f> &pts1, std::vector<cv::Point2f> &pts2,
                                  const uint d, const double threshold, uint &inliers, bool &status);

    extern void vtx_RANSACRigid(std::vector<cv::Point2f> &pts1, std::vector<cv::Point2f> &pts2, vtx_mat &transform, const double threshold,
                                const double inliers_ratio, const double p);

    template <typename T>
    extern std::vector<size_t> sort_indexes(const std::vector<T> &v);

    extern std::vector<uint> random_sampling(const uint range, const uint rand_num);

    extern bool vtx_isIdentity(const vtx_mat &transform);

};

#endif // STAB_GCS_KIIR_HPP
