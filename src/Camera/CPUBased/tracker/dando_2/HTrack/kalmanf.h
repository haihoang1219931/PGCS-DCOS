#ifndef KALMANF_H
#define KALMANF_H

#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
namespace Eye {
    class KalmanFilter
    {
        cv::Mat     m_A;
        cv::Mat     m_H;

        cv::Mat     m_Q;
        cv::Mat     m_R;
        cv::Mat     m_P_;
        cv::Mat     m_P;

        cv::Mat     m_x_;
        cv::Mat     m_x;
        cv::Mat     m_z;

    public:
        KalmanFilter() {}
        ~KalmanFilter() {}
        void getDefaultState();

    public:
        void initState( const float &x, const float &y );

        void correct( const float &dx, const float &dy );

        void predict( );

        void getMotion( float *dx, float *dy );
    };
}


#endif // KALMANF_H
