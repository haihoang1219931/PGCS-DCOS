#include "kalmanf.h"

using namespace Eye;
void KalmanFilter::getDefaultState()
{
    m_A = cv::Mat::eye( 4, 4, CV_32FC1 );
    m_A.at<float>(0, 2) = 1.0f;
    m_A.at<float>(1, 3) = 1.0f;

    m_Q = cv::Mat::eye( 4, 4, CV_32FC1 ) * 1e-4;

    m_R = cv::Mat::zeros( 2, 2, CV_32FC1 );
    m_R.at<float>(0, 0) = 0.2845f;
    m_R.at<float>(0, 1) = 0.0045f;
    m_R.at<float>(1, 0) = 0.0045f;
    m_R.at<float>(1, 1) = 0.0455f;

    m_P = cv::Mat::eye( 4, 4, CV_32FC1 ) * 0.1;

    m_H = cv::Mat::zeros( 2, 4, CV_32FC1 );
    m_H.at<float>(0, 0) = 1.0f;
    m_H.at<float>(1, 1) = 1.0f;
}


void KalmanFilter::initState(const float &x, const float &y)
{
    m_x = cv::Mat::zeros( 4, 1, CV_32FC1 );
    m_x.at<float>(0) = x;
    m_x.at<float>(1) = y;

    m_z = cv::Mat( 2, 1, CV_32FC1 );
    m_z.at<float>(0) = x;
    m_z.at<float>(1) = y;
}


void KalmanFilter::predict()
{
    m_x_ = m_A * m_x;
    m_P_ = m_A * m_P * m_A.t() + m_Q;
}

void KalmanFilter::correct(const float &dx, const float &dy)
{
    m_z.at<float>(0) += dx;
    m_z.at<float>(1) += dy;

    cv::Mat tmp = m_H * m_P_ * m_H.t() + m_R;
    cv::Mat K = m_P_ * m_H.t() * tmp.inv( cv::DECOMP_LU );
    m_x = m_x_ + K * (m_z - m_H * m_x_);

    cv::Mat I = cv::Mat::eye( 4, 4, CV_32FC1 );
    m_P = (I - K * m_H) * m_P_;
}

void KalmanFilter::getMotion(float *dx, float *dy)
{
    *dx = m_x_.at<float>(2);
    *dy = m_x_.at<float>(3);
}
