#ifndef KALMAN_HPP
#define KALMAN_HPP
#include<math.h>
#include<opencv2/opencv.hpp>

class kalmanfilter
{
    public:
        kalmanfilter();
        ~kalmanfilter();
    public:
        void setDefaultModel();
        cv::Point get_estimate_position();
        void setState(const float& _x, const float& _y, const float& _dx, const float& _dy);
        void predictMotion(float* _dx, float* _dy);
        void correctModel(const float& _dx, const float& _dy);
    private:
        cv::Mat m_A;    /**< A matrix */
        cv::Mat m_H;    /**< H matrix */
        cv::Mat m_Q;    /**< Q matrix */
        cv::Mat m_R;    /**< R matrix */

        cv::Mat m_P;    /**< Prior P matrix */
        cv::Mat m_P_;   /**< Posterior P matrix */

        cv::Mat m_x;    /**< Prior x */
        cv::Mat m_x_;   /**< Posterior x */
        cv::Mat m_z;    /**< Mesurement z */
};

#endif // KALMAN_HPP
