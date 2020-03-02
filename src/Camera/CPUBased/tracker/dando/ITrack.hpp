#ifndef ITRACK_HPP
#define ITRACK_HPP

#include <opencv2/opencv.hpp>

#include "Utilities.hpp"
#include "LME/lme.hpp"

// define trackerType
enum TrackerType {
    KCF,
    sKCF,
    thresholding
};

/** ===================================================
 *
 *      Kalman Filter
 *
 * ====================================================
 */
namespace Eye
{
    class KalmanFilter
    {
            cv::Mat m_A;
            cv::Mat m_H;
            cv::Mat m_Q;
            cv::Mat m_R;

            cv::Mat m_P;
            cv::Mat m_P_;

            cv::Mat m_x;
            cv::Mat m_x_;
            cv::Mat m_z;

        public:
            KalmanFilter() {}
            ~KalmanFilter() {}

        public:
            void setDefaultModel();
            void setState(const float &x, const float &y, const float &dx, const float &dy);
            void predictMotion(float *dx, float *dy);
            void correctModel(const float &dx, const float &dy);
    };
};




/** ===================================================
 *
 *      TrackingParams Struct
 *
 * ====================================================
 */
struct TrackingParams {
    // In the formular "alpha = (K + lamda*I)^-1 * Y"
    cv::Mat     Alpha;
    cv::Mat     Y;
    // Hanning Window
    cv::Mat     HannWin;

    float       scale;
    float       sigma;      // Gaussian kernel width
    float       alpha;      //
    float       beta;       // Polynomial kernel parameter
    float       paddRatio;  // Image patch padding ratio

    float       lamda;      // Learning rate
    float       lamda1;     // Learning rate

    cv::Size    tmplSize;   // Size of template (HOG or gray)
    int         feaMapSize[3];  // Size of feature map
    int         featureType;// Feature type to compute correlation
    int         kernelType; // Kernel Type

    TrackingParams(int _featureType, int _kernelType)
    {
        featureType = _featureType;
        kernelType  = _kernelType;

        if (kernelType == KERNEL_POLYNOMIAL) {
            alpha   = 0.08f;
            beta    = 1.15f;
        }

        if (featureType == FEATURE_HOG) {
            sigma   = 0.6f;
        }

        paddRatio = PADDING;
        lamda     = LAMDA_;
        lamda1    = LAMDA_1;
    }
};



/** ===================================================
 *
 *      ITrack Class
 *
 * ====================================================
 */
class ITrack
{
        TrackerType trackertype;
        TrackingParams  m_params;
        cv::Rect        m_objRoi;
        cv::Mat         m_tmpl;
        cv::Mat         m_fftK;
        Eye::KalmanFilter    m_kalmanFilter;
        int             m_trackStatus;
        bool            m_trackInited;
        int             m_trackLostCnt;
        bool            m_running;

        int             m_zoomDirection;
        cv::Rect        m_orgRoi;

    public:
        ITrack(int _featureType, int _kernelType);
        ~ITrack();

    public:

        void initTrack(cv::Mat &_image, cv::Rect _selRoi);
        void performTrack(cv::Mat &_image);
        cv::Rect getPosition();
        int getZoomDirection();
        int trackStatus();
        bool isInitialized();
        void resetTrack();
        bool isRunning();

        std::string getZoomEO(const float &_inIRFov, const float &_inEOFov, float *_outEOFov);
        std::string getZoomIR(const float &_inIRFov);
    private:
        cv::Mat getPatch(cv::Mat &img, float ctx, float cty, float scale, cv::Size target_sz);
        cv::Mat extractFeatures(cv::Mat &patch);
        cv::Point2f fastdetect(cv::Mat &x, cv::Mat &y, float &peak_value , float &psr_value);
        void train(cv::Mat &x, float learning_rate);
        float p;
};

extern bool isTextureLess(cv::Mat &patch);

#endif // ITRACK_HPP

