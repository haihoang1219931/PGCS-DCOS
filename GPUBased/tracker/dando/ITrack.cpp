#include <chrono>
#include <math.h>

#include "ITrack.hpp"
#include "HTrack/fhog.hpp"

/** ===================================================
 *
 *      Kalman Filter
 *
 * ====================================================
 */
//!
//! \brief KalmanFilter::setDefaultModel
//!
namespace Eye
{
    void KalmanFilter::setDefaultModel()
    {
        m_A = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0,
               0, 1, 0, 1,
               0, 0, 1, 0,
               0, 0, 0, 1);
        m_H = cv::Mat(2, 4, CV_32F);
        cv::setIdentity(m_H);
        m_Q = cv::Mat(4, 4, CV_32F);
        cv::setIdentity(m_Q, cv::Scalar::all(1e-2));
        m_R = (cv::Mat_<float>(2, 2) << 0.2845, 0.0045,
               0.0045, 0.0455);
        m_P = cv::Mat(4, 4, CV_32F);
        cv::setIdentity(m_P, cv::Scalar::all(0.1));
    }


    //!
    //! \brief KalmanFilter::setState
    //! \param x
    //! \param y
    //! \param dx
    //! \param dy
    //!
    void KalmanFilter::setState(const float &x, const float &y, const float &dx, const float &dy)
    {
        m_x = (cv::Mat_<float>(4, 1) << x, y, dx, dy);
        m_z = (cv::Mat_<float>(2, 1) << x, y);
    }


    //!
    //! \brief KalmanFilter::predictMotion
    //! \param dx
    //! \param dy
    //!
    void KalmanFilter::predictMotion(float *dx, float *dy)
    {
        m_x_ = m_A * m_x;
        m_P_ = m_A * m_P * m_A.t() + m_Q;
        *dx  = m_x_.at<float>(2, 0);
        *dy  = m_x_.at<float>(3, 0);
    }


    //!
    //! \brief KalmanFilter::correctModel
    //! \param dx
    //! \param dy
    //!
    void KalmanFilter::correctModel(const float &dx, const float &dy)
    {
        m_z.at<float>(0, 0) += dx;
        m_z.at<float>(1, 0) += dy;
        cv::Mat tmp = m_H * m_P_ * m_H.t() + m_R;
        cv::Mat K   = m_P_ *  m_H.t() * tmp.inv();
        m_x = m_x_ + K * (m_z - m_H * m_x_);
        cv::Mat I = cv::Mat::eye(4, 4, CV_32FC1);
        m_P = (I - K * m_H) * m_P_;
    }




};

//!
//! \brief ITrack::ITrack
//! \param _featureType
//! \param _kernelType
//!
ITrack::ITrack(int _featureType, int _kernelType) : m_params(_featureType, _kernelType)
{
    m_trackInited = false;
    m_trackStatus = TRACK_LOST;
    m_trackLostCnt = 0;
    m_running = false;
    m_zoomDirection = 0;
}


//!
//! \brief ITrack::~ITrack
//!
ITrack::~ITrack()
{
}



//!
//! \brief ITrack::getPatch
//! \param img
//! \param ctx                      Patch center's x-coordinate
//! \param cty                      Patch center's y-coordinate
//! \param scale                    Size scaling factor
//! \param target_sz
//!
cv::Mat ITrack::getPatch(cv::Mat &img, float ctx, float cty, float scale, cv::Size target_sz)
{
    cv::Rect roi;
    cv::Mat patch;
    //===== Compute the boundary of the image patch
    roi.width  = (int)(m_params.scale * m_params.tmplSize.width); //target_sz.width; //
    roi.height = (int)(m_params.scale * m_params.tmplSize.height); //target_sz.height; //(int)(m_params.scale * m_params.tmplSize.height);
    roi.x      = ctx - roi.width / 2;
    roi.y      = cty - roi.height / 2;
    std::vector<int> border = limitRect(roi, cv::Size(img.cols, img.rows));
    patch = img(roi).clone();

    if ((border[0] != 0) || (border[1] != 0) || (border[2] != 0) || (border[3] != 0)) {
        cv::copyMakeBorder(patch, patch, border[0], border[1], border[2], border[3], cv::BORDER_CONSTANT);
    }

    if ((patch.cols != m_params.tmplSize.width) || (patch.rows != m_params.tmplSize.height)) {
        cv::resize(patch, patch, m_params.tmplSize);            // ??? Should I crop for indentical ratio
    }

    return patch;
}



//!
//! \brief ITrack::extractFeatures          Extract feature map from given image patch
//! \param patch
//! \return
//!
cv::Mat ITrack::extractFeatures(cv::Mat &patch)
{
    assert(patch.channels() == 1);
    cv::Mat featureMap;

    if (m_params.featureType == FEATURE_HOG) {
        IplImage iplPatch = patch;
        CvLSVMFeatureMapCaskade *map;
        getFeatureMaps(&iplPatch, CELL_SIZE, &map);
        normalizeAndTruncate(map, 0.2f);
        PCAFeatureMaps(map);            // each reduced feature vector has 31 components
        featureMap = cv::Mat(cv::Size(map->numFeatures, map->sizeX * map->sizeY), CV_32F, map->map);
        featureMap = featureMap.t();

        if (m_trackInited) {
            featureMap = m_params.HannWin.mul(featureMap);
        }

        freeFeatureMapObject(&map);
    }

    return featureMap;
}


//!
//! \brief ITrack::fastdetect       Detect the shift of the image patch based on phase correlation
//! \param x
//! \param y
//! \param peak_value
//! \param psr_value
//! \return
//!
cv::Point2f ITrack::fastdetect(cv::Mat &z, cv::Mat &x, float &peak_value, float &psr_value)
{
    //===== Phase Correlation Computation
    cv::Mat kernel;
    kernel = gaussianCorrelation(x, z, m_params.feaMapSize, m_params.featureType, m_params.sigma);
    cv::Mat spec;
    cv::mulSpectrums(m_params.Alpha, FFTTools::fftd(kernel), spec, 0, false);
    cv::Mat response = FFTTools::real(FFTTools::fftd(spec, true));
    //===== Compute PSR score
    psr_value = calcPsr(response);
    //===== Find peak in the response map
    cv::Point2i peakLoc_i;
    double peakVal;
    cv::minMaxLoc(response, NULL, &peakVal, NULL, &peakLoc_i);
    peak_value = (float)peakVal;
    //===== Adjust location of the peak
    cv::Point2f peakLoc_f((float)peakLoc_i.x, (float)peakLoc_i.y);

    if ((peakLoc_f.x > 0) && (peakLoc_f.x < (response.cols - 1))) {
        peakLoc_f.x += fixPeak(response.at<float>(peakLoc_i.y, peakLoc_i.x - 1),
                               peak_value,
                               response.at<float>(peakLoc_i.y, peakLoc_i.x + 1));
    }

    if ((peakLoc_f.y > 0) && (peakLoc_f.y < (response.rows - 1))) {
        peakLoc_f.y += fixPeak(response.at<float>(peakLoc_i.y - 1, peakLoc_i.x),
                               peak_value,
                               response.at<float>(peakLoc_i.y + 1, peakLoc_i.x));
    }

    peakLoc_f.x -= (float)response.cols / 2.0;
    peakLoc_f.y -= (float)response.rows / 2.0;
    return peakLoc_f;
}


//!
//! \brief ITrack::train
//! \param x
//! \param learning_rate
//!
void ITrack::train(cv::Mat &x, float learning_rate)
{
    cv::Mat k = gaussianCorrelation(x, x, m_params.feaMapSize, m_params.featureType, m_params.sigma);
    cv::Mat alphaF;

    if (!m_trackInited) {
        try {
            m_fftK = FFTTools::fftd(k);
//            printf("%s: fftK size: [%d x %d] \tm_params.Y size: [%d x %d]\n", __FUNCTION__, m_fftK.cols, m_fftK.rows, m_params.Y.cols, m_params.Y.rows);
            alphaF = FFTTools::complexDivision(m_params.Y, m_fftK + m_params.lamda);
        } catch (std::exception &e) {
            printf("%s: Exception: complexDivision 1: %s\n", __FUNCTION__, e.what());
        }
    } else {
        cv::Mat fftK = FFTTools::fftd(k);
        m_fftK = (1 - learning_rate) * m_fftK + learning_rate * fftK;
        alphaF = FFTTools::complexDivision(m_params.Y, m_fftK + m_params.lamda);
    }

    m_tmpl          = (1 - learning_rate) * m_tmpl + learning_rate * x;
    m_params.Alpha  = (1 - learning_rate) * m_params.Alpha + learning_rate * alphaF;
}



//!
//! \brief ITrack::initTrack
//! \param _image
//! \param _selRoi
//!
void ITrack::initTrack(cv::Mat &_image, cv::Rect _selRoi)
{
    //===== Initialize feature template of object tracked
    //commented by minhbq6 since assert here will terminate program when condition is satisfied
    //    assert((_selRoi.width > 0) && (_selRoi.height > 0));
    if (_selRoi.width > 0 && _selRoi.height > 0) {
        m_objRoi = _selRoi;
        m_orgRoi = _selRoi;
        //----- Determine template size
        float paddedW = m_objRoi.width * m_params.paddRatio;
        float paddedH = m_objRoi.height * m_params.paddRatio;
        m_params.scale = (paddedW > paddedH) ? (paddedW / TEMPLATE_SIZE) : (paddedH / TEMPLATE_SIZE);
//        printf("%s: m_params.scale: %f\n", __FUNCTION__, m_params.scale);
        m_params.tmplSize.width  = (int)(paddedW / m_params.scale);
        m_params.tmplSize.height = (int)(paddedH / m_params.scale);

        if (m_params.featureType == FEATURE_HOG) {
            m_params.tmplSize.width  = (m_params.tmplSize.width / (2 * CELL_SIZE) + 1) * (2 * CELL_SIZE);
            m_params.tmplSize.height = (m_params.tmplSize.height / (2 * CELL_SIZE) + 1) * (2 * CELL_SIZE);
        } else {
            m_params.tmplSize.width  = (m_params.tmplSize.width / 2) * 2;
            m_params.tmplSize.height = (m_params.tmplSize.height / 2) * 2;
        }

        //
        //----- Extract template
        float   cx = m_objRoi.x + m_objRoi.width / 2.0f,
                cy = m_objRoi.y + m_objRoi.height / 2.0f;
        cv::Mat initPatch = getPatch(_image, cx, cy, 1.0f, cv::Size(m_objRoi.width, m_objRoi.height));
//        printf("%s: initPatch size: [%d x %d]\n", __FUNCTION__, initPatch.cols, initPatch.rows);
        m_tmpl = extractFeatures(initPatch);
        m_params.feaMapSize[0] = m_params.tmplSize.height / CELL_SIZE - 2;
        m_params.feaMapSize[1] = m_params.tmplSize.width / CELL_SIZE - 2;
        m_params.feaMapSize[2] = m_tmpl.rows;
//        printf("%s: tmplSize: [%d, %d]\n", __FUNCTION__, m_params.tmplSize.width, m_params.tmplSize.height);
//        printf("%s: m_params.feaMapSize: [%d, %d, %d]\n", __FUNCTION__, m_params.feaMapSize[0], m_params.feaMapSize[1], m_params.feaMapSize[2]);
        //----- Create filter
        createHannWindow(m_params.HannWin, m_params.feaMapSize, m_params.featureType);
        m_tmpl = m_params.HannWin.mul(m_tmpl);
        m_params.Y = createGaussianDistribution(m_params.feaMapSize[0], m_params.feaMapSize[1]);
//        printf("%s: m_params.Y = [%d x %d]\n", __FUNCTION__,  m_params.Y.cols, m_params.Y.rows);
        m_params.Alpha = cv::Mat::zeros(m_params.feaMapSize[0], m_params.feaMapSize[1], CV_32FC2);
        m_fftK = cv::Mat::zeros(m_params.feaMapSize[0], m_params.feaMapSize[1], CV_32FC2);
        //----- Init training
        train(m_tmpl, 1.0f);
        //===== Init Kalman predictor
        m_kalmanFilter.setDefaultModel();
        m_kalmanFilter.setState(0.0, 0.0, 0.0, 0.0);
        m_trackStatus = TRACK_INVISION;
        m_trackInited = true;
        m_trackLostCnt = 0;
    }
}


//!
//! \brief ITrack::performTrack
//! \param _image
//!
#define DEBUG_PERFORMANCE_OFF
void ITrack::performTrack(cv::Mat &_image)
{
    if (m_trackStatus == TRACK_LOST) {
        m_objRoi = cv::Rect(-1, -1, -1, -1);
        return;
    }

    m_running = true;
    //===== 1. Predict new position of object in current frame
    float   prevCx = (float)m_objRoi.x + (float)m_objRoi.width / 2.0f,
            prevCy = (float)m_objRoi.y + (float)m_objRoi.height / 2.0f;
    float dx_kalman, dy_kalman;
    m_kalmanFilter.predictMotion(&dx_kalman, &dy_kalman);
    float   newCx =  prevCx + dx_kalman,
            newCy =  prevCy + dy_kalman;
    //===== 2. Tracking using Kernel Correlation Filter for different scales
    float peakVal, psrVal, bestPeakVal, bestPsrVal;
    cv::Point2f peakLoc, bestPeakLoc;
    cv::Mat patch, feature, bestPatch, bestFeature;
    float scale_change = 1.0f;
    //----- Apply KCF with scale = 1
    bestPatch   = getPatch(_image, newCx, newCy, 1.0, cv::Size(m_objRoi.width, m_objRoi.height));
    bestFeature = extractFeatures(bestPatch);
    bestPeakLoc = fastdetect(m_tmpl, bestFeature, bestPeakVal, bestPsrVal);
    //----- Apply KCF with scale < 1
    patch    = getPatch(_image, newCx, newCy, 1.0 / SCALE_CHANGE_RATIO, cv::Size(m_objRoi.width, m_objRoi.height));
    feature  = extractFeatures(patch);
    peakLoc  = fastdetect(m_tmpl, feature, peakVal, psrVal);
    peakVal *= SCALE_WEIGTH;

    if (peakVal > bestPeakVal) {
        bestPeakLoc = peakLoc;
        bestPeakVal = peakVal;
        bestPsrVal  = psrVal;
        scale_change = 1 / SCALE_CHANGE_RATIO;
    }

    //----- Apply KCF with scale > 1
    patch   = getPatch(_image, newCx, newCy, SCALE_CHANGE_RATIO, cv::Size(m_objRoi.width, m_objRoi.height));
    feature = extractFeatures(patch);
    peakLoc = fastdetect(m_tmpl, feature, peakVal, psrVal);
    peakVal *= SCALE_WEIGTH;

    if (peakVal >  bestPeakVal) {
        bestPeakLoc = peakLoc;
        bestPeakVal = peakVal;
        bestPsrVal  = psrVal;
        scale_change = SCALE_CHANGE_RATIO;
    }

    p = bestPeakVal;

    //    //----- Training
    //    if(  isTextureLess( bestPatch ) )
    //    {
    //        m_trackStatus = TRACK_LOST;
    //        m_objRoi = cv::Rect(-1, -1, -1, -1);
    //        return;
    //    }

    if ((bestPsrVal < 4.0) || isTextureLess(bestPatch)) {
        m_trackStatus = TRACK_OCCLUDED;
    } else {
        m_trackStatus = TRACK_INVISION;
    }

    if (m_trackStatus == TRACK_INVISION) {
        m_trackLostCnt--;

        if (m_trackLostCnt < 0) {
            m_trackLostCnt = 0;
        }

        //----- Update model base on best scale match
        m_params.scale *= scale_change;
        m_objRoi.width = (int)((float)m_objRoi.width * scale_change + 0.5);
        m_objRoi.height = (int)((float)m_objRoi.height * scale_change + 0.5);

        if (m_objRoi.height < 12) {
            m_objRoi.height = 12;
        }

        if (m_objRoi.width < 12) {
            m_objRoi.width = 12;
        }

        m_objRoi.x =  newCx - m_objRoi.width / 2.0f + ((float) bestPeakLoc.x * m_params.scale * CELL_SIZE);
        m_objRoi.y =  newCy - m_objRoi.height / 2.0f + ((float) bestPeakLoc.y * m_params.scale * CELL_SIZE);

        if (m_objRoi.x >= _image.cols - 1) {
            m_objRoi.x = _image.cols - 1;
        }

        if (m_objRoi.y >= _image.rows - 1) {
            m_objRoi.y = _image.rows - 1;
        }

        if (m_objRoi.x + m_objRoi.width <= 0) {
            m_objRoi.x = -m_objRoi.width + 2;
        }

        if (m_objRoi.y + m_objRoi.height <= 0) {
            m_objRoi.y = -m_objRoi.height + 2;
        }

        assert(m_objRoi.width >= 0 && m_objRoi.height >= 0);
        //----- Training
        newCx = (float)m_objRoi.x + (float)m_objRoi.width / 2.0f;
        newCy = (float)m_objRoi.y + (float)m_objRoi.height / 2.0f;
        cv::Mat trainPatch = getPatch(_image, newCx, newCy, 1.0, cv::Size(m_objRoi.width, m_objRoi.height));
        cv::Mat trainFeature = extractFeatures(trainPatch);
        train(trainFeature, 0.015);
        dx_kalman = newCx - prevCx;
        dy_kalman = newCy - prevCy;
        m_kalmanFilter.correctModel(0.8 * dx_kalman, 0.8 * dy_kalman);
    } else {
        m_trackLostCnt++;
        m_objRoi.x = newCx - m_objRoi.width / 2.0f;
        m_objRoi.y = newCy - m_objRoi.width / 2.0f;

        if (m_objRoi.x >= _image.cols - 1) {
            m_objRoi.x = _image.cols - 1;
        }

        if (m_objRoi.y >= _image.rows - 1) {
            m_objRoi.y = _image.rows - 1;
        }

        if (m_objRoi.x + m_objRoi.width <= 0) {
            m_objRoi.x = -m_objRoi.width + 2;
        }

        if (m_objRoi.y + m_objRoi.height <= 0) {
            m_objRoi.y = -m_objRoi.height + 2;
        }

        assert(m_objRoi.width >= 0 && m_objRoi.height >= 0);
        m_kalmanFilter.correctModel(0.5 * dx_kalman, 0.5 * dy_kalman);
    }

    // Avoiding movements larger than object size
    if ((fabs((double)dx_kalman) > (0.3 * (double)m_objRoi.width)) || (fabs((double)dy_kalman) > (0.3 * (double)m_objRoi.height))) {
        m_trackLostCnt++;
    }

    if ((m_objRoi.x + m_objRoi.width) >= (_image.cols - 10) || (m_objRoi.y + m_objRoi.height) >= (_image.rows - 10) || (m_objRoi.x <= 10) || (m_objRoi.y <= 10)) {
        m_trackLostCnt++;
    }

    if (m_trackLostCnt > 15) {
        m_trackStatus = TRACK_LOST;
    }

    if ((m_objRoi.width * m_objRoi.height) > (m_orgRoi.width * m_orgRoi.height * SCALE_CHANGE_RATIO)) {
        m_zoomDirection = 1;
    } else if ((m_objRoi.width * m_objRoi.height) < (m_orgRoi.width * m_orgRoi.height / SCALE_CHANGE_RATIO)) {
        m_zoomDirection = -1;
    } else {
        m_zoomDirection = 0;
    }

    m_running = false;
}


//!
//! \brief ITrack::getPosition
//! \return
//!
cv::Rect ITrack::getPosition()
{
    return m_objRoi;
}



//!
//! \brief ITrack::trackStatus
//! \return
//!
int ITrack::trackStatus()
{
    return m_trackStatus;
}
bool ITrack::isInitialized()
{
    return m_trackInited;
}
void ITrack::resetTrack()
{
    m_tmpl.release();
    m_params.Alpha.release();
    m_params.Y.release();
    m_params.HannWin.release();
    m_trackInited = false;
}


bool ITrack::isRunning()
{
    return m_running;
}
int ITrack::getZoomDirection()
{
    return m_zoomDirection;
}



std::string ITrack::getZoomEO(const float &_inIRFov, const float &_inEOFov, float *_deltaEOFov)
{
    int maxIRSize = (m_objRoi.width > m_objRoi.height) ? m_objRoi.width : m_objRoi.height;
    double eoCurrSize = EO_MAX_SIZE / IR_MAX_SIZE
                        * tan((double)_inIRFov / 2.0f) / tan((double)_inEOFov / 2.0f)
                        * maxIRSize;
    double tanEONewFov = EO_MAX_SIZE / IR_MAX_SIZE
                         * (double)maxIRSize / EO_STABLE_OBJ_SIZE
                         * tan((double)_inIRFov / 2.0f);
    double EONewFov = 2.0f * atan(tanEONewFov);

    if (EONewFov < EO_MIN_FOV) {
        EONewFov = EO_MIN_FOV;
    }

    if (EONewFov > EO_MAX_FOV) {
        EONewFov = EO_MAX_FOV;
    }

    //    printf( "IRSize = %d \t _inIRFov = %f \t _inEOFov = %f\n",
    //            maxIRSize, _inIRFov, _inEOFov );
    //    printf( "eoCurrSize = %f \t EONewFov = %f\n", eoCurrSize, EONewFov );
    std::string zoomDirection;

    if (EONewFov < _inEOFov) {
        zoomDirection = "ZOOM_IN";
    } else if (EONewFov > _inEOFov) {
        zoomDirection = "ZOOM_OUT";
    } else {
        zoomDirection = "ZOOM_STOP";
    }

    *_deltaEOFov = (float)EONewFov - _inEOFov;
    return zoomDirection;
}


std::string ITrack::getZoomIR(const float &_inIRFov)
{
    int maxObjSize = (m_objRoi.width > m_objRoi.height) ? m_objRoi.width : m_objRoi.height;

    if ((_inIRFov > IR_MIN_FOV) && (maxObjSize < IR_MIN_OBJ_SIZE)) {
        return "ZOOM_IN";
    }

    if ((_inIRFov < IR_MAX_FOV) && (maxObjSize > IR_MAX_OBJ_SIZE)) {
        return "ZOOM_OUT";
    }

    return "ZOOM_STOP";
}

/**
 * @brief isTextureLess
 * @param patch
 * @return
 */
bool isTextureLess(cv::Mat &patch)
{
    cv::Mat tmp;
    double diag = sqrt(patch.cols * patch.cols + patch.rows * patch.rows);
    int kSize = (int)(diag / 10.0 + 0.5);

    if (kSize < 3) {
        kSize = 3;
    }

    kSize = (kSize / 2) * 2 + 1;
    cv::medianBlur(patch, tmp, kSize);
    double minVal, maxVal;
    cv::minMaxLoc(tmp, &minVal, &maxVal, NULL, NULL);

    if ((maxVal - minVal) < 30 || maxVal == 0) {
        return true;
    }

    float rate = (maxVal - minVal) / maxVal;

    if (rate < 0.1) {
        return true;
    }

    return false;
}