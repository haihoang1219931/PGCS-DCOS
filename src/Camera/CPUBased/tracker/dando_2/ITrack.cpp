#include <chrono>

#include "ITrack.hpp"
#include "HTrack/fhog.hpp"

#define NONDEBUG

//!
//! \brief ITrack::ITrack
//! \param _featureType
//! \param _kernelType
//!
ITrack::ITrack( int _featureType, int _kernelType ) : m_params( _featureType, _kernelType )
{
    m_trackInited  = false;
    m_trackStatus  = false;
    m_occlusionCnt = 0;
    m_running      = false;

    //===== Initialize Kalman predictor
    m_kalmanFilter.getDefaultState();
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
    if( m_trackInited )
    {
        roi.width  = (int)(scale * (float)target_sz.width * m_params.paddRatio );
        roi.height = (int)(scale * (float)target_sz.height * m_params.paddRatio );
        roi.x      = ctx - roi.width / 2;
        roi.y      = cty - roi.height / 2;
    }
    else
    {
        roi.width  = target_sz.width; //(int)(m_params.scale * m_params.tmplSize.width);
        roi.height = target_sz.height; //(int)(m_params.scale * m_params.tmplSize.height);
        roi.x      = ctx - roi.width / 2;
        roi.y      = cty - roi.height / 2;
    }

    std::vector<int> border = limitRect( roi, cv::Size(img.cols, img.rows) );
    patch = img(roi).clone();
    if( (border[0] != 0) || (border[1] != 0) || (border[2] != 0) || (border[3] != 0) )
    {
        cv::copyMakeBorder(patch, patch, border[0], border[1], border[2], border[3], cv::BORDER_CONSTANT);
    }
    if( (patch.cols != m_params.tmplSize.width) || (patch.rows != m_params.tmplSize.height) )
    {
        cv::resize( patch, patch, m_params.tmplSize );          // ??? Should I crop for indentical ratio
    }

    return patch;
}


bool ITrack::isInitialized(){
    return m_trackInited;
}
void ITrack::resetTrack(){
    m_tmpl.release();
    m_params.Alpha.release();
    m_params.Y.release();
    m_params.HannWin.release();

    m_trackInited = false;
}


//!
//! \brief ITrack::extractFeatures          Extract feature map from given image patch
//! \param patch
//! \return
//!
cv::Mat ITrack::extractFeatures( cv::Mat &patch )
{
    assert( patch.channels() == 1 );

    cv::Mat featureMap;
    if( m_params.featureType == FEATURE_HOG )
    {
        IplImage iplPatch = patch;
        CvLSVMFeatureMapCaskade *map;
        getFeatureMaps( &iplPatch, CELL_SIZE, &map );
        normalizeAndTruncate( map, 0.2f );
        PCAFeatureMaps( map );          // each reduced feature vector has 31 components

        featureMap = cv::Mat( cv::Size(map->numFeatures, map->sizeX*map->sizeY), CV_32F, map->map );
        featureMap = featureMap.t();
        if( m_trackInited )
        {
            featureMap = m_params.HannWin.mul( featureMap );
        }
        freeFeatureMapObject( &map );
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
cv::Point2f ITrack::fastdetect( cv::Mat &z, cv::Mat &x, float &peak_value, float &psr_value )
{
    //===== Phase Correlation Computation
    cv::Mat kernel;
    if( m_params.kernelType == KERNEL_GAUSSIAN )
    {
        kernel = gaussianCorrelation( x, z, m_params.feaMapSize, m_params.featureType, m_params.sigma );
    }
    else if( m_params.kernelType == KERNEL_POLYNOMIAL )
    {
        kernel = polynomialCorrelation( x, z, m_params.feaMapSize, m_params.featureType, m_params.alpha, m_params.beta );
    }
    cv::Mat spec;
    cv::mulSpectrums( m_params.Alpha, FFTTools::fftd( kernel ), spec, 0, false );
    cv::Mat response = FFTTools::real( FFTTools::fftd( spec, true ) );

    //===== Compute PSR score
    psr_value = calcPsr( response );

    //===== Find peak in the response map
    cv::Point2i peakLoc_i;
    double peakVal;
    cv::minMaxLoc( response, NULL, &peakVal, NULL, &peakLoc_i );
    peak_value = (float)peakVal;

    //===== Adjust location of the peak
    cv::Point2f peakLoc_f( (float)peakLoc_i.x, (float)peakLoc_i.y );
    if( (peakLoc_f.x > 0) && (peakLoc_f.x < (response.cols - 1)) )
    {
        peakLoc_f.x += fixPeak( response.at<float>(peakLoc_i.y, peakLoc_i.x-1),
                                peak_value,
                                response.at<float>(peakLoc_i.y, peakLoc_i.x+1));
    }

    if( (peakLoc_f.y > 0) && (peakLoc_f.y < (response.rows - 1)) )
    {
        peakLoc_f.y += fixPeak( response.at<float>(peakLoc_i.y-1, peakLoc_i.x),
                                peak_value,
                                response.at<float>(peakLoc_i.y+1, peakLoc_i.x) );
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
void ITrack::train( cv::Mat &x, float learning_rate )
{

    cv::Mat k = gaussianCorrelation( x, x, m_params.feaMapSize, m_params.featureType, m_params.sigma );
    cv::Mat alphaF;
    cv::Mat k2 = doubleGaussCorrelation( x, x, m_params.Alpha, m_params.sigma );
    float   tmp1 = m_params.lamda1 / (m_params.lamda * m_params.lamda),
            tmp2 = (1 + 2*m_params.lamda1) / m_params.lamda;
    if( !m_trackInited )
    {
        try
        {
            cv::Mat fftK = FFTTools::fftd( k );
            printf( "%s: fftK size: [%d x %d] \tm_params.Y size: [%d x %d]\n", __FUNCTION__, fftK.cols, fftK.rows, m_params.Y.cols, m_params.Y.rows );
            alphaF = FFTTools::complexDivision( m_params.Y, fftK + m_params.lamda );
        }
        catch (std::exception &e )
        {
            printf( "%s: Exception: complexDivision 1: %s\n", __FUNCTION__, e.what() );
        }
    }
    else
    {
        cv::Mat yk;
        try
        {
            cv::mulSpectrums( ((m_params.lamda1/m_params.lamda)* FFTTools::fftd(k) + m_params.lamda1), m_params.Y, yk, 0, true );
        }
        catch (std::exception &e )
        {
            printf( "%s: Exception: mulSpectrums: %s\n", __FUNCTION__, e.what() );
        }

        try
        {
            alphaF = FFTTools::complexDivision( yk, tmp1 * FFTTools::fftd( k2 ) + tmp2 * FFTTools::fftd( k ) + 1 + m_params.lamda1 );
        }
        catch (std::exception &e )
        {
            printf( "%s: Exception: complexDivision 2: %s\n", __FUNCTION__, e.what() );
        }
    }

    m_tmpl          = (1 - learning_rate) * m_tmpl + learning_rate * x;
    m_params.Alpha  = (1 - learning_rate) * m_params.Alpha + learning_rate * alphaF;
}



//!
//! \brief ITrack::initTrack
//! \param _image
//! \param _selRoi
//!
void ITrack::initTrack( cv::Mat &_image, cv::Rect _selRoi )
{
    //===== Initialize feature template of object tracked
    assert( (_selRoi.width > 0) && (_selRoi.height > 0) );
    assert( _image.channels() == 1 );
    m_objRoi = _selRoi;

    //----- Determine template size
    int paddedW = m_objRoi.width * m_params.paddRatio;
    int paddedH = m_objRoi.height * m_params.paddRatio;
    m_params.scale = (paddedW > paddedH)? ((float)paddedW / TEMPLATE_SIZE) : ((float)paddedH / TEMPLATE_SIZE);

    printf( "%s: m_params.scale: %f\n", __FUNCTION__, m_params.scale);

    m_params.tmplSize.width  = (int)(paddedW / m_params.scale);
    m_params.tmplSize.height = (int)(paddedH / m_params.scale);
    if( m_params.featureType == FEATURE_HOG )
    {
        m_params.tmplSize.width  = (m_params.tmplSize.width / (2 * CELL_SIZE) + 1) * (2 * CELL_SIZE);
        m_params.tmplSize.height = (m_params.tmplSize.height / (2 * CELL_SIZE) + 1) * (2 * CELL_SIZE);
    }
    else
    {
        m_params.tmplSize.width  = (m_params.tmplSize.width / 2) * 2;
        m_params.tmplSize.height = (m_params.tmplSize.height / 2) * 2;
    }

    std::cout << "_selRoi = " << _selRoi << std::endl;
    printf( "%s: m_params.tmplSize: [%d, %d]\n", __FUNCTION__, m_params.tmplSize.width, m_params.tmplSize.height );

    m_params.feaMapSize[0] = m_params.tmplSize.height / CELL_SIZE - 2;
    m_params.feaMapSize[1] = m_params.tmplSize.width / CELL_SIZE - 2;
    m_params.feaMapSize[2] = 31;

    printf( "%s: m_params.feaMapSize: [%d, %d, %d]\n", __FUNCTION__, m_params.feaMapSize[0], m_params.feaMapSize[1], m_params.feaMapSize[2] );


    //----- Create filter
    createHannWindow( m_params.HannWin, m_params.feaMapSize, m_params.featureType );
    std::cout << "m_params.HannWin.size = " << m_params.HannWin.size() << std::endl;
    m_params.Y = createGaussianDistribution( m_params.feaMapSize[0], m_params.feaMapSize[1] );

    printf( "%s: m_params.Y = [%d x %d]\n",__FUNCTION__,  m_params.Y.cols, m_params.Y.rows );
    m_params.Alpha = cv::Mat::zeros( m_params.feaMapSize[0], m_params.feaMapSize[1], CV_32FC2 );

    //----- Extract template
    float   cx = m_objRoi.x + m_objRoi.width / 2.0f,
            cy = m_objRoi.y + m_objRoi.height / 2.0f;
    cv::Mat initPatch = getPatch( _image, cx, cy, 1.0f, cv::Size(paddedW, paddedH) );

    printf( "%s: initPatch size: [%d x %d]\n", __FUNCTION__, initPatch.cols, initPatch.rows );

    std::cout << "initPatch.size = " << initPatch.size() << "\t m_tmpl.size = " << m_tmpl.size() << std::endl;
    m_tmpl = extractFeatures( initPatch );
    m_tmpl = m_params.HannWin.mul( m_tmpl );

    //----- Init training
    train( m_tmpl, 1.0f );

    //===== Init Kalman predictor
    m_kalmanFilter.initState( 0.0f, 0.0f );

    m_trackStatus = true;
    m_trackInited = true;
}


//!
//! \brief ITrack::performTrack
//! \param _image
//!
void ITrack::performTrack( cv::Mat &_image )
{
    if( m_trackStatus == false )
    {
        printf( "========> Object Lost\n" );
        return;
    }

    m_running = true;

    //===== 1. Predict new position of object in current frame
    double  prevX = (double)m_objRoi.x + (double)m_objRoi.width / 2.0f,
            prevY = (double)m_objRoi.y + (double)m_objRoi.height / 2.0f;

    //----- Using GME
    cv::Mat trans = cv::Mat::eye( 3, 3, CV_64F );
    m_gme.run( _image, trans, false );

    cv::Mat_<double> currCenter( 3, 1 );
    currCenter << prevX, prevY, 1.0;

    cv::Mat gmeCenter = trans * currCenter;
    float  currX = (float)(gmeCenter.at<double>(0) / gmeCenter.at<double>(2)),
           currY = (float)(gmeCenter.at<double>(1) / gmeCenter.at<double>(2));

    //----- Predict using Kalman Model
    m_kalmanFilter.predict();


    //===== 2. Tracking using Kernel Correlation Filter for different scales
    float peakVal, psrVal, bestPeakVal, bestPsrVal;
    cv::Point2f peakLoc, bestPeakLoc;
    cv::Mat patch, feature, bestPatch, bestFeature;
    float scale_change = 1.0f;
    float dx_kalman, dy_kalman;

    //----- Apply KCF with scale = 1
    bestPatch   = getPatch( _image, currX, currY, 1.0, cv::Size(m_objRoi.width, m_objRoi.height) );
    bestFeature = extractFeatures( bestPatch );
    bestPeakLoc = fastdetect( m_tmpl, bestFeature, bestPeakVal, bestPsrVal );
//    printf( "scale = %f: PeakVal = %f \t PsrVal = %f\n", 1.0, bestPeakVal, bestPsrVal );

    //----- Apply KCF with scale < 1
    patch    = getPatch( _image, currX, currY, 1.0 / SCALE_CHANGE_RATIO, cv::Size(m_objRoi.width, m_objRoi.height) );
    feature  = extractFeatures( patch );
    peakLoc  = fastdetect( m_tmpl, feature, peakVal, psrVal );
//    printf( "scale = %f: PeakVal = %f \t PsrVal = %f\n", 1.0 / SCALE_CHANGE_RATIO, peakVal, psrVal );

    peakVal *= SCALE_WEIGTH;
    if( peakVal > bestPeakVal )
    {
        bestPeakLoc = peakLoc;
        bestPeakVal = peakVal;
        bestPsrVal  = psrVal;
        scale_change= 1 / SCALE_CHANGE_RATIO;
    }

    //----- Apply KCF with scale > 1
    patch   = getPatch( _image, currX, currY, SCALE_CHANGE_RATIO, cv::Size(m_objRoi.width, m_objRoi.height) );
    feature = extractFeatures( patch );
    peakLoc = fastdetect( m_tmpl, feature, peakVal, psrVal );
//    printf( "scale = %f: PeakVal = %f \t PsrVal = %f\n", SCALE_CHANGE_RATIO, peakVal, psrVal );

    peakVal *= SCALE_WEIGTH;
    if( peakVal >  bestPeakVal )
    {
        bestPeakLoc = peakLoc;
        bestPeakVal = peakVal;
        bestPsrVal  = psrVal;
        scale_change= SCALE_CHANGE_RATIO;
    }

#ifdef DEBUG
//    printf( "%s: bestPeaKVal = %f  -  bestPsrVal = %f  -  bestPeakLoc = (%f; %f)  -  bestScaleChange = %f\n", __FUNCTION__,
//            bestPeakVal, bestPsrVal, bestPeakLoc.x, bestPeakLoc.y, scale_change );
//    printf( "%s: Input: center = (%f; %f)  -  size = [%d x %d]  -  scale = %f\n", __FUNCTION__,
//            currX, currY, m_objRoi.width, m_objRoi.height, m_params.scale );
#endif

    if( bestPsrVal > 4.0 )
    {
        //----- Update model base on best scale match
        m_params.scale *= scale_change;
        m_objRoi.width *= scale_change;
        m_objRoi.height*= scale_change;

        m_objRoi.x =  currX - m_objRoi.width/ 2.0f + ((float) bestPeakLoc.x * m_params.scale * CELL_SIZE );
        m_objRoi.y =  currY - m_objRoi.height/ 2.0f +((float) bestPeakLoc.y * m_params.scale * CELL_SIZE  );
        if (m_objRoi.x >= _image.cols - 1) m_objRoi.x = _image.cols - 1;
        if (m_objRoi.y >= _image.rows - 1) m_objRoi.y = _image.rows - 1;
        if (m_objRoi.x + m_objRoi.width <= 0) m_objRoi.x = -m_objRoi.width + 2;
        if (m_objRoi.y + m_objRoi.height <= 0) m_objRoi.y = -m_objRoi.height + 2;
        assert( m_objRoi.width >= 0 && m_objRoi.height >= 0 );

        float   newCx = (float)m_objRoi.x + (float)m_objRoi.width / 2.0f,
                newCy = (float)m_objRoi.y + (float)m_objRoi.height / 2.0f;

#ifdef DEBUG
//        printf( "%s: Output: center = (%f; %f)  -  size = [%d x %d]  -  scale = %f\n", __FUNCTION__,
//                newCx, newCy, m_objRoi.width, m_objRoi.height, m_params.scale );
#endif
        //----- Training model
        cv::Mat trainPatch = getPatch( _image, newCx, newCy, 1.0, cv::Size(m_objRoi.width, m_objRoi.height) );
        cv::Mat trainFeature = extractFeatures( trainPatch );

        train( trainFeature, 0.015 );

        //----- Update Kalman Model
        dx_kalman = newCx - currX;
        dy_kalman = newCy - currY;

        m_occlusionCnt = 0;
    }
    else
    {
        std::cout << "====> Obect Occluded\n";
        m_kalmanFilter.getMotion( &dx_kalman, &dy_kalman );
        currX += dx_kalman;
        currY += dy_kalman;

        m_objRoi.x = (int)(currX - m_objRoi.width / 2.0f);
        m_objRoi.y = (int)(currY - m_objRoi.height / 2.0f);

        m_occlusionCnt++;
        if( m_occlusionCnt >= 20 )
        {
            m_trackStatus = false;
        }
    }
#ifdef DEBUG
    printf( "dx_kalman = %f  -  dy_kalman = %f\n", dx_kalman, dy_kalman );
#endif
    m_kalmanFilter.correct( dx_kalman, dy_kalman );

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
bool ITrack::trackStatus()
{
    return m_trackStatus;
}


int ITrack::getState()
{
    return (m_trackStatus)? FOUND : LOST;
}
