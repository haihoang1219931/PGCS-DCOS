
#include "Utilities.hpp"

//!
//! \brief limitRect        Confine window dimension in image region and compute border croppped
//! \param window           Initial/Output window dimension
//! \param im_sz            Image size
//! \return
//!
std::vector<int> limitRect( cv::Rect &window, cv::Size im_sz)
{
    //===== Confine the window dimension so that it fits in the image region
    int     top   = window.y,
            left  = window.x,
            bot   = window.y + window.height,
            right = window.x + window.width;
    if( top < 0 )
        top = 0;
    if( left < 0 )
        left = 0;
    if( bot >= im_sz.height )
        bot = im_sz.height - 1;
    if( right >= im_sz.width )
        right = im_sz.width - 1;

    cv::Rect newWindow;
    newWindow.x      = left;
    newWindow.y      = top;
    newWindow.width  = right - left + 1;
    newWindow.height = bot - top + 1;
    assert( (newWindow.width > 0) && (newWindow.height > 0) );

    //===== Compute cut border
    std::vector<int>cutBorder;
    cutBorder.push_back(top - window.y);                    // border size on top
    cutBorder.push_back(window.y + window.height - bot);    // border size below bot
    cutBorder.push_back(left - window.x);                   // border size on the left
    cutBorder.push_back(window.x + window.width - right);   // border size on the right
    assert( cutBorder[0] >= 0 && cutBorder[1] >= 0 && cutBorder[2] >=0 && cutBorder[3] >=0);

    window = newWindow;

    return cutBorder;
}



//!
//! \brief createGaussianLabels
//! \param sizey
//! \param sizex
//! \return
//!
cv::Mat createGaussianDistribution(int sizey, int sizex)
{
    cv::Mat_<float> kernel( sizey, sizex );

    float   centerX = (float)sizex / 2.0f,
            centerY = (float)sizey / 2.0f;
    float   sigma   = sqrt( (float)(sizex * sizey) ) / PADDING * OUTPUT_SIGMA_FACTOR;       // ???
    float   power   = - 0.5 / (sigma * sigma);


    for( int i = 0; i < sizey; i++ )
    {
        for( int j = 0; j < sizex; j++ )
        {
            float dx = (float)(j - centerX);
            float dy = (float)(i - centerY);
            kernel.at<float>(i, j) = exp( power * (dx*dx + dy*dy) );
        }
    }

    return FFTTools::fftd( kernel );
}


//!
//! \brief createHannWindow
//! \param _hann
//! \param width
//! \param height
//!
void createHannWindow( cv::Mat &_hann, int *size_path, int featureType )
{
    cv::Mat hannX = cv::Mat::zeros( 1, size_path[1], CV_32FC1 );
    cv::Mat hannY = cv::Mat::zeros( size_path[0], 1, CV_32FC1 );

    for( int i = 0; i < hannX.cols; i++ )
    {
        hannX.at<float>(i) = 0.5 * (1 - cos(2 * PI * i / (hannX.cols - 1)));
    }

    for( int i = 0; i < hannY.rows; i++ )
    {
        hannY.at<float>(i) = 0.5 * (1 - cos(2 * PI * i / (hannY.rows - 1)));
    }

    cv::Mat hann2d = hannY * hannX;
    if( featureType == FEATURE_HOG )
    {
        cv::Mat hann1d = hann2d.reshape(1, 1);
        _hann = cv::Mat(cv::Size(size_path[0] * size_path[1], size_path[2]), CV_32FC1, cv::Scalar(0));
        for( int i = 0; i < size_path[2]; i++ )
        {
            for( int j = 0; j < (size_path[0] * size_path[1]); j++ )
            {
                _hann.at<float>(i, j) = hann1d.at<float>(0, j);
            }
        }
    }

//    printf( "%s: _hann size: [%d x %d]", __FUNCTION__, _hann.cols, _hann.rows );
}



//!
//! \brief calcPsr
//! \param response
//! \return
//!
float calcPsr(cv::Mat &response)
{
    float psr ;
    double max_val = 0 ;
    cv::Point max_loc ;
    int psr_mask_sz = 11;
    cv::Mat psr_mask = cv::Mat::ones(response.rows, response.cols, CV_8U);
    cv::Scalar mn_ ;
    cv::Scalar std_;

    cv::minMaxLoc(response, NULL, &max_val, NULL, &max_loc);
    int win_sz = floor(psr_mask_sz/2);
    cv::Rect side_lobe = cv::Rect(std::max(max_loc.x - win_sz, 0), std::max(max_loc.y - win_sz,0),11, 11);

    if ((side_lobe.x + side_lobe.width) > psr_mask.cols)
    {
        side_lobe.width = psr_mask.cols - side_lobe.x;
    }
    if ((side_lobe.y + side_lobe.height) > psr_mask.rows)
    {
        side_lobe.height = psr_mask.rows - side_lobe.y;
    }

    cv::Mat tmp = psr_mask(side_lobe);
    tmp *= 0;
    cv::meanStdDev(response, mn_, std_, psr_mask);
    psr = (max_val - mn_[0])/ (std_[0] + std::numeric_limits<float>::epsilon());

    return  psr ;
}


//!
//! \brief fixPeak     Adjust location of a peak
//! \param left
//! \param center
//! \param right
//! \return
//!
float fixPeak(float left, float center, float right){
    float divisor = 2 * center - right - left;
    float fixp  = 0.5 * (right - left) / divisor;
//    printf(" value of ......%f\r\n", 0.5 * (right - left) / divisor);
//    if (divisor < 0.000    std::cout<< " outrect " << outrect <<std ::endl;
//    {
//        printf("==>>>>>>>\r\n");
//        divisor = 0.000003;
//        return 0.5* (left -right)/divisor;
//    }
//    if (divisor == 0)
//        return 0;
//    printf(" value of ......%f %f \r\n", 0.5 * (right - left) / divisor, divisor);
//    printf(" value of right - left : %f \r\n", right - left);
    if (fixp < 0.13f)
        fixp = 0;
    return fixp; //0.5 * (right - left) / divisor;
}



//!
//! \brief gaussianCorrelation
//! \param x1
//! \param x2
//! \return
//!
cv::Mat gaussianCorrelation(cv::Mat x1, cv::Mat x2, int *size_patch, int feature_type, float sigma){
//    printf( "%s: size_patch: [%d x %d x %d]\n", __FUNCTION__, size_patch[0], size_patch[1], size_patch[2] );
    using namespace FFTTools;
    cv::Mat c = cv::Mat(cv::Size(size_patch[1], size_patch[0]), CV_32F, cv::Scalar(0));

    if (feature_type == FEATURE_HOG) {
        cv::Mat caux;
        cv::Mat x1aux;
        cv::Mat x2aux;
        for (int i = 0; i < size_patch[2]; i++) {
            x1aux = x1.row(i);
            x1aux = x1aux.reshape(1, size_patch[0]);
            x2aux = x2.row(i).reshape(1, size_patch[0]);

            cv::mulSpectrums(fftd(x1aux), fftd(x2aux), caux, 0, true);
            caux = fftd(caux, true);
//            cv::Mat icaux;
//            cv::idft(caux, icaux, cv::DFT_REAL_OUTPUT);
//            rearrange(caux);
            caux.convertTo(caux, CV_32F);
            c = c + real(caux);

        }

    }
    // Gray features
    else {
        cv::mulSpectrums(fftd(x1), fftd(x2), c, 0, true);
        c = fftd(c, true);
        rearrange(c);
        c = real(c);
    }
//    rearrange(c);
    cv::Mat d;
    cv::max(((cv::sum(x1.mul(x1))[0] + cv::sum(x2.mul(x2))[0]) - 2. * c), 0, d);

    cv::Mat k;
    cv::exp((-d / (2*sigma * sigma)) / (size_patch[0] * size_patch[1] * size_patch[2]), k);

    return k;
}
cv::Mat doubleCorrelation(cv::Mat x1, cv::Mat x2)
{
    using namespace FFTTools;
    cv::Mat c = cv::Mat(cv::Size(x1.cols, x1.rows), CV_32F, cv::Scalar(0));
    cv::Mat caux;
    for(int i = 0; i < x1.rows; i ++)
    {
        cv::Mat x1aux = x1.row(i);
        cv::Mat x2aux = x2.row(i);
        cv::mulSpectrums(fftd(x1aux), fftd(x2aux), caux, 0);
        caux = fftd(caux, true);
        caux.convertTo(caux,CV_32F);
        c = c + real(caux);
    }
    return c;
}


//!
//! \brief EyeTrack::polynomialCorrelation
//! \param x1
//! \param x2
//! \return
//!
cv::Mat polynomialCorrelation(cv::Mat x1, cv::Mat x2, int *size_patch, int feature_type, float alpha, float beta )
{
    using namespace FFTTools;
    cv::Mat c = cv::Mat(cv::Size(size_patch[1], size_patch[0]), CV_32F, cv::Scalar(0));
    if (feature_type == FEATURE_HOG) {
        cv::Mat caux;
        cv::Mat x1aux;
        cv::Mat x2aux;
        //printf(" before reshape features \r\n");
        for (int i = 0; i < size_patch[2]; i++) {
            x1aux = x1.row(i);   // Procedure do deal with cv::Mat multichannel bug
            x1aux = x1aux.reshape(1, size_patch[0]);
            x2aux = x2.row(i).reshape(1, size_patch[0]);
            cv::mulSpectrums(fftd(x1aux), fftd(x2aux), caux, 0, true);
            caux = fftd(caux, true);
//            rearrange(caux);
            caux.convertTo(caux, CV_32F);
            c = c + real(caux);
        }
        //printf(" After reshape features \r\n");
    }
    // Gray features
    else {
        cv::mulSpectrums(fftd(x1), fftd(x2), c, 0, true);
        c = fftd(c, true);
//        rearrange(c);
        c = real(c);
    }
    cv::Mat d;
    cv::max(c/pow((( size_patch[0]* size_patch[1]* size_patch[2]) + alpha),beta ),0, d);
    cv::Mat k ;
    k = d;
    return k ;
}



//!
//! \brief EyeTrack::doubleGaussCorrelation
//! \param x1
//! \param x2
//! \return
//!
cv::Mat doubleGaussCorrelation(cv::Mat x1, cv::Mat x2, cv::Mat _alphaf, float sigma )
{
    using namespace FFTTools;
    int numf = x1.rows;
    int szX  = _alphaf.cols;
    int szY  = _alphaf.rows;
    cv::Mat c = cv::Mat(cv::Size(szX, szY), CV_32F , cv::Scalar(0));
    cv::Mat xx, yy, xy, kxy, ky;

    for( int i = 0 ; i < numf ; i ++)
    {
        xx = x1.row(i);
        xx = xx.reshape(1, szY);
        yy = x2.row(i);
        yy = yy.reshape(1, szY);
        cv::mulSpectrums(fftd(xx), fftd(yy), xy, 0 , true);
        cv::mulSpectrums(xy, fftd(xx), kxy, 0, false);
        cv::mulSpectrums(kxy, fftd(yy), ky, 0, true);
        ky = fftd(ky, true);
//        rearrange(ky);
        c = c + real(ky );
    }
//    rearrange(c);
    cv::Mat d;

    cv::Mat e;
    cv::sqrt( c, e );
    cv::max(((cv::sum(x1.mul(x1))[0] + cv::sum(x2.mul(x2))[0]) -2. * e) / (szX * szY * numf), 0 , d);

    cv::Mat k;
    cv::exp((-d / (2*sigma * sigma)), k);
    return k;

}

// @Editor: giapvn
cv::Point2f correctPeak(cv::Mat &response, float peakVal)
{
    float thres = 0.7 * peakVal;
    cv::Mat thresMat;
    cv::threshold(response, thresMat, thres, 255, CV_THRESH_BINARY);
    std::vector<std::pair<cv::Point, float>> peaks;
    for(int i = 0; i < response.rows; i++)
        for(int j = 0; j < response.cols; j++)
        {
            if(thresMat.at<float>(i, j) == 255)
            peaks.push_back(std::make_pair(cv::Point(j, i), response.at<float>(i, j) / peakVal * response.at<float>(i, j) / peakVal));
        }

    float sumX = 0,
          sumY = 0,
          sumW = 0;
    for(size_t i = 0; i < peaks.size(); i++)
    {
        sumX += peaks[i].second * peaks[i].first.x;
        sumY += peaks[i].second * peaks[i].first.y;
        sumW += peaks[i].second;
    }

    return cv::Point2f(sumX / sumW, sumY / sumW);
}
