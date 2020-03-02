
#include "gme_short.hpp"


namespace gme {


/********************************************
 *          m a i n    c l a s s
 *******************************************/
//!
//! \brief GMEstimator::ageDelay
//!
void GMEstimator::ageDelay()
{
    _curr_gray.copyTo( _prev_gray );
    _prev_pts.clear();
    _prev_pts = _curr_pts;
}


//!
//! \brief GMEstimator::run
//! \param rgb_input
//! \param trans
//! \param backward_motion
//! \return
//!
int GMEstimator::run(const vtx_image &input_rgb, vtx_mat &trans, bool backward_motion)
{
    ageDelay();

    cv::Mat rgb_resize;
    cv::Size size = input_rgb.size();
    size.width  /= 2;
    size.height /= 2;
    cv::resize( input_rgb, rgb_resize, size );

    //============= Extract key points from current frame
    if( input_rgb.channels() > 1 )
    {
        cv::cvtColor( input_rgb, _curr_gray, cv::COLOR_RGB2GRAY );
    }
    else
    {
        input_rgb.copyTo( _curr_gray );
    }



    vtx_points_vector v_tmp_pts;
    vtx_image img_roi;
    cv::Rect  bound;
    uint w = _curr_gray.size().width,
         h = _curr_gray.size().height;

    _curr_pts.clear();
    for( uint r = 0; r < VERTICA_BLKS; r++ )
    {
        for( uint c = 0; c < HORIZON_BLKS; c++ )
        {
            // capture sub-image
            bound = cv::Rect( c * w/HORIZON_BLKS, r * h/VERTICA_BLKS, w/HORIZON_BLKS, h/VERTICA_BLKS );
            img_roi = _curr_gray( bound );

            // extract key point of that region
            if( vtx_extractKeypoints( img_roi, v_tmp_pts, _keypoint_type, MAX_CORNERS / (VERTICA_BLKS * HORIZON_BLKS) ) != SUCCESS )
            {
                std::cerr << "! ERROR: vtx_StabGCS::estimate_motion(): vtx_extractKeypoints failed" << std::endl;
                return GENERIC_ERROR;
            }

            if( vtx_appendVector( _curr_pts, v_tmp_pts,  cv::Point2f(c*w/HORIZON_BLKS, r*h/VERTICA_BLKS) ) != SUCCESS )
            {
                break;
            }
        }
    }

    if( _prev_gray.empty() ) // first frame
    {
        _last_trans.copyTo( trans );
        return SUCCESS;
    }


    //============= Estimate motion
    if( !_prev_pts.empty() && !_curr_pts.empty() )
    {
        vtx_points_vector prev_pts_list, prev_pts_lk, curr_pts_lk;
        std::vector<uchar> match_status;
        std::vector<float> match_error;
        try
        {
            cv::calcOpticalFlowPyrLK( _curr_gray, _prev_gray, _curr_pts, prev_pts_list, match_status, match_error,
                                      cv::Size( 21, 21 ), 5,
                                      cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03),
                                      0, 0.001);
        }
        catch( std::exception &e )
        {
            _last_trans.copyTo( trans );
            return BAD_TRANSFORM;
        }

        prev_pts_lk.clear();
        curr_pts_lk.clear();
        for ( uint i = 0; i < _curr_pts.size(); i++ )
        {
            if ( match_status[i] )
            {
                prev_pts_lk.push_back( prev_pts_list[i] );
                curr_pts_lk.push_back( _curr_pts[i] );
            }
        }

        // only estimate transform if there are enough correspondent pairs
        if( curr_pts_lk.size() >= MIN_CORNERS )
        {
            int ret;
            if( backward_motion)
            {
                ret = vtx_estimate_transform( _motion_mode, curr_pts_lk, prev_pts_lk, trans );
            }
            else
            {
                ret = vtx_estimate_transform( _motion_mode, prev_pts_lk, curr_pts_lk, trans );
            }
            trans.at<double>(0, 2) *= 2.0f;
            trans.at<double>(1, 2) *= 2.0f;

            if( ret == SUCCESS )
            {
                trans.copyTo( _last_trans );
            }
            else if( ret == BAD_TRANSFORM )
            {
                _last_trans.copyTo( trans );
                return BAD_TRANSFORM;
            }
            else
            {
                std::cerr << "! ERROR: vtx_KIIRStabilizer::run(): vtx_estimate_transform failed" << std::endl;
                return GENERIC_ERROR;
            }
        }
    }
    _last_trans.copyTo( trans );

    return SUCCESS;
}

/********************************************
 *          f u n c t i o n s
 *******************************************/
//!
//! \brief vtx_extractKeypoints
//! \param image
//! \param pts
//! \param keypoint_type
//! \param max_pts_num
//! \return
//!
int vtx_extractKeypoints( vtx_image &image, vtx_points_vector &pts, uint keypoint_type, uint max_pts_num )
{
    if( image.empty() || image.channels() != 1){
        std::cerr << "! ERROR: extractKeypoints: invalid input image" << std::endl;
        return GENERIC_ERROR;
    }

    // Detect corners of different types
    pts.clear();
    switch( keypoint_type )
    {
    case GOOD_FEATURE: 		// Good feature to track
    {
        cv::goodFeaturesToTrack( image, pts, max_pts_num, GOODFEATURE_QUALITY, GOODFEATURE_MIN_DIS);
        try
        {
            cv::cornerSubPix( image, pts,  cv::Size(10, 10), cv::Size(-1, -1),
                              cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03));
        }
        catch( std::exception &e )
        {
            std::cerr << "! ERROR: vtx_extractKeypoints(): cv::cornerSubPix() failed" << std::endl;
        }

        break;
    }
    case FAST_CORNER:
    {
        // apply fast corner
        std::vector<cv::KeyPoint> tmp_pts;
        cv::FAST( image, tmp_pts, FASTCORNERS_MIN_DIS, true );

        pts.clear();
        uint len = tmp_pts.size();
        for ( uint i = 0; i < len; i++ )
        {
            pts.push_back( tmp_pts[i].pt);
        }

        break;
    }
    default:
    {
        std::cerr << "! ERROR: vtx_extractKeypoints: invalid corner type" << std::endl;
        return GENERIC_ERROR;
    }
    }

    return SUCCESS;
}



//!
//! \brief vtx_estimate_transform
//! \param mode
//! \param pts_1
//! \param pts_2
//! \param trans
//! \return
//!
int  vtx_estimate_transform( uint mode, vtx_points_vector &pts_1, vtx_points_vector &pts_2, vtx_mat &trans )
{
    assert( pts_1.size() == pts_2.size() );

    std::vector<uchar> inlier;   // store the inlier status when using robus estimation method (RANSAC)

    auto convert = [&](vtx_mat &m)
    {
        vtx_mat result = vtx_mat::eye(3, 3, CV_64F);
        result.at<double>(0, 0) = m.at<double>(0, 0);
        result.at<double>(0, 1) = m.at<double>(0, 1);
        result.at<double>(0, 2) = m.at<double>(0, 2);
        result.at<double>(1, 0) = m.at<double>(1, 0);
        result.at<double>(1, 1) = m.at<double>(1, 1);
        result.at<double>(1, 2) = m.at<double>(1, 2);
        return result;
    };
    vtx_mat tmp_trans;

    switch( mode )
    {
    case HOMOGRAPHY:
    {
        tmp_trans = cv::findHomography( pts_1, pts_2, CV_RANSAC, RANSAC_INLIER_THRESHOLD, inlier);

        uint inlier_cnt = 0;
        for( uint i = 0; i < inlier.size(); i++ )
        {
            if( inlier[i] )
            {
                inlier_cnt++;
            }
        }
        if( (int)inlier_cnt <= std::min( MIN_INLIER, static_cast<int>(static_cast<float>(inlier.size()) * MIN_INLIER_RATIO) ) )
        {
            return BAD_TRANSFORM;
        }

        break;
    }
    case RIGID_TRANSFORM:
    {
        try {
            tmp_trans = cv::estimateRigidTransform( pts_1, pts_2, false);
        }
        catch (const std::exception &e)
        {
            std::cerr << "* EXCEPTION: vtx_estimate_transform(): " << e.what() << std::endl;
            return GENERIC_ERROR;
        }

        if( tmp_trans.type() != CV_64F )
        {
            tmp_trans.convertTo( tmp_trans, CV_64F );
        }

        break;
    }
    default:
    {
        std::cerr << "! ERROR: vtx_estimate_transform: invalid operation mode" << std::endl;
        return GENERIC_ERROR;
    }
    }

    // Update m_trans
    if ( mode == RIGID_TRANSFORM )
    {
//        std::cout << tmp_trans << std::endl;
        if( !tmp_trans.empty() )
        {
            convert( tmp_trans ).copyTo( trans );
        }
        else
        {
            return BAD_TRANSFORM;
        }

    }
    else
    {
        tmp_trans.copyTo( trans );
    }

    return SUCCESS;
}



//!
//! \brief appendVector
//! \param dst
//! \param src
//! \param shift
//! \return
//!
int vtx_appendVector( vtx_points_vector &dst, vtx_points_vector &src, cv::Point2f shift )
{
    for ( uint i = 0; i < src.size(); i++ )
    {
        dst.push_back( src[i] + shift );
    }

    return SUCCESS;
}

}
