
#include "stab_gcs_kiir.hpp"

namespace stab_gcs_kiir {

//!
//! \brief vtx_KIIRStabilizer::ageDelay
//!
void vtx_KIIRStabilizer::ageDelay()
{
    _curr_gray.copyTo( _prev_gray );
    _prev_pts.clear();
    _prev_pts = _curr_pts;
    _prev_pts_lk.clear();
    _prev_pts_lk = _curr_pts_lk;
}



//!
//! \brief vtx_KIIRStabilizer::setMotionEstimnator
//! \param key_type
//! \param motion_mode
//!
void vtx_KIIRStabilizer::setMotionEstimnator(const uint key_type, const uint motion_mode)
{
    _key_type    = key_type;
    _motion_mode = motion_mode;
}



//!
//! \brief vtx_KIIRStabilizer::run
//! \param input_rgb
//! \param stab_rgb
//! \return
//!
int vtx_KIIRStabilizer::run(const vtx_image &input_rgb, vtx_image &stab_rgb, float *_stab_data)
{
    ageDelay();

    //============= Extract key points from current frame
    vtx_mat tmp_img;
    cv::cvtColor( input_rgb, _curr_gray, cv::COLOR_RGB2GRAY );
    if( _curr_gray.cols >= 1280 )
    {
        cv::resize( _curr_gray, tmp_img, cv::Size( _curr_gray.cols >> 1, _curr_gray.rows >> 1) );
    }
    else if( _curr_gray.cols >= 1920 )
    {
        cv::resize( _curr_gray, tmp_img, cv::Size( _curr_gray.cols >> 2, _curr_gray.rows >> 2) );
    }
    else
    {
        _curr_gray.copyTo( tmp_img );
    }

    // Enhance image
    vtx_enhancement( tmp_img, _curr_gray );

    if( _prev_gray.empty() ) // first frame
    {
        input_rgb.copyTo( stab_rgb );
        return SUCCESS;
    }

    vtx_points_vector v_tmp_pts;
    vtx_image img_roi;
    cv::Rect  bound;
    uint w = _prev_gray.size().width,
         h = _prev_gray.size().height;

    double top, left, bot, right;
    uint pts_cnt;
    std::vector<uint> pts_idx;
    vtx_points_vector new_pts;
    int max_segment_kp = MAX_CORNERS / (VERTICA_BLKS * HORIZON_BLKS);
    int min_segment_kp = max_segment_kp * 3 / 5;
    for( uint r = 1; r < VERTICA_BLKS-1; r++ )
    {
        for( uint c = 1; c < HORIZON_BLKS-1; c++ )
        {
            //<DanDo> 16/1/18: Check the number of key point in current block
            top = (double)r * (double)h/VERTICA_BLKS;
            bot = top + (double)h/VERTICA_BLKS;
            left = (double)c * (double)w/HORIZON_BLKS;
            right = left + (double)w/HORIZON_BLKS;
            pts_idx.clear();
            pts_cnt = 0;
            for( uint i = 0; i < _prev_pts.size(); i++ )
            {
                if( (_prev_pts[i].x > left) && (_prev_pts[i].x < right) && (_prev_pts[i].y > top) && (_prev_pts[i].y < bot) )
                {
                    pts_cnt++;
                    pts_idx.push_back( i );
                }
            }

            if( pts_cnt > min_segment_kp)
            {
                for( uint i = 0; i < pts_cnt; i++ )
                {
                    new_pts.push_back( _prev_pts[pts_idx[i]] );
                }
            }
            else
            {
                // capture sub-image
                bound = cv::Rect( c * w/HORIZON_BLKS, r * h/VERTICA_BLKS, w/HORIZON_BLKS, h/VERTICA_BLKS );
                img_roi = _prev_gray( bound );

                // extract key point of that region
                if( vtx_extractKeypoints( img_roi, v_tmp_pts, _key_type, max_segment_kp ) != SUCCESS )
                {
                    std::cerr << "! ERROR: vtx_StabGCS::estimate_motion(): vtx_extractKeypoints failed" << std::endl;
                    return GENERIC_ERROR;
                }

                // append to the global key point storage
                if( vtx_appendVector( new_pts, v_tmp_pts,  cv::Point2f(c*w/HORIZON_BLKS, r*h/VERTICA_BLKS) ) != SUCCESS )
                {
                    break;
                }
            }
        }
    }
    _prev_pts.clear();
    _prev_pts = new_pts;
    new_pts.clear();
//    std::cout << "@ Key point length: " << _prev_pts.size() << std::endl;

    //============= Estimate motion
    if( !_prev_pts.empty() )
    {
        vtx_points_vector curr_pts_list;
        std::vector<uchar> match_status;
        std::vector<float> match_error;
        try
        {
            cv::calcOpticalFlowPyrLK( _prev_gray, _curr_gray, _prev_pts, curr_pts_list, match_status, match_error,
                                      cv::Size( 21, 21 ), 5,
                                      cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03),
                                      0, 0.001);
        }
        catch( std::exception &e )
        {
            std::cerr << "* Exception: vtx_KIIRStabilizer::run(): calcOpticalFlowPyrLK(): " << e.what() << std::endl;
            return GENERIC_ERROR;
        }

        _prev_pts_lk.clear();
        _curr_pts_lk.clear();
        for ( uint i = 0; i < _prev_pts.size(); i++ )
        {
            if ( match_status[i] )
            {
                _prev_pts_lk.push_back( _prev_pts[i] );
                _curr_pts_lk.push_back( curr_pts_list[i] );
            }
        }

        // only estimate transform if there are enough correspondent pairs
//        std::cout << "------ matched ratio: " << static_cast<double>(_curr_pts_lk.size()) / static_cast<double>(_prev_pts.size()) << '\t';
        if( _curr_pts_lk.size() >= MIN_CORNERS )
        {
            vtx_mat T_last;
            int ret = vtx_estimate_transform( _motion_mode, _curr_pts_lk, _prev_pts_lk, T_last );

            if( ret == SUCCESS )
            {
                T_last.copyTo( _T );
            }
            else if( ret == GENERIC_ERROR )
            {
                std::cerr << "! ERROR: vtx_KIIRStabilizer::run(): vtx_estimate_transform failed" << std::endl;
                return GENERIC_ERROR;
            }
            else
            {
                _T = 0.5 * _T + 0.5 * cv::Mat::eye( cv::Size(3, 3), _T.type() );
            }
        }
        else
        {
            _T = 0.5 * _T + 0.5 * cv::Mat::eye( cv::Size(3, 3), _T.type() );
            std::cout << "====> to few inliers";
        }
//        std::cout << std::endl;
    }

    //<DanDo> 16/1/18:  Consider _curr_pts_lk as the set of key point of current frame
    //                  Filter bad homographies
    _curr_pts.clear();
    _curr_pts = _curr_pts_lk;
    vtx_homographyFilter( _T, _prev_gray.size() );


    //=============== Filtering
//    std::cout << std::endl << _T.at<double>(0, 2) << '\t' << _T.at<double>(1, 2) << std::endl;

    double dx = _T.at<double>(0, 2),
           dy = _T.at<double>(1, 2);

    // Kalman filter
    vtx_mat last_T = _T.inv();
    double da;
    vtx_mat P;

    vtx_homoDecomposition_v2( last_T, &dx, &dy, &da, P );

    _x += dx;
    _y += dy;
    _a += da;
    c_z = Trajectory( _x, _y, _a );

    //time update£¨prediction£©
    c_X_ = c_X;         //X_(k) = X(k-1);
    c_P_ = c_P + c_Q;   //P_(k) = P(k-1)+Q;
    // measurement update£¨correction£©
    c_K = c_P_/( c_P_ + c_R );              //gain;K(k) = P_(k)/( P_(k)+R );
    c_X = c_X_ + c_K * (c_z - c_X_);        //z-X_ is residual,X(k) = X_(k)+K(k)*(z(k)-X_(k));
    c_P = (Trajectory(1,1,1) - c_K) * c_P_; //P(k) = (1-K(k))*P_(k);

    // current-taget difference
    double  diff_x = c_X.x - _x,
            diff_y = c_X.y - _y,
            diff_a = c_X.a - _a;

    // derive kalman filtered image
    vtx_mat kalman_T;
    vtx_homoComposition_v2( kalman_T, diff_x, diff_y, diff_a, P );


    //================ IIR filter
    // update new accumulate transformation
    _acc_T = _acc_T * _T * kalman_T.inv();

    // iir smoothing
    vtx_iirSmoothHomo( _acc_T );
    _acc_T = _acc_T * kalman_T;
    vtx_overlapConstrain( _acc_T );

    //================= Warping
    vtx_mat stab_trans = _acc_T.clone();
    if( input_rgb.cols >= 1280 )
    {
        stab_trans.at<double>(0, 2) *= 2.0f;
        stab_trans.at<double>(1, 2) *= 2.0f;
        stab_trans.at<double>(2, 0) *= 0.5f;
        stab_trans.at<double>(2, 1) *= 0.5f;
    }
    else if( input_rgb.cols >= 1920 )
    {
        stab_trans.at<double>(0, 2) *= 4.0f;
        stab_trans.at<double>(1, 2) *= 4.0f;
        stab_trans.at<double>(2, 0) *= 0.25f;
        stab_trans.at<double>(2, 1) *= 0.25f;
    }

    cv::Mat affine_T(2, 3, CV_64FC1);
    affine_T.at<double>(0, 0) = stab_trans.at<double>(0, 0);
    affine_T.at<double>(0, 1) = stab_trans.at<double>(0, 1);
    affine_T.at<double>(0, 2) = stab_trans.at<double>(0, 2);
    affine_T.at<double>(1, 0) = stab_trans.at<double>(1, 0);
    affine_T.at<double>(1, 1) = stab_trans.at<double>(1, 1);
    affine_T.at<double>(1, 2) = stab_trans.at<double>(1, 2);
//    cv::warpPerspective( input_rgb, stab_rgb, stab_trans, input_rgb.size(), cv::INTER_LINEAR );
    cv::warpAffine( input_rgb, stab_rgb, affine_T, input_rgb.size(), cv::INTER_LINEAR );

//    warp_img.copyTo( stab_rgb );

    return SUCCESS;
}

//!
//! \brief extractKeypoints
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
//        try
//        {
//            cv::cornerSubPix( image, pts,  cv::Size(10, 10), cv::Size(-1, -1),
//                              cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03));
//        }
//        catch( std::exception &e )
//        {
//            std::cerr << "! ERROR: vtx_extractKeypoints(): cv::cornerSubPix() failed" << std::endl;
//        }

        break;
    }
    case FAST_CORNER:
    {
        // apply fast corner
        std::vector<cv::KeyPoint> tmp_pts;
        cv::FAST( image, tmp_pts, FASTCORNERS_MIN_DIS, true, cv::FastFeatureDetector::TYPE_7_12 );

        // sub sampling the key point vector
        if ( tmp_pts.size() < max_pts_num )
        {
            pts.clear();
            uint len = tmp_pts.size();
            for ( uint i = 0; i < len; i++ )
            {
                pts.push_back( tmp_pts[i].pt);
            }
        }
        else
        {
            // sorting key point scores
            std::vector<float> responses;
            for ( uint i = 0; i < tmp_pts.size(); i++ )
            {
                responses.push_back( tmp_pts[i].response);
            }

            std::vector<size_t> indices = sort_indexes( responses );

            // sub sampling the highest score key point
            pts.clear();
            for ( uint i = 0; i < max_pts_num; i++ )
            {
                pts.push_back( tmp_pts[indices[i]].pt);
            }
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

    std::vector<char> inlier;   // store the inlier status when using robus estimation method (RANSAC)

    auto convert = [&](vtx_mat &m)
    {
        vtx_mat result = vtx_mat::eye(3, 3, CV_64F);
        result.at<double>(0, 0) = m.at<float>(0, 0);
        result.at<double>(0, 1) = m.at<float>(0, 1);
        result.at<double>(0, 2) = m.at<float>(0, 2);
        result.at<double>(1, 0) = m.at<float>(1, 0);
        result.at<double>(1, 1) = m.at<float>(1, 1);
        result.at<double>(1, 2) = m.at<float>(1, 2);
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
        if( inlier_cnt <= std::min( MIN_INLIER, static_cast<int>(static_cast<float>(inlier.size()) * MIN_INLIER_RATIO) ) )
        {
            return BAD_TRANSFORM;
        }

        break;
    }
    case RIGID_TRANSFORM:
    {
//        try {
//            tmp_trans = cv::estimateRigidTransform( pts_1, pts_2, false);
//        }
//        catch (const std::exception &e)
//        {
//            std::cerr << "* EXCEPTION: vtx_estimate_transform(): " << e.what() << std::endl;
//            return GENERIC_ERROR;
//        }

        // <DanDo> 2/2/2017: Apply RANSAC for RIGID transform estimation
        cv::videostab::RansacParams ransacParams = cv::videostab::RansacParams::default2dMotion(cv::videostab::MM_SIMILARITY);
        ransacParams.thresh = 2.0f;
        ransacParams.eps = 0.5f;
        ransacParams.prob = 0.99f;
        ransacParams.size = 4;

        int inliers_cnt = 0;
        try
        {
            tmp_trans = cv::videostab::estimateGlobalMotionRansac( pts_1, pts_2, cv::videostab::MM_SIMILARITY, ransacParams, nullptr, &inliers_cnt);
        }
        catch (std::exception &e)
        {
            std::cerr << "Exception: vtx_estimate_transform: "  << e.what() << std::endl;
            return GENERIC_ERROR;
        }
//        std::cout << "\n-----oh-------\n" << tmp_trans << "\n----yeah-------\n" ;

        if( inliers_cnt <= std::min( MIN_INLIER, static_cast<int>(static_cast<float>(pts_1.size()) * MIN_INLIER_RATIO) ) )
        {
            return BAD_TRANSFORM;
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
//        std::cout << "\n-----ojksdkhh-------\n" << tmp_trans << "\n----yjsfhgkjfdheah-------\n" ;
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
//! \brief vtx_appendVector
//! \param dst
//! \param src
//! \param shift
//! \return
//!
int vtx_appendVector( vtx_points_vector &dst, vtx_points_vector &src, cv::Point2f shift )
{
//    if ( dst.capacity() < ( dst.size() + src.size() ) )
//    {
//        std::cerr << "! ERROR: appendVector: not enough space" << std::endl;
//        return GENERIC_ERROR;
//    }

    for ( uint i = 0; i < src.size(); i++ )
    {
        dst.push_back( src[i] + shift );
    }

    return SUCCESS;
}


//!
//! \brief vtx_iirSmoothHomo
//! \param src
//!
void vtx_iirSmoothHomo( vtx_mat &homo_mat )
{
    assert( (homo_mat.cols == 3) && (homo_mat.rows == 3) && (homo_mat.type() == CV_64F) );

    //=============== Extract displacement, rotation, scale and perspective components of the homography matrix
    // displacement
    double dx = homo_mat.at<double>(0, 2),
           dy = homo_mat.at<double>(1, 2);

    // perspective
    vtx_mat perspective = vtx_mat::eye( 3, 3, CV_64F );
    perspective.at<double>(2, 0) = homo_mat.at<double>(2, 0);
    perspective.at<double>(2, 1) = homo_mat.at<double>(2, 1);

    // scaling and rotation (similarity component)
    vtx_mat similarity = vtx_mat::zeros( 2, 2, CV_64F );
    similarity.at<double>(0, 0) = homo_mat.at<double>(0, 0) - dx * homo_mat.at<double>(2, 0);
    similarity.at<double>(0, 1) = homo_mat.at<double>(0, 1) - dx * homo_mat.at<double>(2, 1);
    similarity.at<double>(1, 0) = homo_mat.at<double>(1, 0) - dy * homo_mat.at<double>(2, 0);
    similarity.at<double>(1, 1) = homo_mat.at<double>(1, 1) - dy * homo_mat.at<double>(2, 1);

    cv::SVD sim_svd( similarity, cv::SVD::FULL_UV );
    vtx_mat rotation = sim_svd.u * sim_svd.vt;

    //============== IIR filtering
    // displacement
    double  x_ratio = fabs( dx ) / IIR_MAX_DX,
            y_ratio = fabs( dy ) / IIR_MAX_DY;
    double  x_correct = IIR_CORRECTION_D,
            y_correct = IIR_CORRECTION_D;
    if( x_ratio < 1 )
    {
        x_correct = 1 - (1 - IIR_CORRECTION_D) * exp( IIR_POWER_FACTOR * log(x_ratio));
    }

    if( y_ratio < 1 )
    {
        y_correct = 1 - (1 - IIR_CORRECTION_D) * exp( IIR_POWER_FACTOR * log(y_ratio));
    }
    x_correct += IIR_CORRECTION_BIAS;
    y_correct += IIR_CORRECTION_BIAS;

    dx *= x_correct;
    dy *= y_correct;

    // scaling ( remove shearing by averaging eigen values)
    vtx_mat scale = vtx_mat::eye( 2, 2, CV_64F );
    scale.at<double>(0, 0) = (sim_svd.w).at<double>(0, 0);
    scale.at<double>(1, 1) = (sim_svd.w).at<double>(1, 0);

//    std::cout << "------------------------------\n";
//    std::cout << std::endl << rotation << std::endl;
//    std::cout << std::endl << sim_svd.w << std::endl;

    // rotation
    double  angle = atan( rotation.at<double>(1, 0) / rotation.at<double>(0, 0) );
//    std::cout << "angle:  " << angle * 180 / 3.14159f << std::endl;

    double  angle_ratio = fabs( angle ) / IIR_MAX_DA;

    double  angle_correct = IIR_CORRECTION_A;
    if( angle_ratio < 1 )
    {
        angle_correct = 1 - (1 - IIR_CORRECTION_A) * exp( IIR_POWER_FACTOR * log(angle_ratio));
    }
    angle_correct += IIR_CORRECTION_BIAS;

    angle *= angle_correct;
    rotation.at<double>(0, 0) = cos( angle );
    rotation.at<double>(1, 0) = sin( angle );
    rotation.at<double>(0, 1) = -rotation.at<double>(1, 0);
    rotation.at<double>(1, 1) = rotation.at<double>(0, 0);


    vtx_mat I = vtx_mat::eye( 2, 2, CV_64F );
    double sim_correct = (x_correct + y_correct + angle_correct) / 3.0f;
    cv::addWeighted( scale, 0.95, I, 1 - 0.95, 0, scale );
    perspective.at<double>(2, 0) *= sim_correct;
    perspective.at<double>(2, 1) *= sim_correct;

    //================== Compose filtered mstrix into new hmography matrix
    similarity = rotation * (sim_svd.vt).t() * scale * sim_svd.vt;

    vtx_mat filtered_affine = vtx_mat::eye( 3, 3, CV_64F );
    filtered_affine.at<double>(0, 0) = similarity.at<double>(0, 0);
    filtered_affine.at<double>(0, 1) = similarity.at<double>(0, 1);
    filtered_affine.at<double>(1, 0) = similarity.at<double>(1, 0);
    filtered_affine.at<double>(1, 1) = similarity.at<double>(1, 1);
    filtered_affine.at<double>(0, 2) = dx;
    filtered_affine.at<double>(1, 2) = dy;

    homo_mat = filtered_affine * perspective;
}



//!
//! \brief vtx_homoDecomposition
//! \param homo_mat
//! \param dx
//! \param dy
//! \param da
//!
void vtx_homoDecomposition( vtx_mat &homo_mat, double *dx, double *dy, double *da, vtx_mat &P)
{
    assert( homo_mat.rows == 3 && homo_mat.cols ==3 && homo_mat.type() == CV_64F );
    // displacement
    *dx = homo_mat.at<double>(0, 2);
    *dy = homo_mat.at<double>(1, 2);

    // scaling and rotation (similarity component)
    vtx_mat similarity = vtx_mat::zeros( 2, 2, CV_64F );
    similarity.at<double>(0, 0) = homo_mat.at<double>(0, 0) - (*dx) * homo_mat.at<double>(2, 0);
    similarity.at<double>(0, 1) = homo_mat.at<double>(0, 1) - (*dx) * homo_mat.at<double>(2, 1);
    similarity.at<double>(1, 0) = homo_mat.at<double>(1, 0) - (*dy) * homo_mat.at<double>(2, 0);
    similarity.at<double>(1, 1) = homo_mat.at<double>(1, 1) - (*dy) * homo_mat.at<double>(2, 1);

    cv::SVD sim_svd( similarity, cv::SVD::FULL_UV );

    vtx_mat rotation = sim_svd.u * sim_svd.vt;
    *da = atan( rotation.at<double>(1, 0) / rotation.at<double>(0, 0) );

    vtx_mat scale = vtx_mat::eye( 2, 2, CV_64F );
    scale.at<double>(0, 0) = (sim_svd.w).at<double>(0, 0);
    scale.at<double>(1, 1) = (sim_svd.w).at<double>(1, 0);
    P = (sim_svd.vt).t() * scale * sim_svd.vt;
}


//!
//! \brief vtx_homoDecomposition_v2
//! \param homo_mat
//! \param dx
//! \param dy
//! \param da
//! \param P
//!
void vtx_homoDecomposition_v2( vtx_mat &homo_mat, double *dx, double *dy, double *da, vtx_mat &P)
{
    assert( homo_mat.rows == 3 && homo_mat.cols ==3 && homo_mat.type() == CV_64F );
    // displacement
    *dx = homo_mat.at<double>(0, 2);
    *dy = homo_mat.at<double>(1, 2);

    *da = atan( homo_mat.at<double>(1, 0) / homo_mat.at<double>(0, 0) );

    P = vtx_mat::eye( cv::Size(2, 2), CV_64F );
}



//!
//! \brief vtx_homoComposition
//! \param homo_mat
//! \param dx
//! \param dy
//! \param da
//! \param P
//!
void vtx_homoComposition( vtx_mat &homo_mat, const double dx, const double dy, const double da, const vtx_mat &P)
{
    homo_mat = vtx_mat::eye( 3, 3, CV_64F );

    // displacement
    homo_mat.at<double>(0, 2) = dx;
    homo_mat.at<double>(1, 2) = dy;

    // scaling and rotation (similarity component)
    vtx_mat similarity = vtx_mat::zeros( 2, 2, CV_64F );
    similarity.at<double>(0, 0) = cos( da );
    similarity.at<double>(0, 1) = -sin( da );
    similarity.at<double>(1, 0) = sin( da );
    similarity.at<double>(1, 1) = cos( da );

    similarity = similarity * P;

    homo_mat.at<double>(0, 0) = similarity.at<double>(0, 0);
    homo_mat.at<double>(0, 1) = similarity.at<double>(0, 1);
    homo_mat.at<double>(1, 0) = similarity.at<double>(1, 0);
    homo_mat.at<double>(1, 1) = similarity.at<double>(1, 1);
}



//!
//! \brief vtx_homoComposition_v2
//! \param homo_mat
//! \param dx
//! \param dy
//! \param da
//! \param P
//!
void vtx_homoComposition_v2( vtx_mat &homo_mat, const double dx, const double dy, const double da, const vtx_mat &P)
{
    homo_mat = vtx_mat::eye( 3, 3, CV_64F );

    // displacement
    homo_mat.at<double>(0, 2) = dx;
    homo_mat.at<double>(1, 2) = dy;

    homo_mat.at<double>(0, 0) = cos( da );
    homo_mat.at<double>(0, 1) = -sin( da );
    homo_mat.at<double>(1, 0) = sin( da );
    homo_mat.at<double>(1, 1) = cos( da );
}



//!
//! \brief vtx_overlapConstrain
//!     Apply the constrain on the area of the overlaping region between two adjacent stab frames
//!
//! \param homo_mat
//!
void vtx_overlapConstrain( vtx_mat &homo_mat )
{
    assert( homo_mat.cols == 3 && homo_mat.rows == 3 );
    assert( homo_mat.type() == CV_64F );

    // Compute overlap area
    vtx_mat mask = vtx_mat::ones( 72, 128, CV_8U );
    vtx_mat scale_mat = homo_mat.clone();
    scale_mat.at<double>(0, 2) = homo_mat.at<double>(0, 2) / 10;
    scale_mat.at<double>(1, 2) = homo_mat.at<double>(1, 2) / 10;

    cv::warpPerspective( mask, mask, scale_mat, mask.size(), cv::INTER_LINEAR );
    double overlap_portion = (double)cv::sum( mask )[0] / (128.0f * 72.0f);

//    std::cout << "overlap_portion: " << overlap_portion << std::endl;

    // Compute correction factor
    double correction = 1.0f;
    if( overlap_portion <= 1 && overlap_portion > 0.9f )
    {
        correction = 1.0f;
    }
    else
    {
        correction = overlap_portion / 0.9f;
    }
//    std::cout << "correction: " << correction << std::endl;

    cv::setIdentity( scale_mat );
    cv::addWeighted( homo_mat, correction, scale_mat, 1-correction, 0, homo_mat );
}



//!
//! \brief vtx_homographyFilter
//! \param homo_mat
//! \param frame_size
//!
void vtx_homographyFilter( vtx_mat &homo_mat, const cv::Size frame_size )
{
    //============= Compute new position of vertices
    double  width = frame_size.width,
            height = frame_size.height;
    cv::Mat vertex = cv::Mat::zeros( 3, 4, CV_64F );
    vertex.at<double>(0, 1) = width;
    vertex.at<double>(0, 2) = width;
    vertex.at<double>(1, 2) = height;
    vertex.at<double>(1, 3) = height;
    for( uint i = 0; i < 4; i++ )
    {
        vertex.at<double>(2, i) = 1.0f;
    }

    cv::Mat new_vertex = homo_mat * vertex;
    for( uint i = 0; i < 4; i++ )
    {
        if( new_vertex.at<double>(2, i) )
        {
            new_vertex.at<double>(0, i) /= new_vertex.at<double>(2, i);
            new_vertex.at<double>(1, i) /= new_vertex.at<double>(2, i);
            new_vertex.at<double>(2, i)  = 1.0f;
        }
        else
        {
            std::cout << "------------> polar point" << std::endl;
            cv::setIdentity( homo_mat );
            return;
        }
    }

    //============== Compute lengths of diags and edges
    double dx, dy, diag1, diag2, vedge1, vedge2, hedge1, hedge2;

    // Diags
    dx = new_vertex.at<double>(0, 0) - new_vertex.at<double>(0, 2);
    dy = new_vertex.at<double>(1, 0) - new_vertex.at<double>(1, 2);
    diag1 = sqrt( dx * dx + dy * dy );

    dx = new_vertex.at<double>(0, 1) - new_vertex.at<double>(0, 3);
    dy = new_vertex.at<double>(1, 1) - new_vertex.at<double>(1, 3);
    diag2 = sqrt( dx * dx + dy * dy );

    double diag_ratio = (diag1 > diag2)? (diag1 / diag2) : (diag2 / diag1);
    if( diag_ratio > 1.1 )
    {
        std::cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  diag_ratio: " << diag_ratio << std::endl;
        cv::setIdentity( homo_mat );
//        homo_mat = 0.3 * homo_mat + 0.7 * cv::Mat::eye( 3, 3, CV_64F );
        return;
    }

    // Vertical Edges
    dx = new_vertex.at<double>(0, 0) - new_vertex.at<double>(0, 3);
    dy = new_vertex.at<double>(1, 0) - new_vertex.at<double>(1, 3);
    vedge1 = sqrt( dx * dx + dy * dy );

    dx = new_vertex.at<double>(0, 1) - new_vertex.at<double>(0, 2);
    dy = new_vertex.at<double>(1, 1) - new_vertex.at<double>(1, 2);
    vedge2 = sqrt( dx * dx + dy * dy );

    double vedge_ratio = (vedge1 > vedge2)? (vedge1 / vedge2) : (vedge2 / vedge1);
    if( vedge_ratio > 1.1 )
    {
        std::cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  vedge_ratio: " << vedge_ratio << std::endl;
        cv::setIdentity( homo_mat );
//        homo_mat = 0.3 * homo_mat + 0.7 * cv::Mat::eye( 3, 3, CV_64F );
        return;
    }

    // Horizontal Edges
    dx = new_vertex.at<double>(0, 0) - new_vertex.at<double>(0, 1);
    dy = new_vertex.at<double>(1, 0) - new_vertex.at<double>(1, 1);
    hedge1 = sqrt( dx * dx + dy * dy );

    dx = new_vertex.at<double>(0, 2) - new_vertex.at<double>(0, 3);
    dy = new_vertex.at<double>(1, 2) - new_vertex.at<double>(1, 3);
    hedge2 = sqrt( dx * dx + dy * dy );

    double hedge_ratio = (hedge1 > hedge2)? (hedge1 / hedge2) : (hedge2 / hedge1);
    if( hedge_ratio > 1.1 )
    {
        std::cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  hedge_ratio: " << hedge_ratio << std::endl;
        cv::setIdentity( homo_mat );
//        homo_mat = 0.3 * homo_mat + 0.7 * cv::Mat::eye( 3, 3, CV_64F );
        return;
    }

}





//!
//! \brief enhancement
//! \param input
//! \param output
//!
void vtx_enhancement( const vtx_mat &input, vtx_mat &output )
{
    assert( input.channels() == 1 );
//    std::cout << input.size() << std::endl;

    //========= Sharpening
    cv::Mat blur, sharpened;
    try{
        cv::GaussianBlur( input, blur, cv::Size(0, 0), SIGMA_X );
        cv::addWeighted( input, 1.5f, blur, -0.5f, 0, sharpened );
    }
    catch( std::exception &e )
    {
        input.copyTo( output );
        return;
    }

    //========= Contrast enhancement
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE( 40.0, cv::Size( 8, 8 ) );

    try
    {
        clahe->apply( sharpened, output );
    }
    catch( std::exception &e )
    {
        std::cout << "@ Exception: " << e.what() << std::endl;
        input.copyTo( output );
        return;
    }

//    sharpened.copyTo( output );

    //========= Display result
#ifdef DEBUG
    uint w0 = input.cols;
    uint h0 = input.rows;
    cv::Mat canvas = cv::Mat::zeros(h0, 2*w0, input.type());
    input.copyTo(canvas(cv::Range::all(), cv::Range(0, w0)));
    sharpened.copyTo(canvas(cv::Range::all(), cv::Range(w0, 2*w0)));
    if( canvas.cols >= 1920 )
    {
        cv::resize( canvas, canvas, cv::Size(w0, h0 >> 1) );
    }

    cv::imshow( "Debug", canvas );
    cv::waitKey( 1 );
#endif
}



//!
//! \brief sort_indexes
//! \param v
//! \return
//!
template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v) {

  // initialize original index locations
  std::vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}


//!
//! \brief random_sampling
//! \param range
//! \param rand_num
//! \return
//!
std::vector<uint> random_sampling( const uint range, const uint rand_num )
{
    assert( rand_num <= range );

    // Create a sequence of values from which random numbers are drawn
    uint *values = new uint[range];
    for( uint i = 0; i < range; i++ )
    {
        values[i] = i;
    }

    // Durstenfeld random number generation
    std::vector<uint> rand_values;
    uint tmp, idx;
    for( uint i = 1; i <= rand_num; i++ )
    {
        // Create random index in the remaining array
        idx = rand() % (range - i);

        // Push element at index [idx] to output vector
        rand_values.push_back( values[idx] );

        // Swap element at index [idx] with the last element of the remaining array
        tmp = values[idx];
        values[idx] = values[range - i];
        values[range-i] = tmp;
    }

    delete values;
    return rand_values;
}


//!
//! \brief certify_model
//! \param model
//! \param pt1
//! \param pt2
//! \param status
//!
void vtx_certify_model( vtx_mat &model, std::vector<cv::Point2f> &pts1, std::vector<cv::Point2f> &pts2,
                    const uint d, const double threshold, uint &inliers, bool &status )
{
    assert( pts1.size() == pts2.size() );

    std::vector<cv::Point2f> inliers_1, inliers_2;
    uint points_num = pts1.size();

    vtx_mat homogenous_point = vtx_mat::zeros( 3, 1, CV_64F ), new_point;
    double dx, dy, r;
    inliers = 0;
    for( uint i = 0; i < points_num; i++ )
    {
        //==== working in homogenous coordinate system
        homogenous_point.at<double>(0, 0) = static_cast<double>( pts1[i].x );
        homogenous_point.at<double>(1, 0) = static_cast<double>( pts1[i].y );
        homogenous_point.at<double>(2, 0) = 1.0f;

        //==== transforming
        new_point = model * homogenous_point;
        if( new_point.at<double>(2, 0) == 0 )
        {
            continue;
        }
        new_point = new_point / new_point.at<double>(2, 0);

        //==== calculating error
        dx = new_point.at<double>(0, 0) - static_cast<double>( pts2[i].x );
        dy = new_point.at<double>(1, 0) - static_cast<double>( pts2[i].y );
        r = sqrt( dx * dx + dy * dy );
        if( r < threshold )
        {
            inliers_1.push_back( cv::Point2f( static_cast<float>(new_point.at<double>(0, 0)), static_cast<float>(new_point.at<double>(1, 0)) ) );
            inliers_2.push_back( pts2[i] );
            inliers++;
        }
    }

    //===== Certify if the model is good or not
    if( inliers < d )
    {
        status = false;
        return;
    }

    vtx_estimate_transform( RIGID_TRANSFORM, inliers_1, inliers_2, model );
    status = true;
}



//!
//! \brief vtx_RANSACRigid
//! \param pts1
//! \param pts2
//! \param transform
//! \param threshold
//! \param inliers_ratio
//! \param p
//!
void vtx_RANSACRigid( std::vector<cv::Point2f> &pts1, std::vector<cv::Point2f> &pts2, vtx_mat &transform, const double threshold,
                      const double inliers_ratio, const double p )
{
    //=========== Compute sufficient iterations and inliers for good model
    uint pts_num = pts1.size();
    uint d = static_cast<uint>( pts_num * inliers_ratio * 0.7 );
    uint k = static_cast<uint>( log( 1 - p ) / log( 1 - exp( 4 * log(inliers_ratio ) ) ) );
    if( k < 200 )
    {
        k = 200;
    }
    transform = vtx_mat::eye( 3, 3, CV_64F );


    //=========== RANSAC
    std::vector<uint> random_idx;
    vtx_points_vector random_pts1, random_pts2;
    uint i, j, max_inlier = 0, inliers;
    bool status;
    vtx_mat tmp_model;
    for( i = 0; i < k; i ++ )
    {
        // Randomly choosing key point pairs for estimating model
        random_idx = random_sampling( pts_num, 4 );
        random_pts1.clear();
        random_pts2.clear();
        for( j = 0; j < 4; j++ )
        {
            random_pts1.push_back( pts1[random_idx[j]] );
            random_pts2.push_back( pts2[random_idx[j]] );
        }

        // Estimate model that fit randomly chosen point pairs
        vtx_estimate_transform( RIGID_TRANSFORM, random_pts1, random_pts2, tmp_model );

        // Certify model
        vtx_certify_model( tmp_model, pts1, pts2, d, threshold, inliers, status );
        std::cout << pts1.size() << '\t' << inliers << std::endl;
        if( status )
        {
            if( inliers > max_inlier )
            {
                tmp_model.copyTo( transform );
                max_inlier = inliers;
            }
        }
    }
}


//!
//! \brief vtx_isIdentity
//! \param transform
//! \return
//!
bool vtx_isIdentity( const vtx_mat &transform )
{
    if ( !transform.at<double>(0, 1) && !transform.at<double>(1, 0) && !transform.at<double>(0, 2) && !transform.at<double>(1, 2) )
    {
        if( (transform.at<double>(0, 0) == 1) && (transform.at<double>(1, 1) == 1) && (transform.at<double>(2, 2) == 1) )
        {
            return true;
        }
    }

    return false;
}



}

