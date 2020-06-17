#include <float.h>
#include <math.h>

#include <opencv2/videostab.hpp>

#include "lme.hpp"

template <typename T>
void printVec(std::vector<T> vec)
{
    for (uint i = 0; i < vec.size(); i++)
    {
        std::cout << vec[i] << "  ";
    }

    std::cout << std::endl;
}

static void medianShift(std::vector<cv::Point2f>src, std::vector<cv::Point2f> dst, float* dx, float* dy);



void LMEstimator::ageDelay()
{
    m_currGray.copyTo(m_prevGray);
    m_prevPts.clear();
    m_prevPts = m_currPts;
}



//!
//! \brief LMEstimator::run
//! \param img
//! \param roi
//! \param dx
//! \param dy
//! \return
//!
int LMEstimator::run(const cv::Mat& img, const cv::Rect& roi, float* dx, float* dy)
{
    ageDelay();

    if (img.channels() > 1)
    {
        cv::cvtColor(img, m_currGray, cv::COLOR_RGB2GRAY);
    }
    else
    {
        img.copyTo(m_currGray);
    }

    //===== 1. Corrner Detection
    m_currPts.clear();
    cv::Rect rect = roi;

    if (rect.x < 0)
    {
        rect.x = 0;
    }

    if ((rect.x + rect.width) > img.cols)
    {
        rect.width = img.cols - rect.x;
    }

    if (rect.y < 0)
    {
        rect.y = 0;
    }

    if ((rect.y + rect.height) > img.rows)
    {
        rect.height = img.rows - rect.y;
    }

    cv::Mat patch = m_currGray(rect);
    int maxCorners = (MAX_CORNERS * rect.width * rect.height) / (img.cols * img.rows);
    maxCorners = (maxCorners > 5) ? maxCorners : 5;
    //    printf( "%s: maxCorner = %d\n", __FUNCTION__, maxCorners );
    cv::goodFeaturesToTrack(patch, m_currPts, 50, GOODFEATURE_QUALITY, GOODFEATURE_MIN_DIS);

    try
    {
        cv::cornerSubPix(patch, m_currPts, cv::Size(5, 5), cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03));
    }
    catch (std::exception& e)
    {
        std::cerr << "! ERROR: vtx_extractKeypoints(): cv::cornerSubPix() failed" << std::endl;
    }

    //    printf( "Keypoint numbers: %d\n", m_currPts.size() );
    //    cv::Mat disp = img.clone();
    for (uint i = 0; i < m_currPts.size(); i++)
    {
        m_currPts[i].x += (float)rect.x;
        m_currPts[i].y += (float)rect.y;
        //        cv::circle( disp, cv::Point((int)m_currPts[i].x, (int)m_currPts[i].y), 2, cv::Scalar(0, 255, 255), 2, 8 );
    }

    //    cv::imshow( "window", disp );
    //    cv::waitKey( 30 );

    //===== 2. Opticalflow
    if (m_prevGray.empty())
    {
        *dx = 0.0;
        *dy = 0.0;
        m_trans = cv::Mat::eye(3, 3, CV_64F);
        return SUCCESS;
    }

    if (!m_currPts.empty())
    {
        std::vector<cv::Point2f> prevPts_lk, currPts_lk;
        std::vector<uchar> match_status;
        std::vector<float> match_error;

        try
        {
            cv::calcOpticalFlowPyrLK(m_currGray, m_prevGray, m_currPts, m_prevPts, match_status, match_error,
                                     cv::Size(21, 21), 5,
                                     cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03),
                                     0, 0.001);
        }
        catch (std::exception& e)
        {
            *dx = *dy = 0;
            m_trans = cv::Mat::eye(3, 3, CV_64F);
            return BAD_TRANSFORM;
        }

        prevPts_lk.clear();
        currPts_lk.clear();

        for (uint i = 0; i < m_currPts.size(); i++)
        {
            if (match_status[i])
            {
                prevPts_lk.push_back(m_prevPts[i]);
                currPts_lk.push_back(m_currPts[i]);
            }
        }

        if (currPts_lk.size() > 5)
        {
            medianShift(prevPts_lk, currPts_lk, dx, dy);

            if (currPts_lk.size() > 10)
            {
                cv::Mat tmp;
                cv::videostab::RansacParams ransacParams;
                int inliers = 0;
                // === hainh
                ransacParams = cv::videostab::RansacParams::default2dMotion(cv::videostab::MM_SIMILARITY);
                ransacParams.thresh = 2.0f;
                ransacParams.eps = 0.5f;
                ransacParams.prob = 0.99f;
                ransacParams.size = 4;
                //                cv::Mat tmp = cv::videostab::estimateGlobalMotionRobust( prevPts_lk, currPts_lk,
                //                                                                         cv::videostab::LINEAR_SIMILARITY, ransacParams,
                //                                                                         nullptr, &inliers );
                tmp = cv::videostab::estimateGlobalMotionRansac(prevPts_lk, currPts_lk,
                                                                cv::videostab::MM_SIMILARITY, ransacParams,
                                                                nullptr, &inliers);

                // hainh ===
                if (((float)inliers / (float)currPts_lk.size()) < 0.2f)
                {
                    printf("---> bad gme: to few inliers\n");
                    m_trans = cv::Mat::eye(3, 3, CV_64F);
                }
                else
                {
                    tmp.convertTo(m_trans, CV_64FC1);
                }
            }
            else
            {
                cv::Mat tmp =  cv::estimateRigidTransform(prevPts_lk, currPts_lk, false);

                if (tmp.empty())
                {
                    //                    printf( "-----> bad lme: empty trans\n" );
                    m_trans = cv::Mat::eye(3, 3, CV_64F);
                    return BAD_TRANSFORM;
                }
                else
                {
                    assert(tmp.type() == CV_64FC1);
                    m_trans.at<double>(0, 0) = tmp.at<double>(0, 0);
                    m_trans.at<double>(0, 1) = tmp.at<double>(0, 1);
                    m_trans.at<double>(0, 2) = tmp.at<double>(0, 2);
                    m_trans.at<double>(1, 0) = tmp.at<double>(1, 0);
                    m_trans.at<double>(1, 1) = tmp.at<double>(1, 1);
                    m_trans.at<double>(1, 2) = tmp.at<double>(1, 2);
                    m_trans.at<double>(2, 0) = 0.0f;
                    m_trans.at<double>(2, 1) = 0.0f;
                    m_trans.at<double>(2, 2) = 1.0f;
                }
            }
        }
        else
        {
            *dx = *dy = 0.0f;
            m_trans = cv::Mat::eye(3, 3, CV_64F);
            return BAD_TRANSFORM;
        }
    }
    else
    {
        //        printf( "-----> bad lme: empty currPts_lk\n" );
        *dx = *dy = 0.0f;
        m_trans = cv::Mat::eye(3, 3, CV_64F);
        return BAD_TRANSFORM;
    }

    return SUCCESS;
}


//!
//! \brief LMEstimator::getTrans
//! \return
//!
cv::Mat LMEstimator::getTrans()
{
    return m_trans;
}



//!
//! \brief medianShift
//! \param src
//! \param dst
//! \param dx
//! \param dy
//!
void medianShift(std::vector<cv::Point2f>src, std::vector<cv::Point2f> dst, float* dx, float* dy)
{
    assert(src.size() == dst.size());
    float* len = (float*)malloc(src.size() * sizeof(float));
    float* dir = (float*)malloc(src.size() * sizeof(float));
    assert((len != NULL) && (dir != NULL));
    float x, y;

    for (uint i = 0; i < src.size(); i++)
    {
        x = dst[i].x - src[i].x;
        y = dst[i].y - src[i].y;
        len[i] = sqrtf(x * x + y * y);
        dir[i] = atan2f(y, x);
    }

    std::vector<float> lenVec(len, len + src.size());
    std::vector<float> dirVec(dir, dir + src.size());
    std::sort(lenVec.begin(), lenVec.begin() + src.size());
    std::sort(dirVec.begin(), dirVec.begin() + src.size());
    float medLen, medDir;
    int iMid = src.size() / 2;

    if (src.size() % 0x01)
    {
        medLen = lenVec[iMid];
        medDir = dirVec[iMid];
    }
    else
    {
        medLen = (lenVec[iMid - 1] + lenVec[iMid]) / 2.0f;
        medDir = (dirVec[iMid - 1] + dirVec[iMid]) / 2.0f;
    }

    *dx = medLen * cosf(medDir);
    *dy = medLen * sinf(medDir);
}
