#ifndef IPSEARCH_ORBSEARCHER_H
#define IPSEARCH_ORBSEARCHER_H


#include "ipsearch_utils.h"
#include "ipsearch_orbextractor.h"


namespace ip {
namespace objsearch {

    class ORBSearcher
    {
    public:
        ORBSearcher();
        ~ORBSearcher();

    public:


        int init();
        /**
         * @brief update
         * @param _inputImg
         * @param _corners
         * @return
         */
        bool update( cv::Mat &_inputImg, std::vector<cv::Point2f> &_corners, std::vector<int>& _objectsColor);
        /**
         * @brief update
         * @param _inputImg
         * @param _obj_roi
         * @return true/false
         */
        bool update( cv::Mat &_inputImg, std::vector<cv::Rect> &obj_roi);
        /**
         * @brief predict
         * @param _inputImg
         * @param _iCenter
         * @param _detectedObjBound
         * @param _isFound
         * @param globalSearch
         * @return
         */
        int predict( cv::Mat &_inputImg, cv::Point2f &_iCenter,
                     cv::RotatedRect &_detectedObjBound, std::vector<int>& objectsColor,
                     bool &_isFound, bool& _globalSearch, bool& _updateFlag);
        /**
         * @brief evaluation the best candidate of object cell
         * @param _input image
         * @param set of object candidates input need to evaluation if empty search all image
         * @param out put of rotate rectange bound the best candidate object
         * @return true/false if found the best candidate satisfy
         */
        bool predict( cv::Mat &_inputImg, std::vector<cv::Rect> &suggestObjectSet,
                     cv::RotatedRect &_detectedObjBound);



        void detectObjectColors(cv::InputArray& _img, std::vector<int>& _objectsColor);

//        void setObjColor(std::vector<int> _objectColors)
//        {
//            m_objectColor.reserve(0);
//            for (uint i = 0; i < _objectColors.size(); i++)
//                m_objectColor.push_back(_objectColors[i]);
//        }


    private: // Private Functions
        int evaluationFeature(cv::Mat &inlinerMatrix,std::vector<cv::KeyPoint> &searchKeypoints
                              ,cv::Mat &searchDescriptors,int &bestMatched,Stats &tempStats);
    private: // Private Datas
        std::vector<std::vector<cv::KeyPoint> > m_allImgKeypoints;
        std::vector<cv::Mat> m_allImgDescriptors;
        std::vector<std::vector<cv::Point2f>> m_allBBoxs;
        std::vector<cv::Mat> m_allColorName;
        std::vector<int> m_objectColor;


//        cv::Ptr<cv::ORB> m_orbUpdate;
//        cv::Ptr<cv::ORB> m_orbPredict;
//        cv::Ptr<cv::ORB> m_orbGlobalSearch;
//        ORBextractor m_orbUpdate;
//        ORBextractor m_orbPredict;
//        ORBextractor m_orbGlobalSearch;

        //        m_orbUpdate = cv::ORB::create(NUM_KEYPOINTS_UPDATE,
        //                                  SCALE_FACTOR,
        //                                  SCALE_LEVEL, EDGE_THRESHOLD, 0, 2,
        //                                  cv::ORB::HARRIS_SCORE,
        //                                  PATCH_SIZE, FAST_THRESHOLD);
        ORBextractor m_orbUpdate = ORBextractor(NUM_KEYPOINTS_UPDATE, SCALE_FACTOR, SCALE_LEVEL,
                                                EDGE_THRESHOLD, 0, cv::ORB::HARRIS_SCORE,
                                                PATCH_SIZE, FAST_THRESHOLD);

        ORBextractor m_orbPredict = ORBextractor(NUM_KEYPOINTS_PREDICT, SCALE_FACTOR, SCALE_LEVEL,
                                                 EDGE_THRESHOLD, 0, cv::ORB::HARRIS_SCORE,
                                                 PATCH_SIZE, FAST_THRESHOLD);


        ORBextractor m_orbGlobalSearch = ORBextractor(NUM_KEYPOINTS_GLOBAL_SEARCH, SCALE_FACTOR, SCALE_LEVEL,
                                                      EDGE_THRESHOLD, 0, cv::ORB::HARRIS_SCORE,
                                                      PATCH_SIZE, FAST_THRESHOLD);

        int av2Keypoints;

        cv::Ptr<cv::DescriptorMatcher> m_bfMatcher;    

        Stats stats;

    public:
        struct Dist{
            bool operator() (const cv::Point& a, const cv::Point& b)
            {
                return sqrtf((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
            }
        };
    };
}

}

#endif // IPSEARCH_ORBSEARCHER_H
