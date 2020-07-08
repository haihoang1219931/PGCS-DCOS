
#include "ipsearch_orbSearcher.h"


namespace ip {

namespace objsearch {

    /**
     * @brief ORBSearcher::ORBSearcher
     */
    ORBSearcher::ORBSearcher()
    {
        m_allBBoxs.resize(NUM_DESCRIPTOR);
        m_allImgKeypoints.resize(NUM_DESCRIPTOR);
        m_allImgDescriptors.resize(NUM_DESCRIPTOR);
        av2Keypoints = 0;
    }

    /**
     * @brief ORBSearcher::~ORBSearcher
     */
    ORBSearcher::~ORBSearcher()
    {
       m_allBBoxs.clear();
       m_allImgKeypoints.clear();
       m_allImgDescriptors.clear();
    }

    int ORBSearcher::init()
    {
        m_bfMatcher = cv::DescriptorMatcher::create("BruteForce-Hamming");       

        return 0;
    }

    bool ORBSearcher::update(cv::Mat &_inputImg, std::vector<cv::Point2f> &_corners, std::vector<int>& _objectsColor)

    {

        float tempWidth, tempHeight;
        tempWidth = std::max(std::max((int)_corners[0].x, (int)_corners[1].x) , std::max( (int)_corners[2].x, (int)_corners[3].x))
                - std::min(std::min((int)_corners[0].x, (int)_corners[1].x) , std::min( (int)_corners[2].x, (int)_corners[3].x));

        tempHeight = std::max(std::max((int)_corners[0].y, (int)_corners[1].y) , std::max( (int)_corners[2].y, (int)_corners[3].y))
                - std::min(std::min((int)_corners[0].y, (int)_corners[1].y) , std::min( (int)_corners[2].y, (int)_corners[3].y));        

        cv::Rect region2Compute;
        cv::Point topleftOfRegion;

        topleftOfRegion.x = std::min(std::min((int)_corners[0].x, (int)_corners[1].x) , std::min( (int)_corners[2].x, (int)_corners[3].x));
        topleftOfRegion.y = std::min(std::min((int)_corners[0].y, (int)_corners[1].y) , std::min( (int)_corners[2].y, (int)_corners[3].y));

        topleftOfRegion.x -= (int)((WIDTH_REGION_2_COMPUTE_KP - tempWidth)/2);
        topleftOfRegion.y -= (int)((HEIGHT_REGION_2_COMPUTE_KP - tempHeight)/2);

        region2Compute.x = std::max(topleftOfRegion.x, 0);
        region2Compute.y = std::max(topleftOfRegion.y, 0);

        region2Compute.width = std::min(WIDTH_REGION_2_COMPUTE_KP, _inputImg.cols - region2Compute.x);
        region2Compute.height = std::min(HEIGHT_REGION_2_COMPUTE_KP, _inputImg.rows - region2Compute.y);

        cv::Mat imgCrop = _inputImg(region2Compute);

        std::vector<cv::Point2f> tempRoi(_corners.size());
        for(size_t i = 0; i < _corners.size(); i++)
        {
            tempRoi[i].x = _corners[i].x - region2Compute.x;
            tempRoi[i].y = _corners[i].y - region2Compute.y;
        }

        cv::Point* ptMask = new cv::Point[tempRoi.size()];
        const cv::Point* ptContain = {&ptMask[0]};
        int iSize = static_cast<int>(tempRoi.size());

        for(size_t j = 0; j < tempRoi.size(); j++)
        {
            ptMask[j].x = static_cast<int>(tempRoi[j].x);
            ptMask[j].y = static_cast<int>(tempRoi[j].y);
        }
        cv::Mat matMask = cv::Mat::zeros(imgCrop.size(), CV_8UC1);
        fillPoly(matMask, &ptContain, &iSize, 1, cv::Scalar::all(255));

        std::vector<cv::KeyPoint> updateKeypoints;
        cv::Mat updateDescriptors;

        m_orbUpdate.setObjColor(_objectsColor);

        m_orbUpdate.detectAndCompute(imgCrop, matMask, updateKeypoints, updateDescriptors);

        for(size_t i = 0; i < updateKeypoints.size(); i++)
        {
            updateKeypoints[i].pt.x += region2Compute.x;
            updateKeypoints[i].pt.y += region2Compute.y;
        }          

        delete [] ptMask;        

        bool updateFlag = false;

        for(int i = 0; i < NUM_DESCRIPTOR; i++)
        {
            if (m_allImgKeypoints[i].size() == 0)
            {
                m_allImgKeypoints[i] =  updateKeypoints;
                m_allImgDescriptors[i] =  updateDescriptors;
                m_allBBoxs[i] = _corners;
                av2Keypoints += updateKeypoints.size();
                updateFlag = true;
                return updateFlag;
            }
        }

        if (!updateFlag && updateKeypoints.size() > av2Keypoints*0.7/NUM_DESCRIPTOR)
        {
            m_allImgKeypoints.erase(m_allImgKeypoints.begin());
            m_allImgKeypoints.push_back(updateKeypoints);

            m_allImgDescriptors.erase(m_allImgDescriptors.begin());
            m_allImgDescriptors.push_back(updateDescriptors );

            m_allBBoxs.erase(m_allBBoxs.begin());
            m_allBBoxs.push_back(_corners );
            updateFlag = true;
        }        

        return updateFlag;
    }

    bool ORBSearcher::update( cv::Mat &_inputImg, std::vector<cv::Rect> &obj_roi){
        if(_inputImg.empty() || obj_roi.empty()){
            std::cout << "Input wrong!" << std::endl;
            return false;
        }
        cv::Rect uBox = obj_roi[0];
        std::vector<cv::Point2f> _corners;

        _corners.push_back(cv::Point2f(static_cast<float>(uBox.x), static_cast<float>(uBox.y)));
        _corners.push_back(cv::Point2f(static_cast<float>(uBox.x + uBox.width), static_cast<float>(uBox.y)));
        _corners.push_back(cv::Point2f(static_cast<float>(uBox.x + uBox.width), static_cast<float>(uBox.y + uBox.height)));
        _corners.push_back(cv::Point2f(static_cast<float>(uBox.x), static_cast<float>(uBox.y + uBox.height)));

        cv::Mat cropImg = _inputImg(uBox).clone();
        m_objectColor.clear();
        detectObjectColors(cropImg, m_objectColor);
        return update(_inputImg, _corners, m_objectColor);
    }

    /**
     * @brief ORBSearcher::predict
     */
    int ORBSearcher::predict(cv::Mat &_inputImg, cv::Point2f &_iCenter,
                             cv::RotatedRect &_detectedObjBound, std::vector<int>& _objectsColor,
                             bool &_isFound, bool& _globalSearch, bool & _updateFlag)
    {
        std::vector<cv::KeyPoint> searchKeypoints;
        cv::Mat searchDescriptors;

        if (m_allImgDescriptors.size() == 0)
        {
            std::cout << "False, descriptors had not created ..." << std::endl;
            _isFound = false;
            return -1;
        }
        if (!_globalSearch)
        {
            cv::Rect region2Compute;
            cv::Point topleftOfRegion;

            topleftOfRegion.x = _iCenter.x - (int)(WIDTH_REGION_2_COMPUTE_KP/2);
            topleftOfRegion.y = _iCenter.y - (int)(HEIGHT_REGION_2_COMPUTE_KP/2);

            region2Compute.x = std::max(topleftOfRegion.x, 0);
            region2Compute.y = std::max(topleftOfRegion.y, 0);

            region2Compute.width = std::min(WIDTH_REGION_2_COMPUTE_KP, _inputImg.cols - region2Compute.x);
            region2Compute.height = std::min(HEIGHT_REGION_2_COMPUTE_KP, _inputImg.rows - region2Compute.y);


            cv::Mat imgCrop = _inputImg(region2Compute);           

            m_orbPredict.setObjColor(_objectsColor);

            m_orbPredict.detectAndCompute(imgCrop, cv::noArray(), searchKeypoints, searchDescriptors);

            if (!searchKeypoints.empty())
            {
                for(size_t i = 0; i < searchKeypoints.size(); i++)
                {
                    searchKeypoints[i].pt.x += region2Compute.x;
                    searchKeypoints[i].pt.y += region2Compute.y;
                }
            }
            else {
                _isFound = false;
                return -1;
            }

        }
        else {

            m_orbGlobalSearch.setObjColor(_objectsColor);
            m_orbGlobalSearch.detectAndCompute(_inputImg, cv::noArray(), searchKeypoints, searchDescriptors);

        }

        if(!searchKeypoints.empty()){
            /* *********************************
             *         search
             * *********************************/

            std::vector<std::vector<cv::DMatch> > allKnnMatches;
            std::vector<cv::KeyPoint> matched1, matched2;

            int bestMatched;

            std::vector<std::vector<int> > goodMatchCount(m_allImgDescriptors.size(), std::vector<int>(2));
            for (int patternIdx = 0; patternIdx < static_cast<int>(m_allImgDescriptors.size()) ; patternIdx++)
            {
                std::vector<std::vector< cv::DMatch> > oneMatches;
                std::vector<cv::DMatch> goodMatches;
                goodMatchCount[patternIdx][1] = patternIdx;

                m_bfMatcher->knnMatch(m_allImgDescriptors[patternIdx], searchDescriptors, oneMatches, 2);

                for(auto& matchPair:oneMatches)
                {
                    if(matchPair.size() > 1 && matchPair[0].distance < MATCH_RATIO*matchPair[1].distance)
                    {
                        goodMatches.push_back(matchPair[0]);
                        goodMatchCount[patternIdx][0]++;
                    }
                }
                allKnnMatches.push_back(goodMatches);
            }

            sort(goodMatchCount.begin(), goodMatchCount.end());

            bestMatched = goodMatchCount[m_allImgDescriptors.size()-1][1];

            for(auto& matched : allKnnMatches[bestMatched])
            {


                matched1.push_back(m_allImgKeypoints[bestMatched][matched.queryIdx]);
                matched2.push_back(searchKeypoints[matched.trainIdx]);


            }

            /* ************************************
             *        find object
             * ************************************/

            cv::Mat inlinerMatrix;
            cv::videostab::RansacParams ransacParams;
            ransacParams.eps = 0.5f;
            ransacParams.prob = 0.99f;
            ransacParams.size = 4;
            ransacParams.thresh = 10;
            int numInlier = 0;

            if(matched1.size() >= 4 )
            {
                inlinerMatrix = cv::videostab::estimateGlobalMotionRansac(Points(matched1), Points(matched2),
                                                                          cv::videostab::MM_SIMILARITY, ransacParams, nullptr, &numInlier);
            }

            if (matched1.size() < 4 || inlinerMatrix.empty())
            {
                _isFound = false;
                return -1;
            }

            stats.matches = (int)matched1.size();
            stats.inliers = numInlier;
            stats.ratio = stats.inliers*1.0/stats.matches;

            //        int InlierPerObjKeypoints = numInlier/m_allImgKeypoints[bestMatched].size();

            std::vector<cv::Point2f> newBound;

            cv::perspectiveTransform(m_allBBoxs[bestMatched], newBound, inlinerMatrix);

            if (numInlier > MIN_INLINER_KEYPOINTS && stats.ratio >= 0.05)
            {
                float scale = sqrtf(inlinerMatrix.at<float>(0, 0)*inlinerMatrix.at<float>(0, 0)
                                    + inlinerMatrix.at<float>(0, 1)*inlinerMatrix.at<float>(0, 1));
                if(scale > 0.25 && scale < 4)
                {
                    _detectedObjBound = cv::minAreaRect(newBound);
                    _iCenter = _detectedObjBound.center;
                    _isFound = true;
                    if (stats.ratio > 0.7 || numInlier > 9)
                        //                if (InlierPerObjKeypoints > 0.05)
                        _updateFlag = true;
                    return 0;
                }
            }
            else {
                _isFound = false;
                return -1;

            }

        }



        return 0;
    }

    int ORBSearcher::evaluationFeature(cv::Mat &inlinerMatrix, std::vector<cv::KeyPoint> &searchKeypoints,
                                       cv::Mat &searchDescriptors,int &bestMatched,Stats &tempStats){
        /* *********************************
         *         search
         * *********************************/

        std::vector<std::vector<cv::DMatch> > allKnnMatches;
        std::vector<cv::KeyPoint> matched1, matched2;

        std::vector<std::vector<int> > goodMatchCount(m_allImgDescriptors.size(), std::vector<int>(2));
        for (int patternIdx = 0; patternIdx < static_cast<int>(m_allImgDescriptors.size()) ; patternIdx++)
        {
            std::vector<std::vector< cv::DMatch> > oneMatches;
            std::vector<cv::DMatch> goodMatches;
            goodMatchCount[patternIdx][1] = patternIdx;

            m_bfMatcher->knnMatch(m_allImgDescriptors[patternIdx], searchDescriptors, oneMatches, 2);

            for(auto& matchPair:oneMatches)
            {
                if(matchPair.size() > 1 && matchPair[0].distance < MATCH_RATIO*matchPair[1].distance)
                {
                    goodMatches.push_back(matchPair[0]);
                    goodMatchCount[patternIdx][0]++;
                }
            }
            allKnnMatches.push_back(goodMatches);
        }

        sort(goodMatchCount.begin(), goodMatchCount.end());

        bestMatched = goodMatchCount[m_allImgDescriptors.size()-1][1];

        for(auto& matched : allKnnMatches[bestMatched])
        {
            matched1.push_back(m_allImgKeypoints[bestMatched][matched.queryIdx]);
            matched2.push_back(searchKeypoints[matched.trainIdx]);
        }

        /* ************************************
         *        find object
         * ************************************/

        cv::videostab::RansacParams ransacParams;
        ransacParams.eps = 0.5f;
        ransacParams.prob = 0.99f;
        ransacParams.size = 4;
        ransacParams.thresh = 10;
        int numInlier = 0;
        if(matched1.size() >= 4 )
        {
            inlinerMatrix = cv::videostab::estimateGlobalMotionRansac(Points(matched1), Points(matched2),
                                                                      cv::videostab::MM_SIMILARITY, ransacParams, nullptr, &numInlier);
        }

        if (matched1.size() < 4 || inlinerMatrix.empty())
        {
            return 0;
        }


        tempStats.matches = (int)matched1.size();
        tempStats.inliers = numInlier;
        tempStats.ratio = tempStats.inliers*1.0/tempStats.matches;

        //        int InlierPerObjKeypoints = numInlier/m_allImgKeypoints[bestMatched].size();
        return numInlier;
    }

    bool ORBSearcher::predict( cv::Mat &_inputImg, std::vector<cv::Rect> &suggestObjectSet,
                 cv::RotatedRect &_detectedObjBound)
    {
        if (m_allImgDescriptors.size() == 0)
        {
            std::cout << "False, the searcher not initial yet ..." << std::endl;
            return false;
        }
        std::vector<std::vector<cv::KeyPoint> > ssKeypoints;
        std::vector<cv::Mat> ssDescriptors;
        if(suggestObjectSet.empty()){
//            std::vector<cv::KeyPoint> searchKeypoints;
//            cv::Mat searchDescriptors;
//            m_orbGlobalSearch.setObjColor(m_objectColor);
//            m_orbGlobalSearch.detectAndCompute(_inputImg, cv::noArray(), searchKeypoints, searchDescriptors);
//            if(!searchKeypoints.empty()){
//                ssKeypoints.push_back(searchKeypoints);
//                ssDescriptors.push_back(searchDescriptors);
//            }
        }else{
            for(auto r : suggestObjectSet){
                std::vector<cv::KeyPoint> searchKeypoints;
                cv::Mat searchDescriptors;

                cv::Point2f _iCenter = (r.br() + r.tl())/2.0;
                cv::Rect region2Compute;
                cv::Point topleftOfRegion;

                cv::Size rSize(r.width*2,r.height*2);

//                topleftOfRegion.x = _iCenter.x - (int)(WIDTH_REGION_2_COMPUTE_KP/2);
//                topleftOfRegion.y = _iCenter.y - (int)(HEIGHT_REGION_2_COMPUTE_KP/2);

                topleftOfRegion.x = _iCenter.x - r.width;
                topleftOfRegion.y = _iCenter.y - r.height;

                region2Compute.x = std::max(topleftOfRegion.x, 0);
                region2Compute.y = std::max(topleftOfRegion.y, 0);

//                region2Compute.width = std::min(WIDTH_REGION_2_COMPUTE_KP, _inputImg.cols - region2Compute.x);
//                region2Compute.height = std::min(HEIGHT_REGION_2_COMPUTE_KP, _inputImg.rows - region2Compute.y);
                region2Compute.width = std::min(rSize.width, _inputImg.cols - region2Compute.x);
                region2Compute.height = std::min(rSize.height, _inputImg.rows - region2Compute.y);


                cv::Mat imgCrop = _inputImg(region2Compute);
                m_orbPredict.setObjColor(m_objectColor);

                m_orbPredict.detectAndCompute(imgCrop, cv::noArray(), searchKeypoints, searchDescriptors);
                if (!searchKeypoints.empty())
                {
                    for(size_t i = 0; i < searchKeypoints.size(); i++)
                    {
                        searchKeypoints[i].pt.x += region2Compute.x;
                        searchKeypoints[i].pt.y += region2Compute.y;
                    }
                    ssDescriptors.push_back(searchDescriptors);
                    ssKeypoints.push_back(searchKeypoints);
                }
            }
        }
        cv::Mat bestInlinerMatrix;
        int bestInstance = 0;
        int bestMatched;
        for(size_t i = 0;i < ssKeypoints.size();i ++){
            cv::Mat inlinerMatrix;
            Stats tempStats;
            int tempMatched;
            int tempNumInlier = evaluationFeature(inlinerMatrix,ssKeypoints[i],ssDescriptors[i],tempMatched,tempStats);
            if(tempNumInlier > bestInstance){
                stats = tempStats;
                bestMatched = tempMatched;
                bestInstance = tempNumInlier;
                inlinerMatrix.copyTo(bestInlinerMatrix);
            }
        }
        if(bestInstance > MIN_INLINER_KEYPOINTS && stats.ratio >= 0.05){
            float scale = sqrtf(bestInlinerMatrix.at<float>(0, 0)*bestInlinerMatrix.at<float>(0, 0)
                                + bestInlinerMatrix.at<float>(0, 1)*bestInlinerMatrix.at<float>(0, 1));
            if(scale > 0.25 && scale < 4)
            {
                std::vector<cv::Point2f> newBound;
                cv::perspectiveTransform(m_allBBoxs[bestMatched], newBound, bestInlinerMatrix);
                _detectedObjBound = cv::minAreaRect(newBound);
                if (stats.ratio > 0.7 || bestInstance > 9){
                    //                if (InlierPerObjKeypoints > 0.05)
                    //_updateFlag = true;
                }
                return true;
            }
        }
        return false;
    }

    void ORBSearcher::detectObjectColors(cv::InputArray &_img, std::vector<int> &_objectsColor)
    {
        m_orbUpdate.detectObjectColors(_img, _objectsColor);
    }


}

}
