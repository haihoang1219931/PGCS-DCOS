#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H
#include <vector>
//#include <math.h>
#include <list>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <math.h>
#include <algorithm>
#include <iostream>

#include "ipsearch_utils.h"


namespace ip {
namespace objsearch {

    class ORBextractor
    {
    public:
//        ORBextractor();

        explicit ORBextractor(int _nfeatures, float _scaleFactor, int _nlevels, int _edgeThreshold,
                     int _firstlevel,int _scoreType, int _patchSize, int _fastThreshold);

        ~ORBextractor(){}

        enum {HARRIS_SCORE=0, FAST_SCORE=1 };

        // Compute the ORB features and descriptors on an image.
        // ORB are dispersed on the image using an octree.
        // Mask is ignored in the current implementation.

        void detectAndCompute(cv::InputArray _image, cv::InputArray _mask,
                              std::vector<cv::KeyPoint>& keypoints, cv::OutputArray _descriptors);
        void detectObjectColors(cv::InputArray _img,
                                std::vector<int>& objectsColors);

        void setObjColor(std::vector<int> _objectColors)
        {
            m_objectColor.clear();
            for (uint i = 0; i < _objectColors.size(); i++)
                m_objectColor.push_back(_objectColors[i]);
        }

        int inline GetLevels(){
            return nlevels;}


    protected:

        void computeKeypointsOld(const cv::Mat imagePyramid,
                                 const cv::Mat maskPyramid,
                                 const std::vector<cv::Rect> layerInfo,
                                 const std::vector<float> layerScale,
                                 std::vector<cv::KeyPoint> & allKeypoints,
                                 int nfeatures, double scaleFactor,
                                 int edgeThreshod, int patchSize,
                                 int scoreType, int fastThreshold);

    protected:
        int nfeatures;
        double scaleFactor;
        int nlevels;
        int edgeThreshold;
        int firstLevel;       
        int scoreType;
        int patchSize;
        int fastThreshold;        
        std::vector<int> m_objectColor;        
    };

}

}


#endif // ORBEXTRACTOR_H
