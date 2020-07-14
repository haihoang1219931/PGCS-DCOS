#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <math.h>
#include <algorithm>
#include <iostream>

#include "ipsearch_orbextractor.h"

namespace ip {

namespace objsearch {



using namespace cv;
using namespace std;

const float HARRIS_K = 0.04f;

static void HarrisResponsesOld(const cv::Mat& img, const std::vector<cv::Rect>& layerinfo,
                               std::vector<cv::KeyPoint>& pts, int blockSize, float harris_k)
{
    CV_Assert(img.type() == CV_8UC1 && blockSize*blockSize <= 2048);
    size_t ptidx, ptsize = pts.size();

    const uchar* ptr00 = img.ptr<uchar>();
    int step = (int)(img.step/img.elemSize1());
    int r = blockSize/2;

    float scale = 1.f/((1<<2)*blockSize*255.f);
    float scale_sq_sq = scale*scale*scale*scale;

    AutoBuffer<int> ofsbuf(blockSize*blockSize);
//    int* ofs = ofsbuf.data();
    int* ofs = (ofsbuf);
    for(int i = 0; i < blockSize; i++)
        for(int j = 0; j < blockSize; j++)
            ofs[i*blockSize + j] = (int)(i*step + j);

    for(ptidx = 0; ptidx < ptsize; ptidx++)
    {
        int x0 = cvRound(pts[ptidx].pt.x);
        int y0 = cvRound(pts[ptidx].pt.y);
        int z = pts[ptidx].octave;

        const uchar* ptr0 = ptr00 + (y0 - r +layerinfo[z].y)*step + x0 - r + layerinfo[z].x;
        int a = 0, b = 0, c = 0;

        for(int k = 0; k < blockSize*blockSize; k++)
        {
            const uchar* ptr = ptr0 + ofs[k];
            int Ix = (ptr[1] - ptr[-1])*2 + (ptr[-step + 1] - ptr[-step - 1]) + (ptr[step + 1] - ptr[step - 1]);
            int Iy = (ptr[step] - ptr[-step])*2 + (ptr[step - 1] - ptr[-step - 1]) + (ptr[step + 1] - ptr[-step + 1]);

            a += Ix*Ix;
            b += Iy*Iy;
            c += Ix*Iy;
        }

        pts[ptidx].response = ((float)a*b - (float)c*c - harris_k*((float)a + b)*((float)a + b))*scale_sq_sq;
    }
}


static void ICAnglesOld(const cv::Mat& img, const std::vector<cv::Rect>& layerinfo,
                        std::vector<cv::KeyPoint>& pts, const std::vector<int>& u_max, int half_k)
{
    int step = (int)img.step1();
    size_t ptidx, ptsize = pts.size();

    for(ptidx = 0; ptidx < ptsize; ptidx++)
    {
        const cv::Rect& layer = layerinfo[pts[ptidx].octave];
        const uchar* center = &img.at<uchar>(cvRound(pts[ptidx].pt.y) + layer.y, cvRound(pts[ptidx].pt.x) + layer.x);

        int m_01 = 0, m_10 = 0;
        for(int u = -half_k; u <= half_k; u++)
        {
            m_10 += u*center[u];
        }

        for(int v = 1; v <= half_k; v++)
        {
            int v_sum = 0;
            int d = u_max[v];
            for(int u = -d; u <= d; ++u)
            {
                int val_plus = center[u + v*step], val_minus = center[u - v*step];
                v_sum += (val_plus - val_minus);
                m_10 += u*(val_plus + val_minus);
            }
            m_01 += v*v_sum;

        }

        pts[ptidx].angle = fastAtan2((float)m_01, (float)m_10);
    }
}

static void makeRandomPattern(int patchSize, cv::Point* pattern, int npoints)
{
    cv::RNG rng(0x34985739);
    for (int i = 0; i < npoints; i++)
    {
        pattern[i].x = rng.uniform(-patchSize/2, patchSize/2 + 1);
        pattern[i].y = rng.uniform(-patchSize/2, patchSize/2 + 1);
    }
}

static void initializeOrbPattern(const cv::Point* pattern0, std::vector<cv::Point>& pattern,
                                 int ntuples, int tupleSize, int poolSize)
{
    cv::RNG rng(0x34985739);
    int i, k, k1;
    pattern.resize(ntuples*tupleSize);

    for(i = 0; i < ntuples; i++)
    {
        for(k = 0; k < tupleSize; k++)
        {
            for (;;)
            {
                int idx = rng.uniform(0, poolSize);
                cv::Point pt = pattern0[idx];
                for(k1 = 0; k1 < k; k1++)
                    if(pattern[tupleSize*i + k1] == pt)
                        break;

                if(k1 == k)
                {
                    pattern[tupleSize*i + k] = pt;
                    break;
                }
            }
        }
    }
}

void ORBextractor::detectObjectColors(cv::InputArray& _img,
                                      std::vector<int>& objectsColors)
{
    cv::Mat image = _img.getMat();
    CV_Assert(image.type() == CV_8UC3);

    int borderwidth = (int)image.cols/5;
    int borderheight = (int)image.rows/5;

    cv::Rect objectbody;
    objectbody.x = (int)1.5*borderwidth;
    objectbody.y = (int)1.5*borderheight;
    objectbody.width = image.cols - 3*borderwidth;
    objectbody.height = image.rows - 3*borderheight;

    cv::Mat roiImage = image(objectbody);

    std::vector<cv::Mat> im_split;
    cv::split(roiImage, im_split);
    cv::Mat channelR = im_split[2];
    cv::Mat channelG = im_split[1];
    cv::Mat channelB = im_split[0];

    const uchar* Rdata = &channelR.at<uchar>(0,0);
    const uchar* Gdata = &channelG.at<uchar>(0,0);
    const uchar* Bdata = &channelB.at<uchar>(0,0);

    double R, G, B;
    const int* colorNames = &ColorNames[0];
    double numPixels = channelR.cols*channelR.rows;

    std::vector<std::vector<int> > colorCount(NUM_COLORS_NAME, std::vector<int>(2));
    for (size_t i = 0; i < NUM_COLORS_NAME; i++)
        colorCount[i][1] = (int)(i + 1);

#define GET_OC(idx) \
    (R = *(Rdata + idx), \
    G = *(Gdata + idx), \
    B = *(Bdata + idx), \
    *(colorNames + (int)(floor(R/8) + 32*floor(G/8) + 32*32*floor(B/8))))

    for (size_t i = 0; i < numPixels; i++)
    {
        int k = GET_OC(i);
        if (k > 0)
            colorCount[k-1][0]++;
    }

    sort(colorCount.begin(), colorCount.end());
    objectsColors.clear();  

    for (int i = 0; i< (int)colorCount.size(); i++)
    {

        if (colorCount[colorCount.size() - 1 - i][0] / numPixels > 0.15)
        {
            objectsColors.push_back(colorCount[colorCount.size() - 1 - i][1]);            
        }

    }

    channelB.release();
    channelG.release();
    channelR.release();
}

static bool checkColorInObject(std::vector<int> _objectcolors, int t)
{
    bool result = false;
    for (int i = 0; i < _objectcolors.size(); i++)
    {
        if (t == _objectcolors[i])
        {            
            result = true;
            return result;
            break;
        }
    }
    return result;
}

class ParallelComputeCNDescriptors : public ParallelLoopBody
{
public:
    ParallelComputeCNDescriptors ( std::vector<cv::Mat>& _im_split,  cv::Mat& _imageGrayPyramid,
                                   std::vector<cv::Rect>& _layerInfo,
                                   std::vector<float>& _layerScale,
                                  std::vector<cv::KeyPoint>& _keypoints, cv::Mat& _descriptors,
                                   std::vector<cv::Point>& _pattern,
                                  int _dsize, std::vector<int> _objectcolors): descriptors(_descriptors)
    {
        this->im_split = _im_split;
        this->imageGrayPyramid = _imageGrayPyramid;
        this->layerInfo = _layerInfo;
        this->layerScale = _layerScale;
        this->keypoints = _keypoints;
        this->m_pattern = _pattern;
        this->dsize = _dsize;        
        this->objectcolors = _objectcolors;
    }

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        cv::Mat channelR = im_split[2];
        cv::Mat channelG = im_split[1];
        cv::Mat channelB = im_split[0];

        int step = (int)imageGrayPyramid.step;

        for (int j = range.start; j < range.end; j++)
        {
            const cv::KeyPoint& kpt = keypoints[j];
            const cv::Rect& layer = layerInfo[kpt.octave];
            float scale = 1.f/layerScale[kpt.octave];
            float angle = kpt.angle;

            angle *= (float)(CV_PI/180.f);
            float a = (float)cos(angle), b = (float)sin(angle);

            const uchar* data = &imageGrayPyramid.at<uchar>(cvRound(kpt.pt.y*scale) + layer.y,
                                                            cvRound(kpt.pt.x*scale) + layer.x);

            const uchar* Rdata = &channelR.at<uchar>(cvRound(kpt.pt.y*scale) + layer.y,
                                                     cvRound(kpt.pt.x*scale) + layer.x);
            const uchar* Gdata = &channelG.at<uchar>(cvRound(kpt.pt.y*scale) + layer.y,
                                                     cvRound(kpt.pt.x*scale) + layer.x);
            const uchar* Bdata = &channelB.at<uchar>(cvRound(kpt.pt.y*scale) + layer.y,
                                                     cvRound(kpt.pt.x*scale) + layer.x);

            int cnkpt = ColorNames[(int)(floor(*Rdata/8) + 32*floor(*Gdata/8) + 32*32*floor(*Bdata/8))];

            bool checkColor = false;

            for (int i = 0; i < objectcolors.size(); i++)
            {
                if (cnkpt == objectcolors[i])
                {
                    checkColor = true;
                    break;
                }
            }

            float x, y;
            int ix, iy;
            double RR, GG, BB;
            int sizeObColors = objectcolors.size();
            bool valid = true, invalid = false;

            const cv::Point* pattern = &m_pattern[0];
            const cv::Point* patternCN = &m_pattern[0];
            const int* colorNames = &ColorNames[0];
            uchar* desc = (uchar*) descriptors.ptr<uchar>(j);

#define  GET_CN(idx) \
    (ix = patternCN[idx].x, \
    iy = patternCN[idx].y, \
    RR = *(Rdata + iy*step + ix), \
    GG = *(Gdata + iy*step + ix), \
    BB = *(Bdata + iy*step + ix), \
    *(colorNames + (int)(floor(RR/8) + 32*floor(GG/8) + 32*32*floor(BB/8))))    


#define GET_VALUE(idx) \
    (x = pattern[idx].x*a - pattern[idx].y*b, \
    y = pattern[idx].x*b + pattern[idx].y*a, \
    ix = cvRound(x), \
    iy = cvRound(y), \
    *(data + iy*step + ix))


//            if (checkColor)
//            {
            for(int i = 0; i < dsize; i++)
            {
                int val;
                int t0, t1;
                if (i > 31 && i < 64 )
                {
                    t0 = GET_CN(0); t1 = GET_CN(1);
//                    val = (checkColor && t0 == cnkpt && t1 == cnkpt);
                    val = (checkColor && checkColorInObject(objectcolors, t0) && checkColorInObject(objectcolors, t1));
                    t0 = GET_CN(2); t1 = GET_CN(3);
//                    val |= (checkColor && t0 == cnkpt && t1 == cnkpt) << 1;
                    val |= (checkColor && checkColorInObject(objectcolors, t0) && checkColorInObject(objectcolors, t1)) << 1;
                    t0 = GET_CN(4); t1 = GET_CN(5);
//                    val |= (checkColor && t0 == cnkpt && t1 == cnkpt) << 2;
                    val |= (checkColor && checkColorInObject(objectcolors, t0) && checkColorInObject(objectcolors, t1)) << 2;
                    t0 = GET_CN(6); t1 = GET_CN(7);
//                    val |= (checkColor && t0 == cnkpt && t1 == cnkpt) << 3;
                    val |= (checkColor && checkColorInObject(objectcolors, t0) && checkColorInObject(objectcolors, t1)) << 3;
                    t0 = GET_CN(8); t1 = GET_CN(9);
//                    val |= (checkColor && t0 == cnkpt && t1 == cnkpt) << 4;
                    val |= (checkColor && checkColorInObject(objectcolors, t0) && checkColorInObject(objectcolors, t1)) << 4;
                    t0 = GET_CN(10); t1 = GET_CN(11);
//                    val |= (checkColor && t0 == cnkpt && t1 == cnkpt) << 5;
                    val |= (checkColor && checkColorInObject(objectcolors, t0) && checkColorInObject(objectcolors, t1)) << 5;
                    t0 = GET_CN(12); t1 = GET_CN(13);
//                    val |= (checkColor && t0 == cnkpt && t1 == cnkpt) << 6;
                    val |= (checkColor && checkColorInObject(objectcolors, t0) && checkColorInObject(objectcolors, t1)) << 6;
                    t0 = GET_CN(14); t1 = GET_CN(15);
//                    val |= (checkColor && t0 == cnkpt && t1 == cnkpt) << 7;
                    val |= (checkColor && checkColorInObject(objectcolors, t0) && checkColorInObject(objectcolors, t1)) << 7;

                    desc[i] = (uchar)val;

                    patternCN += 16;

                }

                if (i < 32)
                {
                    t0 = GET_VALUE(0); t1 = GET_VALUE(1);
                    val = (t0 <= t1 && checkColor);
                    t0 = GET_VALUE(2); t1 = GET_VALUE(3);
                    val |= (t0 <= t1 && checkColor) << 1;
                    t0 = GET_VALUE(4); t1 = GET_VALUE(5);
                    val |= (t0 <= t1 && checkColor) << 2;
                    t0 = GET_VALUE(6); t1 = GET_VALUE(7);
                    val |= (t0 <= t1 && checkColor) << 3;
                    t0 = GET_VALUE(8); t1 = GET_VALUE(9);
                    val |= (t0 <= t1 && checkColor) << 4;
                    t0 = GET_VALUE(10); t1 = GET_VALUE(11);
                    val |= (t0 <= t1 && checkColor) << 5;
                    t0 = GET_VALUE(12); t1 = GET_VALUE(13);
                    val |= (t0 <= t1 && checkColor) << 6;
                    t0 = GET_VALUE(14); t1 = GET_VALUE(15);
                    val |= (t0 <= t1 && checkColor) << 7;

                    pattern += 16;
                    desc[i] = (uchar)val;

                }
                if ( i > 63 )
                {
                    if (i == 64)
                        pattern = & m_pattern[0];
                    t0 = GET_VALUE(14); t1 = GET_VALUE(15);
                    val = (t0 <= t1 );
                    t0 = GET_VALUE(13); t1 = GET_VALUE(12);
                    val |= (t0 <= t1) << 1;
                    t0 = GET_VALUE(11); t1 = GET_VALUE(10);
                    val |= (t0 <= t1) << 2;
                    t0 = GET_VALUE(9); t1 = GET_VALUE(8);
                    val |= (t0 <= t1) << 3;
                    t0 = GET_VALUE(7); t1 = GET_VALUE(6);
                    val |= (t0 <= t1 ) << 4;
                    t0 = GET_VALUE(5); t1 = GET_VALUE(4);
                    val |= (t0 <= t1) << 5;
                    t0 = GET_VALUE(3); t1 = GET_VALUE(2);
                    val |= (t0 <= t1) << 6;
                    t0 = GET_VALUE(1); t1 = GET_VALUE(0);
                    val |= (t0 <= t1) << 7;
                    pattern += 16;

                    desc[i] = (uchar)val;

                }

            } // end for

//        } // end if
        }
        channelR.release();
        channelG.release();
        channelB.release();
    }

    ParallelComputeCNDescriptors& operator = (const ParallelComputeCNDescriptors &) {}

private:
    std::vector<cv::Mat> im_split;
    cv::Mat imageGrayPyramid;
    std::vector<cv::Rect> layerInfo;
    std::vector<float> layerScale;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    std::vector<cv::Point> m_pattern;
    int dsize;    
    std::vector<int> objectcolors;

};


static void getw2c(std::vector<double>& w2c)
{
    std::ifstream ifstr;
    ofstream ofstr("../colornames.txt", std::ios_base::out);

    ifstr.open("../DD11_w2c_fast.txt", std::ios_base::in);
    w2c.clear();
    w2c.resize(32768);

    double tmp;

    for (size_t i = 1; i < 32768; i++)
    {
        for(size_t j = 0; j < 4; j++)
        {
            ifstr >> tmp;
        }
        w2c[i] = tmp;
        ofstr << tmp;
    }

    ifstr.close();
    ofstr.close();
}

static float getScale(int level, int firstLevel, double scaleFactor)
{
    return (float)std::pow(scaleFactor, (double)(level - firstLevel));
}


static int bit_pattern_31_[256*4] =
{
    8,-3, 9,5/*mean (0), correlation (0)*/,
    4,2, 7,-12/*mean (1.12461e-05), correlation (0.0437584)*/,
    -11,9, -8,2/*mean (3.37382e-05), correlation (0.0617409)*/,
    7,-12, 12,-13/*mean (5.62303e-05), correlation (0.0636977)*/,
    2,-13, 2,12/*mean (0.000134953), correlation (0.085099)*/,
    1,-7, 1,6/*mean (0.000528565), correlation (0.0857175)*/,
    -2,-10, -2,-4/*mean (0.0188821), correlation (0.0985774)*/,
    -13,-13, -11,-8/*mean (0.0363135), correlation (0.0899616)*/,
    -13,-3, -12,-9/*mean (0.121806), correlation (0.099849)*/,
    10,4, 11,9/*mean (0.122065), correlation (0.093285)*/,
    -13,-8, -8,-9/*mean (0.162787), correlation (0.0942748)*/,
    -11,7, -9,12/*mean (0.21561), correlation (0.0974438)*/,
    7,7, 12,6/*mean (0.160583), correlation (0.130064)*/,
    -4,-5, -3,0/*mean (0.228171), correlation (0.132998)*/,
    -13,2, -12,-3/*mean (0.00997526), correlation (0.145926)*/,
    -9,0, -7,5/*mean (0.198234), correlation (0.143636)*/,
    12,-6, 12,-1/*mean (0.0676226), correlation (0.16689)*/,
    -3,6, -2,12/*mean (0.166847), correlation (0.171682)*/,
    -6,-13, -4,-8/*mean (0.101215), correlation (0.179716)*/,
    11,-13, 12,-8/*mean (0.200641), correlation (0.192279)*/,
    4,7, 5,1/*mean (0.205106), correlation (0.186848)*/,
    5,-3, 10,-3/*mean (0.234908), correlation (0.192319)*/,
    3,-7, 6,12/*mean (0.0709964), correlation (0.210872)*/,
    -8,-7, -6,-2/*mean (0.0939834), correlation (0.212589)*/,
    -2,11, -1,-10/*mean (0.127778), correlation (0.20866)*/,
    -13,12, -8,10/*mean (0.14783), correlation (0.206356)*/,
    -7,3, -5,-3/*mean (0.182141), correlation (0.198942)*/,
    -4,2, -3,7/*mean (0.188237), correlation (0.21384)*/,
    -10,-12, -6,11/*mean (0.14865), correlation (0.23571)*/,
    5,-12, 6,-7/*mean (0.222312), correlation (0.23324)*/,
    5,-6, 7,-1/*mean (0.229082), correlation (0.23389)*/,
    1,0, 4,-5/*mean (0.241577), correlation (0.215286)*/,
    9,11, 11,-13/*mean (0.00338507), correlation (0.251373)*/,
    4,7, 4,12/*mean (0.131005), correlation (0.257622)*/,
    2,-1, 4,4/*mean (0.152755), correlation (0.255205)*/,
    -4,-12, -2,7/*mean (0.182771), correlation (0.244867)*/,
    -8,-5, -7,-10/*mean (0.186898), correlation (0.23901)*/,
    4,11, 9,12/*mean (0.226226), correlation (0.258255)*/,
    0,-8, 1,-13/*mean (0.0897886), correlation (0.274827)*/,
    -13,-2, -8,2/*mean (0.148774), correlation (0.28065)*/,
    -3,-2, -2,3/*mean (0.153048), correlation (0.283063)*/,
    -6,9, -4,-9/*mean (0.169523), correlation (0.278248)*/,
    8,12, 10,7/*mean (0.225337), correlation (0.282851)*/,
    0,9, 1,3/*mean (0.226687), correlation (0.278734)*/,
    7,-5, 11,-10/*mean (0.00693882), correlation (0.305161)*/,
    -13,-6, -11,0/*mean (0.0227283), correlation (0.300181)*/,
    10,7, 12,1/*mean (0.125517), correlation (0.31089)*/,
    -6,-3, -6,12/*mean (0.131748), correlation (0.312779)*/,
    10,-9, 12,-4/*mean (0.144827), correlation (0.292797)*/,
    -13,8, -8,-12/*mean (0.149202), correlation (0.308918)*/,
    -13,0, -8,-4/*mean (0.160909), correlation (0.310013)*/,
    3,3, 7,8/*mean (0.177755), correlation (0.309394)*/,
    5,7, 10,-7/*mean (0.212337), correlation (0.310315)*/,
    -1,7, 1,-12/*mean (0.214429), correlation (0.311933)*/,
    3,-10, 5,6/*mean (0.235807), correlation (0.313104)*/,
    2,-4, 3,-10/*mean (0.00494827), correlation (0.344948)*/,
    -13,0, -13,5/*mean (0.0549145), correlation (0.344675)*/,
    -13,-7, -12,12/*mean (0.103385), correlation (0.342715)*/,
    -13,3, -11,8/*mean (0.134222), correlation (0.322922)*/,
    -7,12, -4,7/*mean (0.153284), correlation (0.337061)*/,
    6,-10, 12,8/*mean (0.154881), correlation (0.329257)*/,
    -9,-1, -7,-6/*mean (0.200967), correlation (0.33312)*/,
    -2,-5, 0,12/*mean (0.201518), correlation (0.340635)*/,
    -12,5, -7,5/*mean (0.207805), correlation (0.335631)*/,
    3,-10, 8,-13/*mean (0.224438), correlation (0.34504)*/,
    -7,-7, -4,5/*mean (0.239361), correlation (0.338053)*/,
    -3,-2, -1,-7/*mean (0.240744), correlation (0.344322)*/,
    2,9, 5,-11/*mean (0.242949), correlation (0.34145)*/,
    -11,-13, -5,-13/*mean (0.244028), correlation (0.336861)*/,
    -1,6, 0,-1/*mean (0.247571), correlation (0.343684)*/,
    5,-3, 5,2/*mean (0.000697256), correlation (0.357265)*/,
    -4,-13, -4,12/*mean (0.00213675), correlation (0.373827)*/,
    -9,-6, -9,6/*mean (0.0126856), correlation (0.373938)*/,
    -12,-10, -8,-4/*mean (0.0152497), correlation (0.364237)*/,
    10,2, 12,-3/*mean (0.0299933), correlation (0.345292)*/,
    7,12, 12,12/*mean (0.0307242), correlation (0.366299)*/,
    -7,-13, -6,5/*mean (0.0534975), correlation (0.368357)*/,
    -4,9, -3,4/*mean (0.099865), correlation (0.372276)*/,
    7,-1, 12,2/*mean (0.117083), correlation (0.364529)*/,
    -7,6, -5,1/*mean (0.126125), correlation (0.369606)*/,
    -13,11, -12,5/*mean (0.130364), correlation (0.358502)*/,
    -3,7, -2,-6/*mean (0.131691), correlation (0.375531)*/,
    7,-8, 12,-7/*mean (0.160166), correlation (0.379508)*/,
    -13,-7, -11,-12/*mean (0.167848), correlation (0.353343)*/,
    1,-3, 12,12/*mean (0.183378), correlation (0.371916)*/,
    2,-6, 3,0/*mean (0.228711), correlation (0.371761)*/,
    -4,3, -2,-13/*mean (0.247211), correlation (0.364063)*/,
    -1,-13, 1,9/*mean (0.249325), correlation (0.378139)*/,
    7,1, 8,-6/*mean (0.000652272), correlation (0.411682)*/,
    1,-1, 3,12/*mean (0.00248538), correlation (0.392988)*/,
    9,1, 12,6/*mean (0.0206815), correlation (0.386106)*/,
    -1,-9, -1,3/*mean (0.0364485), correlation (0.410752)*/,
    -13,-13, -10,5/*mean (0.0376068), correlation (0.398374)*/,
    7,7, 10,12/*mean (0.0424202), correlation (0.405663)*/,
    12,-5, 12,9/*mean (0.0942645), correlation (0.410422)*/,
    6,3, 7,11/*mean (0.1074), correlation (0.413224)*/,
    5,-13, 6,10/*mean (0.109256), correlation (0.408646)*/,
    2,-12, 2,3/*mean (0.131691), correlation (0.416076)*/,
    3,8, 4,-6/*mean (0.165081), correlation (0.417569)*/,
    2,6, 12,-13/*mean (0.171874), correlation (0.408471)*/,
    9,-12, 10,3/*mean (0.175146), correlation (0.41296)*/,
    -8,4, -7,9/*mean (0.183682), correlation (0.402956)*/,
    -11,12, -4,-6/*mean (0.184672), correlation (0.416125)*/,
    1,12, 2,-8/*mean (0.191487), correlation (0.386696)*/,
    6,-9, 7,-4/*mean (0.192668), correlation (0.394771)*/,
    2,3, 3,-2/*mean (0.200157), correlation (0.408303)*/,
    6,3, 11,0/*mean (0.204588), correlation (0.411762)*/,
    3,-3, 8,-8/*mean (0.205904), correlation (0.416294)*/,
    7,8, 9,3/*mean (0.213237), correlation (0.409306)*/,
    -11,-5, -6,-4/*mean (0.243444), correlation (0.395069)*/,
    -10,11, -5,10/*mean (0.247672), correlation (0.413392)*/,
    -5,-8, -3,12/*mean (0.24774), correlation (0.411416)*/,
    -10,5, -9,0/*mean (0.00213675), correlation (0.454003)*/,
    8,-1, 12,-6/*mean (0.0293635), correlation (0.455368)*/,
    4,-6, 6,-11/*mean (0.0404971), correlation (0.457393)*/,
    -10,12, -8,7/*mean (0.0481107), correlation (0.448364)*/,
    4,-2, 6,7/*mean (0.050641), correlation (0.455019)*/,
    -2,0, -2,12/*mean (0.0525978), correlation (0.44338)*/,
    -5,-8, -5,2/*mean (0.0629667), correlation (0.457096)*/,
    7,-6, 10,12/*mean (0.0653846), correlation (0.445623)*/,
    -9,-13, -8,-8/*mean (0.0858749), correlation (0.449789)*/,
    -5,-13, -5,-2/*mean (0.122402), correlation (0.450201)*/,
    8,-8, 9,-13/*mean (0.125416), correlation (0.453224)*/,
    -9,-11, -9,0/*mean (0.130128), correlation (0.458724)*/,
    1,-8, 1,-2/*mean (0.132467), correlation (0.440133)*/,
    7,-4, 9,1/*mean (0.132692), correlation (0.454)*/,
    -2,1, -1,-4/*mean (0.135695), correlation (0.455739)*/,
    11,-6, 12,-11/*mean (0.142904), correlation (0.446114)*/,
    -12,-9, -6,4/*mean (0.146165), correlation (0.451473)*/,
    3,7, 7,12/*mean (0.147627), correlation (0.456643)*/,
    5,5, 10,8/*mean (0.152901), correlation (0.455036)*/,
    0,-4, 2,8/*mean (0.167083), correlation (0.459315)*/,
    -9,12, -5,-13/*mean (0.173234), correlation (0.454706)*/,
    0,7, 2,12/*mean (0.18312), correlation (0.433855)*/,
    -1,2, 1,7/*mean (0.185504), correlation (0.443838)*/,
    5,11, 7,-9/*mean (0.185706), correlation (0.451123)*/,
    3,5, 6,-8/*mean (0.188968), correlation (0.455808)*/,
    -13,-4, -8,9/*mean (0.191667), correlation (0.459128)*/,
    -5,9, -3,-3/*mean (0.193196), correlation (0.458364)*/,
    -4,-7, -3,-12/*mean (0.196536), correlation (0.455782)*/,
    6,5, 8,0/*mean (0.1972), correlation (0.450481)*/,
    -7,6, -6,12/*mean (0.199438), correlation (0.458156)*/,
    -13,6, -5,-2/*mean (0.211224), correlation (0.449548)*/,
    1,-10, 3,10/*mean (0.211718), correlation (0.440606)*/,
    4,1, 8,-4/*mean (0.213034), correlation (0.443177)*/,
    -2,-2, 2,-13/*mean (0.234334), correlation (0.455304)*/,
    2,-12, 12,12/*mean (0.235684), correlation (0.443436)*/,
    -2,-13, 0,-6/*mean (0.237674), correlation (0.452525)*/,
    4,1, 9,3/*mean (0.23962), correlation (0.444824)*/,
    -6,-10, -3,-5/*mean (0.248459), correlation (0.439621)*/,
    -3,-13, -1,1/*mean (0.249505), correlation (0.456666)*/,
    7,5, 12,-11/*mean (0.00119208), correlation (0.495466)*/,
    4,-2, 5,-7/*mean (0.00372245), correlation (0.484214)*/,
    -13,9, -9,-5/*mean (0.00741116), correlation (0.499854)*/,
    7,1, 8,6/*mean (0.0208952), correlation (0.499773)*/,
    7,-8, 7,6/*mean (0.0220085), correlation (0.501609)*/,
    -7,-4, -7,1/*mean (0.0233806), correlation (0.496568)*/,
    -8,11, -7,-8/*mean (0.0236505), correlation (0.489719)*/,
    -13,6, -12,-8/*mean (0.0268781), correlation (0.503487)*/,
    2,4, 3,9/*mean (0.0323324), correlation (0.501938)*/,
    10,-5, 12,3/*mean (0.0399235), correlation (0.494029)*/,
    -6,-5, -6,7/*mean (0.0420153), correlation (0.486579)*/,
    8,-3, 9,-8/*mean (0.0548021), correlation (0.484237)*/,
    2,-12, 2,8/*mean (0.0616622), correlation (0.496642)*/,
    -11,-2, -10,3/*mean (0.0627755), correlation (0.498563)*/,
    -12,-13, -7,-9/*mean (0.0829622), correlation (0.495491)*/,
    -11,0, -10,-5/*mean (0.0843342), correlation (0.487146)*/,
    5,-3, 11,8/*mean (0.0929937), correlation (0.502315)*/,
    -2,-13, -1,12/*mean (0.113327), correlation (0.48941)*/,
    -1,-8, 0,9/*mean (0.132119), correlation (0.467268)*/,
    -13,-11, -12,-5/*mean (0.136269), correlation (0.498771)*/,
    -10,-2, -10,11/*mean (0.142173), correlation (0.498714)*/,
    -3,9, -2,-13/*mean (0.144141), correlation (0.491973)*/,
    2,-3, 3,2/*mean (0.14892), correlation (0.500782)*/,
    -9,-13, -4,0/*mean (0.150371), correlation (0.498211)*/,
    -4,6, -3,-10/*mean (0.152159), correlation (0.495547)*/,
    -4,12, -2,-7/*mean (0.156152), correlation (0.496925)*/,
    -6,-11, -4,9/*mean (0.15749), correlation (0.499222)*/,
    6,-3, 6,11/*mean (0.159211), correlation (0.503821)*/,
    -13,11, -5,5/*mean (0.162427), correlation (0.501907)*/,
    11,11, 12,6/*mean (0.16652), correlation (0.497632)*/,
    7,-5, 12,-2/*mean (0.169141), correlation (0.484474)*/,
    -1,12, 0,7/*mean (0.169456), correlation (0.495339)*/,
    -4,-8, -3,-2/*mean (0.171457), correlation (0.487251)*/,
    -7,1, -6,7/*mean (0.175), correlation (0.500024)*/,
    -13,-12, -8,-13/*mean (0.175866), correlation (0.497523)*/,
    -7,-2, -6,-8/*mean (0.178273), correlation (0.501854)*/,
    -8,5, -6,-9/*mean (0.181107), correlation (0.494888)*/,
    -5,-1, -4,5/*mean (0.190227), correlation (0.482557)*/,
    -13,7, -8,10/*mean (0.196739), correlation (0.496503)*/,
    1,5, 5,-13/*mean (0.19973), correlation (0.499759)*/,
    1,0, 10,-13/*mean (0.204465), correlation (0.49873)*/,
    9,12, 10,-1/*mean (0.209334), correlation (0.49063)*/,
    5,-8, 10,-9/*mean (0.211134), correlation (0.503011)*/,
    -1,11, 1,-13/*mean (0.212), correlation (0.499414)*/,
    -9,-3, -6,2/*mean (0.212168), correlation (0.480739)*/,
    -1,-10, 1,12/*mean (0.212731), correlation (0.502523)*/,
    -13,1, -8,-10/*mean (0.21327), correlation (0.489786)*/,
    8,-11, 10,-6/*mean (0.214159), correlation (0.488246)*/,
    2,-13, 3,-6/*mean (0.216993), correlation (0.50287)*/,
    7,-13, 12,-9/*mean (0.223639), correlation (0.470502)*/,
    -10,-10, -5,-7/*mean (0.224089), correlation (0.500852)*/,
    -10,-8, -8,-13/*mean (0.228666), correlation (0.502629)*/,
    4,-6, 8,5/*mean (0.22906), correlation (0.498305)*/,
    3,12, 8,-13/*mean (0.233378), correlation (0.503825)*/,
    -4,2, -3,-3/*mean (0.234323), correlation (0.476692)*/,
    5,-13, 10,-12/*mean (0.236392), correlation (0.475462)*/,
    4,-13, 5,-1/*mean (0.236842), correlation (0.504132)*/,
    -9,9, -4,3/*mean (0.236977), correlation (0.497739)*/,
    0,3, 3,-9/*mean (0.24314), correlation (0.499398)*/,
    -12,1, -6,1/*mean (0.243297), correlation (0.489447)*/,
    3,2, 4,-8/*mean (0.00155196), correlation (0.553496)*/,
    -10,-10, -10,9/*mean (0.00239541), correlation (0.54297)*/,
    8,-13, 12,12/*mean (0.0034413), correlation (0.544361)*/,
    -8,-12, -6,-5/*mean (0.003565), correlation (0.551225)*/,
    2,2, 3,7/*mean (0.00835583), correlation (0.55285)*/,
    10,6, 11,-8/*mean (0.00885065), correlation (0.540913)*/,
    6,8, 8,-12/*mean (0.0101552), correlation (0.551085)*/,
    -7,10, -6,5/*mean (0.0102227), correlation (0.533635)*/,
    -3,-9, -3,9/*mean (0.0110211), correlation (0.543121)*/,
    -1,-13, -1,5/*mean (0.0113473), correlation (0.550173)*/,
    -3,-7, -3,4/*mean (0.0140913), correlation (0.554774)*/,
    -8,-2, -8,3/*mean (0.017049), correlation (0.55461)*/,
    4,2, 12,12/*mean (0.01778), correlation (0.546921)*/,
    2,-5, 3,11/*mean (0.0224022), correlation (0.549667)*/,
    6,-9, 11,-13/*mean (0.029161), correlation (0.546295)*/,
    3,-1, 7,12/*mean (0.0303081), correlation (0.548599)*/,
    11,-1, 12,4/*mean (0.0355151), correlation (0.523943)*/,
    -3,0, -3,6/*mean (0.0417904), correlation (0.543395)*/,
    4,-11, 4,12/*mean (0.0487292), correlation (0.542818)*/,
    2,-4, 2,1/*mean (0.0575124), correlation (0.554888)*/,
    -10,-6, -8,1/*mean (0.0594242), correlation (0.544026)*/,
    -13,7, -11,1/*mean (0.0597391), correlation (0.550524)*/,
    -13,12, -11,-13/*mean (0.0608974), correlation (0.55383)*/,
    6,0, 11,-13/*mean (0.065126), correlation (0.552006)*/,
    0,-1, 1,4/*mean (0.074224), correlation (0.546372)*/,
    -13,3, -9,-2/*mean (0.0808592), correlation (0.554875)*/,
    -9,8, -6,-3/*mean (0.0883378), correlation (0.551178)*/,
    -13,-6, -8,-2/*mean (0.0901035), correlation (0.548446)*/,
    5,-9, 8,10/*mean (0.0949843), correlation (0.554694)*/,
    2,7, 3,-9/*mean (0.0994152), correlation (0.550979)*/,
    -1,-6, -1,-1/*mean (0.10045), correlation (0.552714)*/,
    9,5, 11,-2/*mean (0.100686), correlation (0.552594)*/,
    11,-3, 12,-8/*mean (0.101091), correlation (0.532394)*/,
    3,0, 3,5/*mean (0.101147), correlation (0.525576)*/,
    -1,4, 0,10/*mean (0.105263), correlation (0.531498)*/,
    3,-6, 4,5/*mean (0.110785), correlation (0.540491)*/,
    -13,0, -10,5/*mean (0.112798), correlation (0.536582)*/,
    5,8, 12,11/*mean (0.114181), correlation (0.555793)*/,
    8,9, 9,-6/*mean (0.117431), correlation (0.553763)*/,
    7,-4, 8,-12/*mean (0.118522), correlation (0.553452)*/,
    -10,4, -10,9/*mean (0.12094), correlation (0.554785)*/,
    7,3, 12,4/*mean (0.122582), correlation (0.555825)*/,
    9,-7, 10,-2/*mean (0.124978), correlation (0.549846)*/,
    7,0, 12,-2/*mean (0.127002), correlation (0.537452)*/,
    -1,-6, 0,-11/*mean (0.127148), correlation (0.547401)*/
};


ORBextractor::ORBextractor(int _nfeatures, float _scaleFactor, int _nlevels, int _edgeThreshold,
                           int _firstlevel, int _scoreType, int _patchSize, int _fastThreshold) :
    nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels), edgeThreshold(_edgeThreshold),
    firstLevel(_firstlevel), scoreType(_scoreType), patchSize(_patchSize), fastThreshold(_fastThreshold) { }



void ORBextractor::computeKeypointsOld(const cv::Mat imagePyramid,
                                       const cv::Mat maskPyramid,
                                       const std::vector<cv::Rect> layerInfo,
                                       const std::vector<float> layerScale,
                                       std::vector<cv::KeyPoint> &allKeypoints,
                                       int nfeatures, double scaleFactor,
                                       int edgeThreshod, int patchSize,
                                       int scoreType, int fastThreshold)
{
    int i, nkeypoints, level, nlevels = (int)layerInfo.size();
    std::vector<int> nfeaturesPerLevel(nlevels);

    float factor = (float)(1.0/scaleFactor);
    float ndesiredFeaturesPerScale = nfeatures*(1- factor)/(1 - (float)std::pow((double)factor, (double)nlevels));

    int sumFeatures = 0;
    for(level = 0; level < nlevels - 1; level++)
    {
        nfeaturesPerLevel[level] = cvRound(ndesiredFeaturesPerScale);
        sumFeatures +=nfeaturesPerLevel[level];
        ndesiredFeaturesPerScale *= factor;
    }
    nfeaturesPerLevel[nlevels - 1] = std::max(nfeatures - sumFeatures, 0);

    int halfPatchSize = patchSize/2;
    std::vector<int> umax(halfPatchSize + 2);

    int v, v0, vmax = cvFloor(halfPatchSize*std::sqrt(2.f)/2 + 1);
    int vmin = cvCeil(halfPatchSize*std::sqrt(2.f)/2);

    for(v = 0; v <= vmax; v++)
    {
        umax[v] = cvRound(std::sqrt((double)halfPatchSize*halfPatchSize - v*v));
    }

    for(v = halfPatchSize, v0 = 0; v >= vmin; --v)
    {
        while (umax[v0] == umax[v0+1]) {
            ++v0;
        }
        umax[v] = v0;
        ++v0;
    }

    allKeypoints.clear();
    std::vector<cv::KeyPoint> keypoints;
    std::vector<int> counters(nlevels);
    keypoints.reserve(nfeaturesPerLevel[0]*2);

    for(level = 0; level < nlevels; level++)
    {
        int featuresNum = nfeaturesPerLevel[level];
        cv::Mat img = imagePyramid(layerInfo[level]);
        cv::Mat mask = maskPyramid.empty() ? cv::Mat() : maskPyramid(layerInfo[level]);

        {
            Ptr<cv::FastFeatureDetector> fd = cv::FastFeatureDetector::create(fastThreshold, true);
            fd->detect(img, keypoints, mask);
        }

        // remove keypoints very close to the border
        cv::KeyPointsFilter::runByImageBorder(keypoints, img.size(), edgeThreshod);
        cv::KeyPointsFilter::retainBest(keypoints, scoreType == cv::ORB::HARRIS_SCORE ? 2*featuresNum : featuresNum);

        nkeypoints = (int)keypoints.size();
        counters[level] = nkeypoints;

        float sf = layerScale[level];

        for(i = 0; i < nkeypoints; i++)
        {
            keypoints[i].octave = level;
            keypoints[i].size = patchSize*sf;
        }

        std::copy(keypoints.begin(), keypoints.end(), std::back_inserter(allKeypoints));
    }

    nkeypoints = (int)allKeypoints.size();

    if(nkeypoints == 0)
    {
        return;
    }

    cv::Mat responses;

    if(scoreType == cv::ORB::HARRIS_SCORE)
    {
        HarrisResponsesOld(imagePyramid, layerInfo, allKeypoints, 7, HARRIS_K);
        std::vector<cv::KeyPoint> newAllKeypoints;
        newAllKeypoints.reserve(nfeaturesPerLevel[0]*nlevels);

        int offset = 0;
        for(level = 0; level < nlevels; level++)
        {
            int featuresNum = nfeaturesPerLevel[level];
            nkeypoints = counters[level];
            keypoints.resize(nkeypoints);
            std::copy(allKeypoints.begin() + offset,
                      allKeypoints.begin() + offset + nkeypoints,
                      keypoints.begin());
            offset += nkeypoints;

            cv::KeyPointsFilter::retainBest(keypoints, featuresNum);
            std::copy(keypoints.begin(), keypoints.end(), std::back_inserter(newAllKeypoints));
        }
        std::swap(allKeypoints, newAllKeypoints);
    }
    nkeypoints = (int)allKeypoints.size();
    ICAnglesOld(imagePyramid, layerInfo, allKeypoints, umax, halfPatchSize);

    for(i = 0; i < nkeypoints; i++)
    {
        float scale = layerScale[allKeypoints[i].octave];
        allKeypoints[i].pt *= scale;
    }
}


void ORBextractor::detectAndCompute(cv::InputArray _image, cv::InputArray _mask,
                                    std::vector<cv::KeyPoint> &keypoints,
                                    cv::OutputArray _descriptors)
{
    CV_Assert(patchSize >= 2);

    int halfPatchSize = patchSize/2;
    int descPatchSize = cvCeil(halfPatchSize*1.4142);
    int border = std::max(edgeThreshold, std::max(descPatchSize, HARRIS_BLOCK_SIZE/2)) + 1;

    cv::Mat imageGray = _image.getMat().clone();
    cv::Mat imageBRG = _image.getMat();
    cv::Mat mask = _mask.getMat();
    if(_image.type() != CV_8UC1)
        cvtColor(_image, imageGray, COLOR_BGR2GRAY);
    if(_image.type() == CV_8UC1)
        cvtColor(_image, imageBRG, COLOR_GRAY2BGR);

    int i, level, nLevels = this->nlevels, nkeypoints = (int)keypoints.size();
    std::vector<cv::Rect> layerInfo(nLevels);
    std::vector<int> layerOfs(nLevels);
    std::vector<float> layerScale(nLevels);
    cv::Mat imageGrayPyramid, imageBRGPyramid, maskPyramid;

    int level_dy = imageGray.rows + border*2;
    cv::Point level_ofs(0,0);
    Size bufSize((cvRound(imageGray.cols/getScale(0, firstLevel, scaleFactor)) + border*2 + 15)& -16, 0);

    for(level = 0; level < nLevels; level++)
    {
        float scale = getScale(level, firstLevel, scaleFactor);
        layerScale[level] = scale;
        Size sz(cvRound(imageGray.cols/scale), cvRound(imageGray.rows/scale));
        Size wholeSize(sz.width + border*2, sz.height + border*2);

        if(level_ofs.x + wholeSize.width > bufSize.width)
        {
            level_ofs = cv::Point(0, level_ofs.y + level_dy);
            level_dy = wholeSize.height;
        }

        cv::Rect linfo(level_ofs.x + border, level_ofs.y + border, sz.width, sz.height);
        layerInfo[level] = linfo;
        layerOfs[level] = linfo.y*bufSize.width + linfo.x;
        level_ofs.x += wholeSize.width;
    }
    bufSize.height = level_ofs.y + level_dy;

    imageBRGPyramid.create(bufSize, CV_8UC3);
    imageGrayPyramid.create(bufSize, CV_8U);

    if(!mask.empty())
        maskPyramid.create(bufSize, CV_8U);

    cv::Mat prevImgBRG = imageBRG, prevImgGray = imageGray, prevMask = mask;

    // pre_compute the scale pyramids

    for(level = 0; level < nLevels; level++)
    {
        cv::Rect linfo = layerInfo[level];
        Size sz(linfo.width, linfo.height);
        Size wholeSize(sz.width + border*2, sz.height + border*2);
        cv::Rect wholeLinfo = cv::Rect(linfo.x - border, linfo.y - border, wholeSize.width, wholeSize.height);
        cv::Mat extImgBRG = imageBRGPyramid(wholeLinfo);
        cv::Mat extImgGray = imageGrayPyramid(wholeLinfo);
        cv::Mat extMask;
        cv::Mat currImgBRG = extImgBRG(cv::Rect(border, border, sz.width, sz.height));
        cv::Mat currImgGray = extImgGray(cv::Rect(border, border, sz.width, sz.height));
        cv::Mat currMask;

        if(!mask.empty())
        {
            extMask = maskPyramid(wholeLinfo);
            currMask = extMask(cv::Rect(border, border, sz.width, sz.height));
        }

        if(level != firstLevel)
        {
            resize(prevImgBRG, currImgBRG, sz, 0, 0, INTER_LINEAR_EXACT);
            resize(prevImgGray, currImgGray, sz, 0, 0, INTER_LINEAR_EXACT);
            if(!mask.empty())
            {
                resize(prevMask, currMask, sz, 0, 0, INTER_LINEAR_EXACT);
                if(level > firstLevel)
                {
                    threshold(currMask, currMask, 254, 0, THRESH_TOZERO);
                }
            }

            copyMakeBorder(currImgBRG, extImgBRG, border, border, border, border, BORDER_REFLECT_101 + BORDER_ISOLATED);
            copyMakeBorder(currImgGray, extImgGray, border, border, border, border, BORDER_REFLECT_101 + BORDER_ISOLATED);

            if(!mask.empty())
                copyMakeBorder(currMask, extMask, border, border, border, border, BORDER_CONSTANT + BORDER_ISOLATED);
        }
        else {
            copyMakeBorder(imageBRG, extImgBRG, border, border, border, border, BORDER_REFLECT_101);
            copyMakeBorder(imageGray, extImgGray, border, border, border, border, BORDER_REFLECT_101);
            if(!mask.empty())
                copyMakeBorder(mask, extMask, border, border, border, border, BORDER_CONSTANT + BORDER_ISOLATED);
        }

        if (level > firstLevel)
        {
            prevImgBRG = currImgBRG;
            prevImgGray = currImgGray;
            prevMask = currMask;
        }
    }

    computeKeypointsOld(imageGrayPyramid, maskPyramid,
                        layerInfo, layerScale, keypoints,
                        nfeatures, scaleFactor, edgeThreshold, patchSize, scoreType, fastThreshold);

    nkeypoints = (int)keypoints.size();

    if(nkeypoints == 0)
    {
        _descriptors.release();
        return;
    }

    int dsize = 88;

    _descriptors.create(nkeypoints, dsize, CV_8U);

    std::vector<cv::Point> pattern;
    std::vector<cv::Point> patternCN;

    const int npoints = 512;
    cv::Point patternbuf[npoints];
    const cv::Point* pattern0 = (const cv::Point*)bit_pattern_31_;

    if(patchSize != 31)
    {
        pattern0 = patternbuf;
        makeRandomPattern(patchSize, patternbuf, npoints);
    } 

    std::copy(pattern0, pattern0 + npoints, std::back_inserter(pattern));

    std::vector<cv::Mat> im_split;
    cv::split(imageBRGPyramid, im_split);

    cv::Mat descriptors = _descriptors.getMat();

    ParallelComputeCNDescriptors parallelcomputeCNDescriptors(im_split, imageGrayPyramid, layerInfo,
                                                              layerScale, keypoints, descriptors, pattern,
                                                              dsize, m_objectColor);

    cv::parallel_for_(Range(0, static_cast<int>(keypoints.size())), parallelcomputeCNDescriptors);  

}

}  // end objsearch

}  // end ip





