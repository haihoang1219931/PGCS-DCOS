#ifndef TRACKER_H
#define TRACKER_H

#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <stdio.h>
#include <vector>
#include <time.h>


#define LOST 0
#define FOUND 1
#define OCCLUDED 2
using namespace std;

//Struct that contains image frame and its relatives
struct image_track{
    cv::Mat real_image;
    cv::Mat image_spectrum;
    cv::Mat filter_output;

    int cols;
    int rows;

    int opti_dft_comp_rows;
    int opti_dft_comp_cols;
};

//Struct ROI, tracked object
struct ROI_track{
    cv::Rect ROI;
    cv::Point ROI_center;
};


//Tracker class
class Tracker
{

public:
    Tracker();
    ~Tracker();

    image_track prev_img;                               //Previous frame
    image_track current_img;                            //Current frame

    ROI_track prev_ROI;                                 //Object location within previous frame
    ROI_track current_ROI;                              //Object location within current frame

    void InitTracker(const cv::Mat &, cv::Point, cv::Point);     //Init tracker from user selection
    void InitTracker(const cv::Mat &, cv::Rect);

    void Track(const cv::Mat&);                             //Perform tracking over current frame
    void resetTrack();
    bool isInitialized() const;                         //Verify tracker is init

    cv::Rect getPosition() const;                           //Get ROI position

    int getState() const;                               //Get the state { FOUND, OCCLUDED, LOST }

    cv::Mat GetFilter() const;                              //Get filter

    void SetPSR_mask(int);                              //Set PSR var
    void SetPSR_ratio_low(int);
    void SetPSR_ratio_high(int);

    int GetPSR_mask() const;                            //Get PSR var
    int GetPSR_ratio_low() const;
    int GetPSR_ratio_high() const;

    float Get_Learning() const;                         //Get/Set learning ratio
    void Set_Learning(float);
    int Get_State();

    string getTag();
    void setTag(string tag);
private:
    std::string tag;

    cv::Mat _filter;                                        //Tracker filter
    cv::Mat _HanningWin;                                    //Pre-processing Hanning-window
    bool _eps;
    bool _init;
    int PSR_mask;                                       //PSR var
    double PSR_ratio[2];
    double _learning;                                   //Learning ratio
    int state_;
    cv::Size _im_size;                                      //Full frame size

    void InitFilter(const cv::Mat&);                        //Init filter from user selection
    void InitFilter();

    //Apply same randomly defined affine transform to both input matrice
    void AffineTransform(const cv::Mat&, const cv::Mat&, cv::Mat&, cv::Mat&);

    //Compute Direct Fourier Transform on input image, with or without a pre-processing
    void ComputeDFT(image_track&, bool preprocess = false);
    cv::Mat ComputeDFT(const cv::Mat&, bool preprocess = false);

    //Init ROI position and center
    void SetRoi(cv::Rect);

    //Update ROI position
    void UpdateRoi(cv::Point, bool);

    //Perform tracking over current frame
    cv::Point PerformTrack();

    //Compute Peak-to-Sidelobe Ratio
    float ComputePSR(const cv::Mat &);

    //Update Tracker
    void Update(cv::Point);

    //Update filter
    void UpdateFilter();

    //Create 2D Gaussian
    void MaskDesiredG(cv::Mat &, int u_x, int u_y, double sigma = 2, bool norm_energy = true);

    //Inverse DFT and save image
    void inverseAndSave(const cv::Mat &, const std::string &, const bool &shift = false);

    //Compute complex conjugate
    cv::Mat conj(const cv::Mat&);

    //Compute complex divison
    void dftDiv(const cv::Mat&, const cv::Mat&, cv::Mat&);

    //Compute regularization parameter
    cv::Mat createEps(const cv::Mat&, double std = 0.00001);

};

#endif // TRACKER_H
