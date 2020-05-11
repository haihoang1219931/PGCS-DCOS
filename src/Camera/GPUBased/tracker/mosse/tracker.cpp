#include "tracker.h"

Tracker::Tracker()
{
    //Var initialization
    this->tag = "";
    this->state_ = TRACK_INVISION;
    this->PSR_mask = 11;
    this->PSR_ratio[0] = 2;
    this->PSR_ratio[1] = 11;

    this->_learning = 0.125;

    this->_im_size.width = 0;
    this->_im_size.height = 0;

    this->_init = false;
    this->_eps = true;

}

Tracker::~Tracker()
{


}

string Tracker::getTag(){
    return this->tag;
}

void Tracker::setTag(string tag){
    this->tag = tag;
}

void Tracker::initTrack(const cv::Mat &input_image, cv::Point topLeft, cv::Point bottomRight)
{//Init tracker from user selection

    this->initTrack(input_image,
        cv::Rect(topLeft.x, topLeft.y,
        bottomRight.x - topLeft.x,
        bottomRight.y - topLeft.y));

}

void Tracker::initTrack(const cv::Mat &input_image, cv::Rect input_rect)
{//Init tracker from user selection

    if (this->_init) return;

    if (input_image.empty())
    {
        cout << "Error while selecting target !" << endl;
        return;
    }

    this->_im_size.width = input_image.cols;
    this->_im_size.height = input_image.rows;

    input_image(input_rect).copyTo(prev_img.real_image);  //Store frame as previous frame

    this->prev_img.cols = this->prev_img.real_image.cols;
    this->prev_img.rows = this->prev_img.real_image.rows;

    ComputeDFT(this->prev_img, true); //Compute Direct Fourier Transform
    SetRoi(input_rect);               //User selection is object current pos

    InitFilter();                     //Init filter from ROI (ROI + affine transf)

    this->_init = true;
}

void Tracker::SetRoi(cv::Rect input_ROI)
{//Init ROI position and center
printf("Begin %s\r\n",__func__);
    this->current_ROI.ROI = input_ROI;
    this->current_ROI.ROI_center.x = round(input_ROI.width / 2);
    this->current_ROI.ROI_center.y = round(input_ROI.height / 2);
printf("End %s\r\n",__func__);
}

void Tracker::performTrack(const cv::Mat &input_image)
{
    //Perform tracking over current frame
    //cv::imshow("input_image Tracker", input_image);
    if (!this->_init) return;

    if (this->_filter.empty())
    {
        cout << "Error, must initialize tracker first ! " << endl;
        return;
    }

    cv::Point new_loc;
    input_image(this->current_ROI.ROI).copyTo(this->current_img.real_image);    //Crop new search area
    ComputeDFT(this->current_img, true);                                        //Compute Direct Fourier Transform
    new_loc = PerformTrack();                                                   //Perform tracking

    if (new_loc.x >= 0 && new_loc.y >= 0)                                            //If PSR > ratio then update
    {
        this->state_ = TRACK_INVISION;
        //printf("Found at[%dx%d]\r\n", new_loc.x, new_loc.y);
        Update(new_loc);                                                        //Update Tracker
    }
    else this->state_ = TRACK_OCCLUDED;
}
int Tracker::Get_State(){
    return this->state_;
}

void Tracker::ComputeDFT(image_track &input_image, bool preprocess)
{//Compute Direct Fourier Transform on input image, with or without a pre-processing

    cv::Mat res = this->ComputeDFT(input_image.real_image, preprocess);

    input_image.image_spectrum = res;
    input_image.opti_dft_comp_rows = res.rows;
    input_image.opti_dft_comp_cols = res.cols;
}

cv::Mat Tracker::ComputeDFT(const cv::Mat &input_image, bool preprocess)
{//Compute Direct Fourier Transform on input image, with or without a pre-processing

    cv::Mat gray_padded, complexI;

    int x = input_image.rows;
    int y = input_image.cols;

    //Get optimal dft image size
    int i = cv::getOptimalDFTSize(x);
    int j = cv::getOptimalDFTSize(y);

    //Get optimal dct image size
    //int i = 2*getOptimalDFTSize((x+1)/2);
    //int j = 2*getOptimalDFTSize((y+1)/2);

    //Zero pad input image up to optimal dft size
    cv::copyMakeBorder(input_image, gray_padded, 0, i - x, 0, j - y, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    input_image.copyTo(gray_padded);

    //If input image is RGB, convert to gray scale
    if (gray_padded.channels() > 1) cvtColor(gray_padded, gray_padded, CV_RGB2GRAY);
    gray_padded.convertTo(gray_padded, CV_32F);

    if (preprocess)     //Apply pre-processing to input image
    {

        // DCT Stuff
        //        cv::dct(gray_padded, gray_padded, CV_DXT_FORWARD);
        //        for (int i=0; i<gray_padded.rows; i+=1)
        //        {
        //            for (int j=0; j<gray_padded.cols; j+=1)
        //            {
        //                gray_padded.at<float>(i,j) *= (1 / (1 + (exp(-(i * gray_padded.cols + j)))));
        //            }
        //        }
        //        gray_padded.at<float>(0,0) = 0;
        //        cv::dct(gray_padded, gray_padded, CV_DXT_INVERSE);


        cv::normalize(gray_padded, gray_padded, 0.0, 1.0, cv::NORM_MINMAX);

        gray_padded += cv::Scalar::all(1);
        cv::log(gray_padded, gray_padded);

        cv::Scalar mean, stddev;
        cv::meanStdDev(gray_padded, mean, stddev);
        gray_padded -= mean.val[0];

        cv::Mat tmp;
        cv::multiply(gray_padded, gray_padded, tmp);
        cv::Scalar sum_ = cv::sum(tmp);
        gray_padded /= sum_.val[0];

        //Apply Hanning window to reduce image boundaries effect
        if (this->_HanningWin.empty() || gray_padded.size() != this->_HanningWin.size())
        {
            cv::Mat hanningWin_;
            cv::createHanningWindow(hanningWin_, gray_padded.size(), CV_32F);
            hanningWin_.copyTo(this->_HanningWin);
        }

        cv::multiply(gray_padded, this->_HanningWin, gray_padded);

    }

    dft(gray_padded, complexI, cv::DFT_COMPLEX_OUTPUT);    //Compute Direct Fourier Transform

    //Crop the spectrum, if it has an odd number of rows or columns
    printf("Begin %s Crop the spectrum,\r\n",__func__);
    cv::Rect cropRect = cv::Rect(0, 0, complexI.cols & -2, complexI.rows & -2);
    if(cropRect.width > complexI.cols){
        cropRect.width = complexI.cols;
    }
    if(cropRect.height > complexI.rows){
        cropRect.height = complexI.rows;
    }
    complexI = complexI(cropRect);
    printf("End %s Crop the spectrum,\r\n",__func__);
    return complexI;
}

cv::Point Tracker::PerformTrack()
{
    cv::Mat mat_correlation, idft_correlation;
    float PSR_val;
    cv::Point maxLoc;

    //Element-wise matrice multiplication, second arg is complex conjugate H*
    mulSpectrums(this->current_img.image_spectrum, this->_filter, mat_correlation, 0, false);

    //Inverse DFT real output
    dft(mat_correlation, idft_correlation, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);


    // ***************************************************************************************************
    /* ---------------------------------------------------- */
    // Showing the images in the GUI

    // Correlation image
    cv::Mat corrImg;
    normalize(idft_correlation, idft_correlation, 0.0, 255.0, cv::NORM_MINMAX);
    //corrImg = idft_correlation.clone();
    //resize(idft_correlation, corrImg, cv::Size(136, 79)); // not taking into account a resized window!

    // Filter image
    cv::Mat filtImg;
    //imshow("thisfilter", this->_filter);
    dft(this->_filter, filtImg, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
    //imshow("filter", filtImg);
    // Shifting the image!!
    int cx = filtImg.cols / 2;
    int cy = filtImg.rows / 2;
    cv::Mat q0(filtImg, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    cv::Mat q1(filtImg, cv::Rect(cx, 0, cx, cy));  // Top-Right
    cv::Mat q2(filtImg, cv::Rect(0, cy, cx, cy));  // Bottom-Left
    cv::Mat q3(filtImg, cv::Rect(cx, cy, cx, cy)); // Bottom-Right



    cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);

    normalize(filtImg, filtImg, 0.0, 255.0, cv::NORM_MINMAX);
    //resize(filtImg, filtImg, cv::Size(136, 79)); // not taking into account a resized window!
    flip(filtImg, filtImg, 0);

    // ***************************************************************************************************


    PSR_val = ComputePSR(idft_correlation);  //Compute PSR

    //cout << "PSR = " << PSR_val << endl;


    if (PSR_val >= PSR_ratio[1])    //Get new pos if object detected
    {

        minMaxLoc(idft_correlation, NULL, NULL, NULL, &maxLoc);
        //imshow("idft_correlation", idft_correlation);
        // ******* Trying to keep a nice and clean filter output image ***********************
        cv::Mat new_output = cv::Mat::zeros(mat_correlation.size(), CV_32F);

        MaskDesiredG(new_output, maxLoc.x, maxLoc.y);

        new_output = ComputeDFT(new_output, false);
        // ***********************************************************************************

        new_output.copyTo(this->current_img.filter_output);
        //imshow("new_output",new_output);

    }
    else if (PSR_val > PSR_ratio[0]){ //Return -1 coordinates if object occluded
        maxLoc.x = -2;
        maxLoc.y = -2;
    }
    else{                           //Return -2 coordinates if object lost
        maxLoc.x = -10;
        maxLoc.y = -10;
    }

    cv::Mat output;
    //cvtColor(this->current_img.filter_output, output, CV_BGR2GRAY);
    //imshow("output", output);
    return maxLoc;
}

float Tracker::ComputePSR(const cv::Mat &correlation_mat)
{//Compute Peak-to-Sidelobe Ratio

    double max_val = 0;
    cv::Point max_loc;
    cv::Mat PSR_mask = cv::Mat::ones(correlation_mat.rows, correlation_mat.cols, CV_8U);
    cv::Scalar mean, stddev;

    minMaxLoc(correlation_mat, NULL, &max_val, NULL, &max_loc);     //Get location of max arg

    //Define PSR mask
    int win_size = floor(this->PSR_mask / 2);
    cv::Rect mini_roi = cv::Rect(std::max(max_loc.x - win_size, 0), std::max(max_loc.y - win_size, 0), this->PSR_mask, this->PSR_mask);

    //Handle image boundaries
    if ((mini_roi.x + mini_roi.width) > PSR_mask.cols)
    {
        mini_roi.width = PSR_mask.cols - mini_roi.x;
    }
    if ((mini_roi.y + mini_roi.height) > PSR_mask.rows)
    {
        mini_roi.height = PSR_mask.rows - mini_roi.y;
    }

    cv::Mat temp = PSR_mask(mini_roi);
    temp *= 0;
    meanStdDev(correlation_mat, mean, stddev, PSR_mask);   //Compute matrix mean and std

    return (max_val - mean.val[0]) / stddev.val[0];     //Compute PSR
}

void Tracker::UpdateRoi(cv::Point new_center, bool scale_rot)
{//Update ROI position

    int diff_x, diff_y;

    //Current ROI pos is previous ROI pos
    this->prev_ROI = this->current_ROI;

    //    if (scale_rot)
    //    {
    //        //ComputeOrientationScale();
    //    }

    //Define new ROI position
    this->current_ROI.ROI_center = new_center;

    new_center.x += this->prev_ROI.ROI.x;
    new_center.y += this->prev_ROI.ROI.y;

    //Handle image boundarie
    diff_x = new_center.x - round(this->current_ROI.ROI.width / 2);
    diff_y = new_center.y - round(this->current_ROI.ROI.height / 2);

    if (diff_x < 0)
    {
        this->current_ROI.ROI.x = 0;
    }
    else if ((diff_x + this->current_ROI.ROI.width) >= this->_im_size.width)
    {
        this->current_ROI.ROI.x = this->_im_size.width - this->current_ROI.ROI.width - 1;
    }
    else{
        this->current_ROI.ROI.x = diff_x;
    }

    if (diff_y < 0)
    {
        this->current_ROI.ROI.y = 0;
    }
    else if ((diff_y + this->current_ROI.ROI.height) >= this->_im_size.height)
    {
        this->current_ROI.ROI.y = this->_im_size.height - this->current_ROI.ROI.height - 1;
    }
    else{
        this->current_ROI.ROI.y = diff_y;
    }

    this->current_ROI.ROI.width = this->prev_ROI.ROI.width;
    this->current_ROI.ROI.height = this->prev_ROI.ROI.height;
}

void Tracker::Update(cv::Point new_location)
{//Update Tracker

    UpdateFilter();                         //Update filter
    this->prev_img = this->current_img;     //Update frame
    UpdateRoi(new_location, false);          //Update ROI position
}

void Tracker::InitFilter()
{//Init filter from user selection

    cv::Mat affine_G, affine_image, temp_image_dft, temp_desi_G, filter;
    cv::Mat temp_FG, temp_FF, num, dem, eps;

    //Number of images to init filter
    int N = 8;

    //Create the the desired output - 2D Gaussian
    cv::Mat Mask_gauss = cv::Mat::zeros(this->prev_img.real_image.size(), CV_32F);
    MaskDesiredG(Mask_gauss, round(this->current_ROI.ROI.width / 2),
        round(this->current_ROI.ROI.height / 2));


    temp_FG = cv::Mat::zeros(this->prev_img.opti_dft_comp_rows, this->prev_img.opti_dft_comp_cols, this->prev_img.image_spectrum.type());
    temp_FF = cv::Mat::zeros(this->prev_img.opti_dft_comp_rows, this->prev_img.opti_dft_comp_cols, this->prev_img.image_spectrum.type());

    temp_image_dft = this->prev_img.image_spectrum;
    temp_desi_G = ComputeDFT(Mask_gauss, false);

    temp_desi_G.copyTo(this->prev_img.filter_output);

    mulSpectrums(temp_desi_G, temp_image_dft, num, 0, true);       //Element-wise spectrums multiplication G o F*
    temp_FG += num;

    mulSpectrums(temp_image_dft, temp_image_dft, dem, 0, true);     //Element-wise spectrums multiplication F o F*
    temp_FF += dem;

    if (_eps)
    {
        //Regularization parameter
        eps = createEps(dem);
        dem += eps;
        temp_FF += dem;
    }

    srand(time(NULL));

    for (int i = 0; i<(N - 1); i++)
    {//Create image dataset with input image affine transforms

        AffineTransform(Mask_gauss, this->prev_img.real_image, affine_G, affine_image);    //Input image and desired output affine transform

        temp_image_dft = ComputeDFT(affine_image, true);        //Affine image DFT
        temp_desi_G = ComputeDFT(affine_G, false);              //Affine output DFT

        mulSpectrums(temp_desi_G, temp_image_dft, num, 0, true);   //Element-wise spectrums multiplication G o F*
        temp_FG += num;

        mulSpectrums(temp_image_dft, temp_image_dft, dem, 0, true); //Element-wise spectrums multiplication F o F*

        if (_eps)
        {
            eps = createEps(dem);
            dem += eps;
        }

        temp_FF += dem;
    }

    dftDiv(temp_FG, temp_FF, filter);       //Element-wise spectrum Division

    filter.copyTo(this->_filter);           //Filter

    //filter = conj(this->_filter);
    //inverseAndSave(filter, "filter_inv_shift.jpg", true);
}

void Tracker::AffineTransform(const cv::Mat &input_image, const cv::Mat &input_image2, cv::Mat &aff_img, cv::Mat &aff_img2)
{//Apply same randomly defined affine transform to both input matrice

    if (input_image.size() != input_image2.size())
    {
        cout << "Error while computing affine transform !" << endl;
        return;
    }

    //output images
    aff_img = cv::Mat::zeros(input_image.rows, input_image.cols, input_image.type());
    aff_img2 = cv::Mat::zeros(input_image2.rows, input_image2.cols, input_image2.type());

    int cols = input_image.cols;
    int rows = input_image.rows;

    cv::Point2f input_pts[3];
    cv::Point2f output_pts[3];

    float pts0_r, pts0_c, pts1_r, pts1_c, pts2_r, pts2_c;

    cv::Mat affine_tr(2, 3, CV_32FC1);

    input_pts[0] = cv::Point2f(0, 0);
    input_pts[1] = cv::Point2f(cols - 1, 0);
    input_pts[2] = cv::Point2f(0, rows - 1);

    //Define affine transform 'intensity'
    pts0_r = rand() % 5; pts0_r /= 100;
    pts0_c = rand() % 5; pts0_c /= 100;

    pts1_r = rand() % 5; pts1_r /= 100;
    pts1_c = rand() % 5 + 95; pts1_c /= 100;

    pts2_r = rand() % 5 + 95; pts2_r /= 100;
    pts2_c = rand() % 5; pts2_c /= 100;

    output_pts[0] = cv::Point2f(cols*pts0_c, rows*pts0_r);
    output_pts[1] = cv::Point2f(cols*pts1_c, rows*pts1_r);
    output_pts[2] = cv::Point2f(cols*pts2_c, rows*pts2_r);

    affine_tr = getAffineTransform(input_pts, output_pts);        //Get transformation matrix

    warpAffine(input_image, aff_img, affine_tr, aff_img.size());  //Apply transformation matrix
    warpAffine(input_image2, aff_img2, affine_tr, aff_img2.size());
}

void Tracker::MaskDesiredG(cv::Mat &output, int u_x, int u_y, double sigma, bool norm_energy)
{//Create 2D Gaussian

    sigma *= sigma;

    //Fill input matrix as 2D Gaussian
    for (int i = 0; i<output.rows; i++)
    {
        for (int j = 0; j<output.cols; j++)
        {
            output.at<float>(i, j) = 255 * exp((-(i - u_y)*(i - u_y) / (2 * sigma)) +
                (-(j - u_x)*(j - u_x) / (2 * sigma)));
        }
    }

    if (norm_energy)    //If true, norm image energy so that it sum up to 1
    {
        cv::Scalar sum_;
        sum_ = sum(output);
        output /= sum_.val[0];
    }

}

void Tracker::UpdateFilter()
{//Update filter
    cv::Mat Ai, Bi, Ai_1, Bi_1, A, B, filter, eps, eps_1;

    cv::mulSpectrums(this->current_img.filter_output, this->current_img.image_spectrum, Ai, 0, true);      //Element-wise spectrums multiplication G o F*
    cv::mulSpectrums(this->prev_img.filter_output, this->prev_img.image_spectrum, Ai_1, 0, true);          //Element-wise spectrums multiplication G-1 o F-1*

    cv::mulSpectrums(this->current_img.image_spectrum, this->current_img.image_spectrum, Bi, 0, true);     //Element-wise spectrums multiplication F o F*
    cv::mulSpectrums(this->prev_img.image_spectrum, this->prev_img.image_spectrum, Bi_1, 0, true);         //Element-wise spectrums multiplication F-1 o F-1*

    if (_eps)
    {
        //Regularization parameter
        eps = createEps(Bi);
        Bi += eps;

        eps_1 = createEps(Bi_1);
        Bi_1 += eps;
    }


    // MOSSE update

    A = (((1.0 - _learning)*Ai) + ((_learning)*Ai_1));
    B = (((1.0 - _learning)*Bi) + ((_learning)*Bi_1));
    dftDiv(A, B, filter);
    filter.copyTo(this->_filter);


    // ASEF update

    //    dftDiv(Ai, Bi, A);
    //    filter = (A * (1.0 - _learning)) - (this->_filter * _learning);
    //    filter.copyTo(this->_filter);

}

void Tracker::inverseAndSave(const cv::Mat &img, const std::string &filename, const bool &shift)
{//Inverse DFT and save image

    cv::Mat img_i;

    cv::dft(img, img_i, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);

    if (shift)      //If true swap quadrants
    {
        int cx = img_i.cols / 2;
        int cy = img_i.rows / 2;
        cv::Mat q0(img_i, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
        cv::Mat q1(img_i, cv::Rect(cx, 0, cx, cy));  // Top-Right
        cv::Mat q2(img_i, cv::Rect(0, cy, cx, cy));  // Bottom-Left
        cv::Mat q3(img_i, cv::Rect(cx, cy, cx, cy)); // Bottom-Right
        cv::Mat tmp;                             // swap quadrants (Top-Left with Bottom-Right)
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);
        q1.copyTo(tmp);                      // swap quadrant (Top-Right with Bottom-Left)
        q2.copyTo(q1);
        tmp.copyTo(q2);
    }

    cv::normalize(img_i, img_i, 0.0, 255.0, cv::NORM_MINMAX);
    cv::imwrite(filename, img_i);
}

cv::Mat Tracker::conj(const cv::Mat &input_dft)
{//Compute complex conjugate

    assert(input_dft.channels() == 2);

    cv::Mat conj_dft;

    input_dft.copyTo(conj_dft);

    //Invert imaginary part sign
    for (int x = 0; x<input_dft.rows; x++)
    {
        for (int y = 0; y<input_dft.cols; y++)
        {
            conj_dft.at<cv::Vec2f>(x, y)[1] *= -1.0;
        }
    }

    return conj_dft;
}

void Tracker::dftDiv(const cv::Mat &dft_a, const cv::Mat &dft_b, cv::Mat &output_dft)
{//Compute complex divison

    assert(dft_a.size() == dft_b.size() && dft_a.type() == dft_b.type() &&
        dft_a.channels() == dft_b.channels() && dft_a.channels() == 2);

    cv::Mat out_temp = cv::Mat::zeros(dft_a.rows, dft_a.cols, dft_a.type());

    for (int x = 0; x<dft_a.rows; x++)
    {
        for (int y = 0; y<dft_a.cols; y++)
        {
            out_temp.at<cv::Vec2f>(x, y)[0] = ((dft_a.at<cv::Vec2f>(x, y)[0] * dft_b.at<cv::Vec2f>(x, y)[0]) +
                (dft_a.at<cv::Vec2f>(x, y)[1] * dft_b.at<cv::Vec2f>(x, y)[1])) /
                ((dft_b.at<cv::Vec2f>(x, y)[0] * dft_b.at<cv::Vec2f>(x, y)[0]) +
                (dft_b.at<cv::Vec2f>(x, y)[1] * dft_b.at<cv::Vec2f>(x, y)[1]));

            out_temp.at<cv::Vec2f>(x, y)[1] = ((dft_a.at<cv::Vec2f>(x, y)[1] * dft_b.at<cv::Vec2f>(x, y)[0]) -
                (dft_a.at<cv::Vec2f>(x, y)[0] * dft_b.at<cv::Vec2f>(x, y)[1])) /
                ((dft_b.at<cv::Vec2f>(x, y)[0] * dft_b.at<cv::Vec2f>(x, y)[0]) +
                (dft_b.at<cv::Vec2f>(x, y)[1] * dft_b.at<cv::Vec2f>(x, y)[1]));
        }
    }

    out_temp.copyTo(output_dft);
}

cv::Mat Tracker::createEps(const cv::Mat &input_, double std)
{//Compute regularization parameter for a given input matrix

    //Compute input matrix mean and std
    cv::Scalar mean, stddev;
    cv::meanStdDev(input_, mean, stddev);

    cv::Mat eps = cv::Mat::zeros(input_.size(), input_.type());

    //Fill output matrix so that white noise zero mean and std a fraction of input matrix mean value
    cv::randn(eps, 0, std*(mean.val[0]));

    //Set imaginary part of noise to all zeros
    for (int x = 0; x<eps.rows; x++)
    {
        for (int y = 0; y<eps.cols; y++)
        {
            eps.at<cv::Vec2f>(x, y)[1] = 0;
        }
    }

    eps.at<cv::Vec2f>(0, 0)[0] = 0;
    eps.at<cv::Vec2f>(input_.rows - 1, 0)[0] = 0;
    eps.at<cv::Vec2f>(0, input_.cols - 1)[0] = 0;
    eps.at<cv::Vec2f>(input_.rows - 1, input_.cols - 1)[0] = 0;

    return eps;
}

/* ------------------------------------ */

bool Tracker::isRunning(){
    return false;
}
cv::Rect Tracker::getPosition() const       //Get ROI position
{
    return current_ROI.ROI;
}

int Tracker::trackStatus() const
{
    return state_;
}

bool Tracker::isInitialized() const     //Verify tracker is init
{
    return _init;
}
void Tracker::resetTrack(){
    _init = false;
}

void Tracker::SetPSR_mask(int input)        //Set PSR var
{
    this->PSR_mask = input;
}

void Tracker::SetPSR_ratio_low(int input)
{
    this->PSR_ratio[0] = input;
}

void Tracker::SetPSR_ratio_high(int input)
{
    this->PSR_ratio[1] = input;
}

int Tracker::GetPSR_mask() const        //Get PSR var
{
    return this->PSR_mask;
}

int Tracker::GetPSR_ratio_low() const
{
    return this->PSR_ratio[0];
}

int Tracker::GetPSR_ratio_high() const
{
    return this->PSR_ratio[1];
}

cv::Mat Tracker::GetFilter() const      //Get filter
{
    return this->_filter;
}

float Tracker::Get_Learning() const     //Get/Set learning ratio
{
    return this->_learning;
}

void Tracker::Set_Learning(float val)
{
    this->_learning = val;
}
/* ------------------------------------ */
