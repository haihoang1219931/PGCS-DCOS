#ifndef RECOGNITION_HPP
#define RECOGNITION_HPP

#include <opencv2/opencv.hpp>
#include <ostream>
#include <string>
#include <vector>

#include "Structures.hpp"

#define MIN_TEXT_WIDTH 50
#define MIN_TEXT_HEIGHT 25

struct Angles {
    float s_angle;
    int s_count;
};

struct Char {
    char c;
    float score;
};

enum State { WAITING, COUNTING };

class OCR
{
    public:
        OCR()
        {
            m_api->Init(NULL, "eng", tesseract::OEM_TESSERACT_LSTM_COMBINED);
            m_api->SetPageSegMode(tesseract::PSM_SINGLE_LINE);
            m_stop = false;
            m_c.c = 'W';
            m_c.score = 0;
            m_vc.push_back(m_c);

            for (int i = 0; i < 9; i++)
                m_codeTable.push_back(m_vc);

            m_codeTable0 = m_codeTable;
            m_counter = 0;
        }
        ~OCR()
        {
            m_api->End();
        }

    public:
        bool recognize(cv::Mat plate, const int type);

    private:
        void    insertPadding(cv::Mat input, cv::Mat &output, const int paddingSize, const int paddingValue);
        void    binarize(cv::Mat input, cv::Mat &output, const int windowSize);
        void    filter(const cv::Mat input, cv::Mat &output);
        void    filterBlobNoise(const cv::Mat input, cv::Mat &output, const int windowSize);
        bool    removeNoise(cv::Mat input, std::vector<std::vector<cv::Point>> contoursPoints, cv::Mat &output);
        int     getPatch(const cv::Mat &_grayImg, cv::Mat &_grayPatch, const cv::RotatedRect _targetBound);
        std::string returnCode(cv::Mat &input, std::vector<std::vector<cv::Point>> contoursPoints);
        float   findRightAngle(std::vector<float> angles);
        void contrastEnhance(cv::Mat &src, cv::Mat &dst, int dist = 10);
        cv::Mat deskewImage(cv::Mat image);

        std::string rawCorrect(std::string code);
        bool isDigit(const char c);
        bool isUpperCase(const char c);
        bool isLowerCase(const char c);

        void updateCodeTable(std::string code, std::vector<std::vector<Char>> &codeTable);
        std::string getResult(std::vector<std::vector<Char>> &codeTable, int plateType);

    public:
        tesseract::TessBaseAPI *m_api = new tesseract::TessBaseAPI();
        std::string m_finalResult;
        bool m_stop;
        std::string m_result;
        Char m_c;
        std::vector<Char> m_vc;
        std::vector<std::vector<Char>> m_codeTable;
        std::vector<std::vector<Char>> m_codeTable0;
        int m_counter;


        //        cv::Mat m_binImg;
        //        cv::Mat m_filImg;
        //        cv::Mat m_newPlate;
        //        cv::Mat m_plate;
};

#endif // RECOGNITION_HPP
