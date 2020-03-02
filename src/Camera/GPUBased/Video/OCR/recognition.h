#include "tensorflow/cc/ops/standard_ops.h"
#include <tensorflow/core/public/session.h>
#include <opencv2/opencv.hpp>

#ifndef RECOGNITION_H
#define RECOGNITION_H

class OCR
{
public:
    OCR();
    ~OCR();

    std::string recognize(std::vector<cv::Mat> chars, int sign);
    tensorflow::Status loadGraph(std::string &graphPath, std::unique_ptr<tensorflow::Session> *session);
    tensorflow::Status getLabels(std::vector<tensorflow::Tensor> &outputs, std::vector<std::pair<int, float>> &labels);
    std::string correctLP(std::string code);
    std::string correctSP(std::string code, int sign);
    bool isDigit(char c);
    bool isUpperCase(char c);
    void combine(std::string code);
    int countHyphens(std::string code);
    std::vector<int> cmpStr(std::string refStr, std::string tarStr);
    std::string findBest();

public:
    int m_counter = 0;
    bool m_stop = false;
    std::string m_result = "";
    std::vector<std::pair<std::string, int>> m_codeVector;

private:
    std::unique_ptr<tensorflow::Session> m_session;
    bool m_flag;

    std::string m_inputLayerName = "conv2d_1_input";
    std::string m_outputLayerName = "k2tfout_0";
    std::vector<char> m_dict = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                              'A', 'C', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N',
                              'P', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', '_'};
};

#endif // RECOGNITION_H
