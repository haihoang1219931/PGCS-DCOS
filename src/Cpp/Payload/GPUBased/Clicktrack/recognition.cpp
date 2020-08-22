#include "recognition.h"
#include <chrono>

OCR::OCR()
{
    std::cout << "Tensorflow model is loading...\n";
    std::string graphPath = "../GPUBased/Clicktrack/mynet.pb";
    tensorflow::Status status = loadGraph(graphPath, &m_session);

    if (!status.ok()) {
        std::cerr << "Fail to load tensorflow model!\n";
    }

    // first run (decoy)
    tensorflow::Tensor inTensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, 32, 32, 1}));
    tensorflow::StringPiece tmp_data = inTensor.tensor_data();
    std::vector<tensorflow::Tensor> outputs;
    cv::Mat c = cv::Mat::zeros(32, 32, CV_32FC1);
    std::memcpy(const_cast<char *>(tmp_data.data()), c.data, 32 * 32 * sizeof(float));
    m_session->Run({{m_inputLayerName, inTensor}}, {m_outputLayerName}, {}, &outputs);

    m_flag = false;
}

OCR::~OCR()
{
    m_session->Close();
}

std::string OCR::recognize(std::vector<cv::Mat> chars, int sign)
{
    std::lock_guard<std::mutex> locker(m_mtx);    
    std::string code = "";
    cv::Mat c;
    tensorflow::Status status;
    tensorflow::Tensor inTensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, 32, 32, 1}));
    tensorflow::StringPiece tmp_data = inTensor.tensor_data();
    std::vector<tensorflow::Tensor> outputs;

    for (uint i = 0; i < chars.size(); i++) {
        chars[i].convertTo(c, CV_32FC1);
        std::memcpy(const_cast<char *>(tmp_data.data()), c.data, 32 * 32 * sizeof(float));

        status = m_session->Run({{m_inputLayerName, inTensor}}, {m_outputLayerName}, {}, &outputs);
        std::vector<std::pair<int, float>> labels;
        getLabels(outputs, labels);

        if (labels.size() == 1) {
            code += m_dict[labels[0].first];
        } else {
            std::sort(labels.begin(), labels.end(), [](std::pair<int, float> a, std::pair<int, float> b) {
                return a.second > b.second;
            });

            if (labels[0].first == 28) {
                if (labels[1].second >= 0.1) {
                    code += m_dict[labels[1].first];
                } else {
                    code += m_dict[labels[0].first];
                }
            } else {
                code += m_dict[labels[0].first];
            }
        }

    }

//	printf("Code : %s with pos = %d\n", code.c_str(), sign);
    std::string result;
    if(sign == -1)
        result = correctLP(code);
    else
        result = correctSP(code, sign);
//	printf("Result : %s\n", result.c_str());

    return result;
}

tensorflow::Status OCR::loadGraph(std::string &graphPath, std::unique_ptr<tensorflow::Session> *session)
{
    tensorflow::GraphDef graphDef;
    tensorflow::Status status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), graphPath, &graphDef);

    if (!status.ok()) {
        return tensorflow::errors::NotFound("Failed to load binary graph at '" + graphPath + "'");
    }

    auto ops = tensorflow::SessionOptions();
    ops.config.mutable_gpu_options()->set_allocated_visible_device_list(0);
    ops.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.75);
    ops.config.mutable_gpu_options()->set_allow_growth(true);
    session->reset(tensorflow::NewSession(ops));
    status = (*session)->Create(graphDef);

    if (status.ok()) {
        return status;
    }

    return tensorflow::Status::OK();
}

tensorflow::Status OCR::getLabels(std::vector<tensorflow::Tensor> &outputs, std::vector<std::pair<int, float>> &labels)
{
    if (outputs.at(0).shape().dims() == 1) {
        if (outputs.at(0).shape().dim_size(0) == 29) {
            float *data_ptr = outputs.at(0).flat<float>().data();

            for (int i = 0; i < 29; i++) {
                if (data_ptr[i] >= 0.05)
                    labels.push_back(std::make_pair(i, data_ptr[i]));
            }
        } else {
            std::cerr << "Invalid Output.\n";
        }
    } else {
        std::cerr << "Invalid Output.\n";
    }

    return tensorflow::Status::OK();
}

std::string OCR::correctLP(std::string code)
{
    std::string newCode = "";
        int length = code.size();

        std::string hCode, rCode;
        hCode = code.substr(0, 3);
        rCode = code.substr(3, length - 3);

        for(uint i = 0; i < hCode.length(); i++)
        {
            if(i != 2)
            {
                if(isDigit(hCode[i]) || hCode[i] == '_')
                    newCode += hCode[i];
                else if(hCode[i] == 'A')
                    newCode += '8';
                else if(hCode[i] == 'C' || hCode[i] == 'U')
                    newCode += '0';
                else if(hCode[i] == 'G')
                    newCode += '6';
                else if(hCode[i] == 'L')
                    newCode += '4';
                else if(hCode[i] == 'S')
                    newCode += '5';
                else if(hCode[i] == 'T' || hCode[i] == 'X' || hCode[i] == 'Y')
                    newCode += '1';
                else if(hCode[i] == 'Z')
                    newCode += '2';
                else
                    newCode += '_';
            }
            else if(i == 2)
            {
                if(isUpperCase(hCode[i]))
                    newCode += hCode[i];
                else if(hCode[i] == '0')
                    newCode += 'D';
                else if(hCode[i] == '2' || hCode[i] == '7')
                    newCode += 'Z';
                else if(hCode[i] == '5')
                    newCode += 'S';
                else if(hCode[i] == '6')
                    newCode += 'G';
                else if(hCode[i] == '8')
                    newCode += 'B';
                else
                    newCode += '_';
            }
        }

        newCode += '-';
        int c = 0;

        for(uint i = 0; i < rCode.length(); i++)
        {
            if(c > 4)
                break;
            else
            {
                if(isDigit(rCode[i]))
                {
                    newCode += rCode[i];
                    c++;
                }
                else if(rCode[i] == 'A')
                {
                    newCode += '8';
                    c++;
                }
                else if(rCode[i] == 'C' || rCode[i] == 'U')
                {
                    newCode += '0';
                    c++;
                }
                else if(rCode[i] == 'G')
                {
                    newCode += '6';
                    c++;
                }
                else if(rCode[i] == 'L')
                {
                    newCode += '4';
                    c++;
                }
                else if(rCode[i] == 'S')
                {
                    newCode += '5';
                    c++;
                }
                else if(rCode[i] == 'T' || rCode[i] == 'X' || rCode[i] == 'Y')
                {
                    newCode += '1';
                    c++;
                }
                else if(rCode[i] == 'Z')
                {
                    newCode += '2';
                    c++;
                }
//                else if(rCode[i] == '_')
//                    continue;
//                else
//                    newCode += '_';
            }
        }
        for(int i = length; i > 0; i--)
        {
            if(newCode[i] != '_')
                return newCode.substr(0, i + 1);
        }
        return newCode;
}

std::string OCR::correctSP(std::string code, int sign)
{
    std::string newCode = "";
    std::string subCode = "";
    int length = code.size();

    if(length < 5)
        return "";

    std::string hCode, rCode;
    hCode = code.substr(0, sign);
    rCode = code.substr(sign, length - sign);
//    return hCode + "/" + rCode;
    std::string temp;
    bool count = false;

	// Process for hCode
	for(uint i = 0; i < hCode.size(); i++)
	{
		if(hCode[i] != '_')
		{
			temp = hCode.substr(i, hCode.size() - i);
			break;
		}
	}
	if(temp.size() < 3)
        temp = "";
    else
    {
        int idx = -1;
        for(uint i = 0; i < temp.size(); i++)
        {
            if(temp[i] == '-')
                idx = i;
        }
        if(idx == -1)
        {
            for(uint i = 0; i < temp.size(); i++)
            {
                if(temp[i] != '-')
                    subCode += temp[i];
            }
            if(subCode.size() < 3)
                return "";
            else
			{
                int c = 0;
                for(uint i = 0; i < subCode.size(); i++)
                {
                    if(c == 0 || c == 1 || c == 3)
                    {
                        if(isDigit(subCode[i]))
                        {
                            newCode += subCode[i];
                            c++;
                        }
                        else if(subCode[i] == 'C' || subCode[i] == 'U')
                        {
                            newCode += '0';
                            c++;
                        }
                        else if (subCode[i] == 'G')
                        {
                            newCode += '6';
                            c++;
                        }
                        else if (subCode[i] == 'S')
                        {
                            newCode += '5';
                            c++;
                        }
                        else if(subCode[i] == 'U')
                        {
                            newCode += '0';
                            c++;
                        }
                        else if (subCode[i] == 'X' || subCode[i] == 'Y' || subCode[i] == 'T')
                        {
                            newCode += '1';
                            c++;
                        }
                    }
                    else if(c == 2)
                    {
                        if(isUpperCase(subCode[i]))
                        {
//                            newCode += '-';
                            newCode += subCode[i];
                            c++;
                        }
                        else if(subCode[i] == '0')
                        {
//                            newCode += '-';
                            newCode[i] += 'D';
                            c++;
                        }
                        else if(subCode[i] == '1')
                        {
//                            newCode += '-';
                            newCode[i] += 'T';
                            c++;
                        }
                        else if(subCode[i] == '2')
                        {
//                            newCode += '-';
                            newCode[i] += 'Z';
                            c++;
                        }
                        else if(subCode[i] == '5')
                        {
//                            newCode += '-';
                            newCode[i] += 'S';
                            c++;
                        }
                        else if(subCode[i] == '6')
                        {
//                            newCode += '-';
                            newCode[i] += 'G';
                            c++;
                        }
                        else if(subCode[i] == '7')
                        {
//                            newCode += '-';
                            newCode[i] += 'Z';
                            c++;
                        }
                        else if(subCode[i] == '8')
                        {
//                            newCode += '-';
                            newCode[i] += 'B';
                            c++;
                        }
                    }
                    else if(c > 3)
                    {
                        break;
                    }
                }
				if(c < 2)
					return "";
            }
        }
//        else if(idx != 2)
//            return newCode;
        else
        {
            newCode += temp[idx - 2];
            newCode += temp[idx - 1];
//            newCode += '-';
            int c = 0;
            for(uint i = idx + 1; i < temp.size(); i++)
            {
                if(c == 0)
                {
                    if(isUpperCase(temp[i]))
                    {
                        newCode += temp[i];
                        c = 1;
                    }
                    else if(subCode[i] == '0')
                    {
                        newCode[i] += 'D';
                        c++;
                    }
                    else if(subCode[i] == '1')
                    {
                        newCode[i] += 'T';
                        c++;
                    }
                    else if(subCode[i] == '2')
                    {
                        newCode[i] += 'Z';
                        c++;
                    }
                    else if(subCode[i] == '5')
                    {
                        newCode[i] += 'S';
                        c++;
                    }
                    else if(subCode[i] == '6')
                    {
                        newCode[i] += 'G';
                        c++;
                    }
                    else if(subCode[i] == '7')
                    {
                        newCode[i] += 'Z';
                        c++;
                    }
                    else if(subCode[i] == '8')
                    {
                        newCode[i] += 'B';
                        c++;
                    }
                }
                else if(c == 1)
                {
                    if(isDigit(temp[i]))
                    {
                        newCode += temp[i];
                        break;
                    }
                    else if(temp[i] == 'C')
                    {
                        newCode += '0';
                        break;
                    }
                    else if (temp[i] == 'G')
                    {
                        newCode += '6';
                        break;
                    }
                    else if (temp[i] == 'S')
                    {
                        newCode += '5';
                        break;
                    }
                    else if(temp[i] == 'U')
                    {
                        newCode += '0';
                        break;
                    }
                    else if (temp[i] == 'X' || temp[i] == 'Y' || temp[i] == 'T')
                    {
                        newCode += '1';
                        break;
                    }
                }
            }
        }
    }
    newCode += "/";

	// Process for rCode
    for(uint i = 0; i < rCode.size(); i++)
    {
        if(isDigit(rCode[i]))
            newCode += rCode[i];
        else if(isUpperCase(rCode[i]))
        {
            if(rCode[i] == 'A' || rCode[i] == 'B')
            {
                newCode += '8';
            }
            else if (rCode[i] == 'C' || rCode[i] == 'D')
            {
                newCode += '0';
//                c++;
            }
            else if (rCode[i] == 'G')
            {
                newCode += '6';
//                c++;
            }
            else if (rCode[i] == 'L')
            {
                newCode += '4';
//                c++;
            }
            else if (rCode[i] == 'S')
            {
                newCode += '5';
//                c++;
            }
            else if (rCode[i] == 'X' || rCode[i] == 'Y' || rCode[i] == 'T')
            {
                newCode += '1';
//                c++;
            }
        }
    }
    return newCode;
}

bool OCR::isDigit(char c)
{
    if ((int)c >= 48 && (int)c <= 57)
        return true;

    return false;
}

bool OCR::isUpperCase(char c)
{
    if ((int)c >= 65 && (int)c <= 90)
        return true;

    return false;
}

int OCR::countHyphens(std::string code)
{
    int counter = 0;
    for(uint i = 0; i < code.size(); i++)
        if(code[i] == '_')
            counter++;

    return counter;
}

std::vector<int> OCR::cmpStr(std::string refStr, std::string tarStr)
{
    std::vector<int> diff;
    int refLen = refStr.size();
    int tarLen = tarStr.size();
    if(refLen == tarLen)
    {
        for(int i = 0; i < refLen; i++)
        {
            if((int)refStr[i] != (int)tarStr[i])
                diff.push_back(i);
        }
    }
    else if(refLen > tarLen)
    {
        // Assume that tarStr is benchmark
        for(int i = 0; i < tarLen; i++)
        {
            if((int)refStr[i] != (int)tarStr[i])
                diff.push_back(i);
        }
        diff.push_back(1 - refLen);
    }
    else
    {
        for(int i = 0; i < refLen; i++)
        {
            if((int)refStr[i] != (int)tarStr[i])
                diff.push_back(i);
        }
        diff.push_back(tarLen - 1);
    }

    return diff;
}

std::string OCR::findBest()
{
    std::lock_guard<std::mutex> locker(m_mtx);
    std::sort(m_codeVector.begin(), m_codeVector.end(), [](std::pair<std::string, int> a, std::pair<std::string, int> b)
    {
        return a.second > b.second;
    });
    for(uint i = 0; i < m_codeVector.size(); i++)
    {
        if(countHyphens(m_codeVector[i].first) == 0)
        {
            return m_codeVector[i].first;
        }
    }
    return m_codeVector[0].first;
}

void OCR::combine(std::string code)
{
    std::lock_guard<std::mutex> locker(m_mtx);
    int tarLen = code.size();
    int score = 0;
    if(m_stop == false)
    {
        if(countHyphens(code) < 2)
        {
            bool insert = true;
            std::vector<int> diff;
            for(uint i = 0; i < m_codeVector.size(); i++)
            {
                diff.clear();
                diff = cmpStr(m_codeVector[i].first, code);
                if(diff.size() == 0)
                {
                    m_codeVector[i].second++;
                    insert = false;
                }
                if(diff.size() == 1)
                {
                    if(m_codeVector[i].first.size() != tarLen)
                    {
                        m_codeVector[i].second++;
                        score++;
                    }
                    else
                    {
                        if(m_codeVector[i].first[diff[0]] == '_' || code[diff[0]] == '_')
                        {
                            m_codeVector[i].second++;
                            score++;
                        }
                    }
                }
            }
            if(insert)
            {
                m_codeVector.push_back(std::make_pair(code, 1));
            }
        }
    }
}
