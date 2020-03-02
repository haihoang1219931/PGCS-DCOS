#include <opencv2/opencv.hpp>
#include <string>

#ifndef UTILS_HPP
#define UTILS_HPP

struct Char
{
    char c;
    float score;
};

std::string rawCorrect(std::string code);
bool        isDigit(const char c);
bool        isLowerCase(const char c);
bool        isUpperCase(const char c);
bool        haveChar(std::string code);
void        updateCodeTable(std::string code, std::vector<std::vector<Char>> &codeTable, int start, int end);
std::vector<int>
            match(std::string refString, std::string input);
std::string getResult(std::vector<std::vector<Char>> &codeTable, int plateType);

#endif // UTILS_HPP
