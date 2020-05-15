#ifndef STRINGS_HPP
#define STRINGS_HPP

#include <iostream>
#include <string>
#include <vector>
#include <istream>
#include <fstream>
#include <dirent.h>
#include <string>
#include <algorithm>

namespace Utils {
    /**
     * @brief splitStr: Split string into vector by delimiter char
     * @param s
     * @param delimiter
     * @return
     */
    inline std::vector<std::string> splitStr(const std::string& s, char delimiter)
    {
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream tokenStream(s);
        while( std::getline(tokenStream, token, delimiter) )
        {
            tokens.push_back(token);
        }
        return tokens;
    }

    /**
     * @brief checkInputSourceExist
     * @param pathToDev
     * @return
     */
    inline bool checkInputSourceExist(const std::string& pathToDev)
    {
        return (bool)std::ifstream(pathToDev);
    }

    /**
     * @brief getAllSourceCams: Get all camera sources
     * @return
     */
    inline std::vector<std::string> getAllSourceCams()
    {
        std::vector<std::string> srcCams;

        DIR *dir;
        struct dirent *ent;
        if ((dir = opendir ("/dev/")) != NULL) {
          /* print all the files and directories within directory */
          std::string fileName;
          while ((ent = readdir (dir)) != NULL) {
            fileName = std::string(ent->d_name);
            if( fileName.find("video") != std::string::npos )
            {
                srcCams.push_back(std::string("/dev/") + fileName);
            }
          }
          closedir (dir);
        } else {
          /* could not open directory */
          perror ("");

        }
        return srcCams;
    }

    /**
     * @brief toUpper: string to uppercase
     * @param str
     */
    inline void strToUpper( std::string& str)
    {
        std::transform(str.begin(), str.end(), str.begin(), ::toupper);
    }

    /**
     * @brief strToLower: string to lowercase
     * @param str
     */
    inline void strToLower( std::string& str)
    {
        std::transform(str.begin(), str.end(), str.begin(), ::tolower);
    }
}

#endif // STRINGS_HPP

