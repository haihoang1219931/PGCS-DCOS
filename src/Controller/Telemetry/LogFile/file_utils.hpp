/**
 * ===========================================================
 * Project:
 * Module: Utils
 * Module Short Description: Utils saving
 * Author: Trung Nguyen
 * Date: 18/11/2018
 * Viettel Aerospace Institude - Viettel Group
 * ===========================================================
 */

#ifndef FILE_UTILS_HPP
#define FILE_UTILS_HPP


#include <iostream>
#include <string>
#include <sys/stat.h>
#include <dirent.h>
#include <ctime>
#include <fstream>
#include <unistd.h>
#include <QObject>
#include "helper.hpp"

namespace Utils {
    struct dateTimeUints {
        dateTimeUints(int sec_, int min_, int hour_, int day_, int mon_, int year_)
        {
            sec = sec_;
            min = min_;
            hour = hour_;
            day = day_;
            mon = mon_;
            year = year_;
        }

        int sec = 0;
        int min = 0;
        int hour = 0;
        int day = 0;
        int mon = 0;
        int year = 0;
    };

    /* FileExtension checking supported method */
    inline std::string fileExtension( const std::string& fileName )
    {
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream tokenStream( fileName );
        char delimiter = '.';
        while ( std::getline(tokenStream, token, delimiter) )
        {
           tokens.push_back(token);
        }
        return tokens.back();
    }

    inline bool createLogFolder(std::string folderName_)
    {
        const int dir_err = mkdir(folderName_.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (-1 == dir_err)
        {
            printf("Error creating directory!\n");
            return false;
        }
        return true;
    }

    inline bool directoryExists( const char* pzPath )
    {
        if ( pzPath == NULL) return false;

        DIR *pDir;
        bool bExists = false;

        pDir = opendir (pzPath);

        if (pDir != NULL)
        {
            bExists = true;
            (void) closedir (pDir);
        }

        return bExists;
    }

    inline dateTimeUints getDateTimeUints()
    {
        time_t t = time(NULL);
        tm* timePtr = localtime(&t);
        return dateTimeUints( timePtr->tm_sec, timePtr->tm_min, timePtr->tm_hour,
                              timePtr->tm_mday, timePtr->tm_mon + 1, timePtr->tm_year + 1900 );
    }

    inline std::string getCurrentDate()
    {
        std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

        char buf[100] = {0};
        std::strftime(buf, sizeof(buf), "%m/%d/%y - %H:%M:%S", std::localtime(&now));
        return buf;
    }

    inline bool createFile( const std::string& filePath )
    {
        std::vector<std::string> pathElements = splitStr(filePath, '/');
        std::string paths = pathElements[0];
        for(int i = 1; i < pathElements.size(); i++)
        {
            if( !directoryExists(paths.c_str()) )
            {
                createLogFolder(paths);
            }
            paths = paths + "/" + pathElements[i];
        }
        std::fstream file;
        file.open(paths, std::ios::out | std::ios::app);
        if( !file )
        {
            qDebug("error in creating file!!!\n");
            return false;
        }
        if( file.is_open() )
        {
            file << "<?xml version=\"1.0\"?>\n";
        }
        //file.close();
        return true;
    }

    inline bool fileExist( const std::string& filePath )
    {
        std::ifstream f(filePath.c_str());
        return f.good();
    }
}
#endif // FILE_UTILS_HPP
