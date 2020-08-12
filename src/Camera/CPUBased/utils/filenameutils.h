#ifndef FILENAMEUTILS_H
#define FILENAMEUTILS_H
#include <iostream>
#include <time.h>
#include <stdio.h>
#ifdef __linux__
    //linux code goes here
    #include <sys/time.h>
    #include <string.h>
#elif _WIN32
    // windows code goes here
#else

#endif
using namespace std;
namespace Utils
{
    inline std::string get_day()
    {
        time_t t = time(0);   // get time now
        struct tm * now = localtime( & t );
        char day[64];
        sprintf(day, "%04d-%02d-%02d",
                (now->tm_year + 1900),
                now->tm_mon + 1,
                now->tm_mday);
        return std::string(day);
    }
    inline std::string get_time_stamp()
    {
        time_t t = time(0);   // get time now
        struct tm * now = localtime( & t );
        char timestamp[64];
        sprintf(timestamp, "%04d-%02d-%02d_%02d-%02d-%02d",
                (now->tm_year + 1900),
                now->tm_mon + 1,
                now->tm_mday,
                now->tm_hour,
                now->tm_min,
                now->tm_sec);

        return std::string(timestamp);
    }
    inline int parseLine(char* line){
        // This assumes that a digit will be found and the line ends in " Kb".
        int i = strlen(line);
        const char* p = line;
        while (*p <'0' || *p > '9') p++;
        line[i-3] = '\0';
        i = atoi(p);
        return i;
    }
}
#endif // FILENAMEUTILS_H
