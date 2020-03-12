#ifndef FileControler_H
#define FileControler_H

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <sys/stat.h>
#ifdef _WIN32
   #include <io.h>
   #define access    _access_s
#else
   #include <unistd.h>
#endif
#ifdef __linux__
    //linux code goes here
    #include <sys/time.h>
    #include <string.h>
#elif _WIN32
    // windows code goes here
#else

#endif

using namespace std;
class FileController
{
public:
    FileController();
    static bool isExists( const std::string &Filename );
    static std::string get_day();
    static std::string get_time();
    static std::string get_time_stamp();
    static void addLine(string _file, string _newLine);
    static vector<string> readFile(string _file, int _start, int _numLine);
    static vector<string> readFile(string _file);
    static int parseLine(char* line);
};

#endif // FileControler_H
