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

using namespace std;
class FileController
{
public:
    FileController();
    static bool isExists( const std::string &Filename );
    static std::string get_day();
    static std::string get_time_stamp();
    static std::string get_time();
    static void addLine(string _file, string _newLine);
    static vector<string> readFile(string _file, int _start, int _numLine);
    static vector<string> readFile(string _file);
};

#endif // FileControler_H
