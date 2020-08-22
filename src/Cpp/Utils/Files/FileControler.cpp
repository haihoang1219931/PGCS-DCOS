#include "FileControler.h"

FileController::FileController()
{

}

bool FileController::isExists( const std::string &filename )
{
    struct stat buffer;
        int exist = stat(filename.c_str(),&buffer);
        if(exist == 0)
            return true;
        else // -1
            return false;
//    return access( Filename.c_str(), 0 ) == 0;
}
std::string FileController::get_day()
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
std::string FileController::get_time()
{
    time_t t = time(0);   // get time now
    struct tm * now = localtime( & t );
    char timestamp[64];
    sprintf(timestamp, "%02d-%02d-%02d",
            now->tm_hour,
            now->tm_min,
            now->tm_sec);

    return std::string(timestamp);
}
std::string FileController::get_time_stamp(){
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
void FileController::addLine(string _file, string _newLine){
    ofstream file; // out file stream
    file.open(_file,ios_base::openmode::_S_app);
    file << _newLine <<std::endl;
    file.close();
}
vector<string> FileController::readFile(string _file,int _start, int _numLine){
    vector<string> lstFiles;
    std::ifstream infile(_file);
    std::string lineString;
    int count = 0;
    while (infile >> lineString)
    {
        if(count >= _start+_numLine) break;
        if(count >= _start){
            lstFiles.push_back(lineString);
        }
        count++;
    }
    infile.close();
    return lstFiles;
}

vector<string> FileController::readFile(string _file){
    vector<string> lstFiles;
    std::ifstream infile(_file);
    std::string lineString;
    while (infile >> lineString)
    {
        if(lineString.size() > 0)
        lstFiles.push_back(lineString);
    }
    infile.close();
    return lstFiles;
}
int FileController::parseLine(char* line){
    // This assumes that a digit will be found and the line ends in " Kb".
    int i = strlen(line);
    const char* p = line;
    while (*p <'0' || *p > '9') p++;
    line[i-3] = '\0';
    i = atoi(p);
    return i;
}
