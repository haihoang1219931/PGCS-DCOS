/**
 * ===========================================================
 * Project:
 * Module: Log file
 * Module Short Description:
 * Author: Trung Ng
 * Date: 05/12/2020
 * Viettel Aerospace Institude - Viettel Group
 * ===========================================================
 */

#ifndef LOGFILE_HPP
#define LOGFILE_HPP

#include <iostream>
#include <fstream>
#include "file_utils.hpp"

namespace Utils {
    class LogFile {
        public:
        LogFile(std::string fileName) {
            mFileName = fileName;
            if( !Utils::directoryExists("microhard") ) {
               Utils::createLogFolder("microhard");
            }
            qDebug("%s", fileName.c_str());
            file.open ("microhard/" + fileName, std::ios::out | std::ios::app );
            if (file.is_open()) {
                file << "\n ================ " << Utils::getCurrentDate() << "\n";
            }
            file << "RSSI, " <<  "SNR, " << "DISTANCE\n";
            file.close();
        }

        ~LogFile() {
            file.close();
        }

        void write(std::string content){
            file.open("microhard/" + mFileName, std::ios::app | std::ios::out);
            file << content;
            file.close();
        }

        private:
            std::fstream file;
            std::string mFileName;
    };
}


#endif // LOGFILE_HPP
