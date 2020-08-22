#include "LogController.h"

LogController::LogController(QObject *parent) : QObject(parent)
{
    system("/bin/mkdir -p logs");
}
void LogController::writeBinaryLog(QString filePath, QByteArray bytes){
    if(filePath != ""){
        QFile tmpFile(filePath);
        tmpFile.open(QFile::ReadWrite | QFile::Append);
        tmpFile.write(bytes);
        tmpFile.close();
    }
}
