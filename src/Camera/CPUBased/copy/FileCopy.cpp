#include "FileCopy.h"

FileCopy::FileCopy(QObject *parent) : QObject(parent)
{

}
FileCopy::~FileCopy()
{

}
void FileCopy::start(){
    m_stop = false;
}
void FileCopy::stop(){
    m_stop = true;
}
void FileCopy::setState(QString state){
    if(state == "PAUSE"){
        m_mutex->lock();
        m_pause = true;
        m_mutex->unlock();
    }else if(state == "CONTINUE"){
        m_mutex->lock();
        m_pause = false;
        m_mutex->unlock();
        m_pauseCond->wakeAll();
    }
}
void FileCopy::copyFile(QString src,QString dst){
    m_src = src;
    m_dst = dst;
    printf("Copy file %s to %s\r\n",
           m_src.toStdString().c_str(),
           m_dst.toStdString().c_str());
}
void FileCopy::doWork(){
    QFile srcFile(m_src);
    QFile dstFile(m_dst);
    if(!srcFile.open(QIODevice::ReadOnly)){
        printf("Not read file\r\n");
        return;
    }
    if(!dstFile.open(QIODevice::WriteOnly | QIODevice::Unbuffered)){
        printf("Not write file\r\n");
        return;
    }
    qint64 count = 0;
    qint64 srcFileSize = srcFile.size();
    qint64 dSize = 1000000;
    QByteArray buffer;
    printf("Ready to copy file %ld bytes\r\n",srcFileSize);
    while(m_stop == false){
        m_mutex->lock();
        if(m_pause)
            m_pauseCond->wait(m_mutex); // in this place, your thread will stop to execute until someone calls resume
        m_mutex->unlock();
        if(m_stop == true) break;
        buffer= srcFile.read(dSize);
        if(!buffer.isEmpty()){
            dstFile.write(buffer);
            count+=buffer.size();
//            float percent = (float)((double)count/(double)srcFileSize);
//            printf("copied %f\%\r\n",percent);
            processChanged((float)count/(1000*1000),(float)srcFileSize/(1000*1000));
        }else{
            break;
        }
        msleep(1);
//        sleep(1);
    }
    printf("waiting file write file %s finish\r\n",m_dst.toStdString().c_str());
//    sleep(20);
    stopped();
//    stateChange("COPY_DONE");
    printf("Copied done\r\n");
}
void FileCopy::msleep(int ms){
#ifdef __linux__
    //linux code goes here
    struct timespec ts = { ms / 1000, (ms % 1000) * 1000 * 1000 };
    nanosleep(&ts, NULL);
#elif _WIN32
    // windows code goes here
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
#else

#endif
}
