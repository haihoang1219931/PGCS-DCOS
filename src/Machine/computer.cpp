#include "computer.hpp"
COMPUTER_INFO::COMPUTER_INFO(QObject *parent)
    : QObject(parent)
{
    QProcess p;
    p.start("awk", QStringList() << "/MemTotal/ { print $2 }" << "/proc/meminfo");
    p.waitForFinished();
    QString memory = p.readAllStandardOutput();
    m_ramTotal = memory.toLong() / 1024;
//    printf("memory = %ld\r\n",memory.toLong());
    p.close();
    m_timer = new QTimer(this);
    m_timer->setInterval(1000);
    m_timer->setSingleShot(false);

    connect(m_timer,&QTimer::timeout, this, &COMPUTER_INFO::checkSystemMem);
    m_timer->start();
    m_usbWatcher = new QFileSystemWatcher(this);
    std::string path;
#ifdef __linux__
    //linux code goes here
    std::string user = getenv("USER");
    path = "/media/"+user;
    printf("Config USB at %s\r\n",path.c_str());
#elif _WIN32
    // windows code goes here
#else

#endif
    m_usbWatcher->addPath(QString::fromStdString(path));
    connect(m_usbWatcher,SIGNAL(directoryChanged(QString)),this,SLOT(handleUsbDetected(QString)));
}
COMPUTER_INFO::~COMPUTER_INFO(){
    m_timer->deleteLater();
    m_usbWatcher->deleteLater();
}

QString COMPUTER_INFO::freeSpace(){
    char cFreeSpace[100];
    float fFreeSpace,fTotalSpace;
#ifdef __linux__
    //linux code goes here
    struct statvfs fiData;
    statvfs("/",&fiData);
    fFreeSpace = fiData.f_bsize * fiData.f_bfree/1000000000;
    fTotalSpace = fiData.f_bsize * fiData.f_blocks/1000000000;
#elif _WIN32
    // windows code goes here
    QStorageInfo storage = QStorageInfo::root();

    qDebug() << storage.rootPath();
    if (storage.isReadOnly())
       qDebug() << "isReadOnly:" << storage.isReadOnly();

    qDebug() << "name:" << storage.name();
    qDebug() << "filesystem type:" << storage.fileSystemType();
    qDebug() << "size:" << storage.bytesTotal()/1024/1024 << "MB";
    qDebug() << "free space:" << storage.bytesAvailable()/1024/1024 << "MB";
    fFreeSpace = storage.bytesAvailable()/1024/1024/1024;
    fTotalSpace = storage.bytesTotal()/1024/1024/1024;
#else

#endif
    sprintf(cFreeSpace,"%.1f% (%.1f GiB / %.1f GiB)",fFreeSpace/fTotalSpace*100,fFreeSpace,fTotalSpace);
    return QString(cFreeSpace);
}
QString COMPUTER_INFO::version(){
    return "rc-emu-2.5.0.3.239.7547590";
}
QString COMPUTER_INFO::media(){
    std::string path;
#ifdef __linux__
    //linux code goes here
    std::string user = getenv("USER");
    path = "/media/"+user;
#elif _WIN32
    // windows code goes here
#else

#endif
    return QString::fromStdString(path);
}
QString COMPUTER_INFO::user(){
    std::string user;
#ifdef __linux__
    //linux code goes here
    user = getenv("USER");
#elif _WIN32
    // windows code goes here
#else

#endif
    return QString::fromStdString(user);
}
QString COMPUTER_INFO::homeFolder(){
    return QDir::homePath();
}

void COMPUTER_INFO::calibrateJoystick(){
    QString cmd = "jstest-gtk";
    QProcess process;
    process.startDetached(cmd);
}

void COMPUTER_INFO::setSystemTime(){
    QString cmd = "time-admin";
    QProcess process;
    process.startDetached(cmd);
}

void COMPUTER_INFO::restartApplication(){
    QStringList args = QApplication::arguments();
    args.removeFirst();
    QProcess::startDetached(QApplication::applicationFilePath(),args);
        QCoreApplication::quit();
}
void COMPUTER_INFO::quitApplication(){
    printf("%s\r\n",__func__);
    QGuiApplication::quit();
}

void COMPUTER_INFO::restartComputer(){
    printf("restartComputer\r\n");
    QProcess process;
    process.startDetached("echo 1 | sudo -S reboot");
}
void COMPUTER_INFO::shutdownComputer(){
    printf("%s\r\n",__func__);
    QProcess process;
    process.startDetached("echo 1 | sudo -S shutdown -P now");
}
void COMPUTER_INFO::openFolder(QString folder){
    QString cmd = QString("nautilus ") + folder;
    QProcess process;
    process.startDetached(cmd);
}
void COMPUTER_INFO::handleUsbDetected(QString mediaFolder){
    printf("Detect new USB at : %s\r\n",mediaFolder.toStdString().c_str());
    QDir directory(mediaFolder);
    QFileInfoList lstCurrentUSB = directory.entryInfoList(QDir::NoDotAndDotDot);
    QStringList lstUSBAdded;
    QStringList lstUSBRemoved;
    if(lstUSBAdded.size() > 0){
        usbAdded(lstUSBAdded);
    }
    if(lstUSBRemoved.size() > 0){
        usbRemoved(lstUSBRemoved);
    }
//    usbDetected(usbName);
}

void COMPUTER_INFO::checkSystemMem(){
    std::string timeStamp = FileController::get_time_stamp();
    FILE* file = fopen("/proc/self/status", "r");
    char line[128];
    while (fgets(line, 128, file) != NULL){
        if (strncmp(line, "VmRSS:", 6) == 0){
            m_ramUsed = FileController::parseLine(line) / 1024;
            break;
        }
    }
    fclose(file);
#ifdef DEBUG
    printf("[%s] [RAM]: %d/%d %.02f%\r\n",timeStamp.c_str(),
           m_ramUsed,m_ramTotal,static_cast<float>(m_ramUsed)/static_cast<float>(m_ramTotal)*100.0f);
#endif
}
int COMPUTER_INFO::fileSize(QString fileName){
    QFileInfo fileinfo(fileName);
    qint64 size = fileinfo.size();
    return size;
}
void COMPUTER_INFO::fileCopy(QString fileSrc,QString fileDst){
    QFile::copy(fileSrc,fileDst);
}
void COMPUTER_INFO::fileDelete(QString fileName){

}
