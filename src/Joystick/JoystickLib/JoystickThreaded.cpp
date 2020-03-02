#include "JoystickThreaded.h"

JoystickThreaded::JoystickThreaded(QObject* parent) :
    QObject(parent)
{
    m_task = new JoystickTask();
    m_workerThread = new QThread(NULL);
    m_task->moveToThread(m_workerThread);

    connect(m_task,SIGNAL(joystickConnected(bool)),this,SIGNAL(joystickConnected(bool)));
    connect(m_task,SIGNAL(joystickConnected(bool)),this,SLOT(updateButtonAxis(bool)));
    connect(m_task,SIGNAL(btnClicked(int,bool)),this,SLOT(btnClicked(int,bool)));
    connect(m_task,SIGNAL(axisStateChanged(int,float)),this,SLOT(axisStateChanged(int,float)));
    //
//    connect(m_task,SIGNAL(btnClicked(int,bool)),this,SIGNAL(axesChanged()));
//    connect(m_task,SIGNAL(axisStateChanged(int,float)),this,SIGNAL(buttonsChanged()));
    //
    connect(m_workerThread, SIGNAL(started()), m_task, SLOT(doWork()));
    qDebug("joystick connect done");
}
/**/
JoystickThreaded::~JoystickThreaded()
{
    m_task->m_stop = true;
    m_workerThread->quit();
    if(!m_workerThread->wait(1000)){
         m_workerThread->terminate();
        m_workerThread->wait(1000);
    }
    m_workerThread->deleteLater();
    m_task->deleteLater();
    printf("JoystickThreaded destroyed\r\n");
}

void JoystickThreaded::start()
{
    if(m_workerThread->isRunning() == false)
        m_workerThread->start();
    else{
        pause(false);
    }
}

void JoystickThreaded::pause(bool _pause)
{
    m_task->pause(_pause);
}
void JoystickThreaded::stop()
{
    m_task->m_stop = true;
    m_workerThread->quit();

    if(!m_workerThread->wait(1000)){
         m_workerThread->terminate();
        m_workerThread->wait(1000);
    }
}
void JoystickThreaded::setJoyID(QString joyID){
    m_task->setJoyID(joyID);
}
QStringList JoystickThreaded::getListJoystick()
{
    std::string folderName = "/dev/input/";
    unsigned long numOfFile;
    QStringList listFiles;
    std::cout << "Checking\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\: " << folderName << std::endl;
    unsigned long countFiles = 0;
    DIR *dir = opendir(folderName.c_str());
    struct dirent *de;
    if(!dir)
    {
        printf("opendir() failed! Does it exist? %s\n", folderName.c_str());
        closedir(dir);
        return listFiles;
//        createNewFolder(folderName);
//        dir = opendir(folderName.c_str());
    }
    while((de = readdir(dir)) != NULL)
    {
       countFiles++;
       if(countFiles > 2)
       {
           if(listFiles.size()>1024)
           {
               listFiles.erase(listFiles.begin());
           }
           std::string fileName = string(de->d_name);
           std::size_t found = fileName.find("js");
           if(found!=std::string::npos){
                listFiles.append(QString::fromStdString(folderName + fileName));
//                std::cout << "File: " << folderName + fileName << std::endl;
           }
       }
    }
    numOfFile = countFiles;
    std::cout << "Count File: " << numOfFile << std::endl;
    closedir(dir);
    return listFiles;
}
QVariant JoystickThreaded::getJoystickInfo(QString jsFile){
    QVariant res;
    QVariantMap map;

    uint32_t version;
    uint8_t axes;
    uint8_t buttons;
    char name[256];
    int _fd = open(jsFile.toStdString().c_str(), false ? O_RDONLY : O_RDONLY | O_NONBLOCK);
    if(_fd < 0){
        map.insert("NAME", QString::fromStdString("Unknown"));
        map.insert("VERSION", QString::fromStdString("Unknown"));
        map.insert("AXES", QString::fromStdString("0"));
        map.insert("BUTTONS", QString::fromStdString("0"));
    }else{
        ioctl(_fd, JSIOCGNAME(256), name);
        ioctl(_fd, JSIOCGVERSION, &version);
        ioctl(_fd, JSIOCGAXES, &axes);
        ioctl(_fd, JSIOCGBUTTONS, &buttons);
        close(_fd);
        map.insert("NAME", QString::fromStdString(std::string(name)));
        map.insert("VERSION", QString::fromStdString(std::to_string(version)));
        map.insert("AXES", QString::fromStdString(std::to_string(axes)));
        map.insert("BUTTONS", QString::fromStdString(std::to_string(buttons)));
    }
    res = QVariant(map);
    return res;
}
void JoystickThreaded::updateButtonAxis(bool connected){
    if(connected){
        if(m_task->m_joystick.m_axes != m_axes.size() ||
                m_task->m_joystick.m_buttons != m_buttons.size()){
            printf("updateButtonAxis ============================== axis[%d] button[%d]\r\n",
                   m_task->m_joystick.m_axes,
                   m_task->m_joystick.m_buttons);
            // update axes
            m_axes.clear();
            for(int i=0; i< m_task->m_joystick.m_axes; i++)
                m_axes.append(new JSAxis(i));
            Q_EMIT axesChanged();
            // update buttons
            m_buttons.clear();
            for(int i=0; i< m_task->m_joystick.m_buttons; i++)
                m_buttons.append(new JSButton(i));
            Q_EMIT buttonsChanged();
        }
    }
}
void JoystickThreaded::btnClicked(int btnID,bool clicked){
    if(btnID < m_buttons.size()){
//        qDebug("Button %d is %s\n", btnID, !clicked ? "up" : "down");
        m_buttons[btnID]->setPressed(clicked);
    }
}
void JoystickThreaded::axisStateChanged(int axisID, float value){
    if(axisID < m_axes.size()){
//        qDebug("axisStateChanged %d value to %f\n", axisID, value);
        m_axes[axisID]->setValue(value);
        Q_EMIT axisValueChanged(axisID,value);
    }
}
