#include "ConnectionChecking.h"

ConnectionChecking::ConnectionChecking(QObject *parent) : QObject(parent)
{

}
ConnectionChecking::~ConnectionChecking()
{

}
void ConnectionChecking::start(){
    m_stop = false;
}
/**
 * @brief Convert String to Number
 */
template <typename TP>
TP str2num( std::string const& value ){

    std::stringstream sin;
    sin << value;
    TP output;
    sin >> output;
    return output;
}


/**
 * @brief Convert number to string
 */
template <typename TP>
std::string num2str( TP const& value ){
    std::stringstream sin;
    sin << value;
    return sin.str();
}


/**
 * @brief Execute Generic Shell Command
 *
 * @param[in]   command Command to execute.
 * @param[out]  output  Shell output.
 * @param[in]   mode read/write access
 *
 * @return 0 for success, 1 otherwise.
 *
*/
int Execute_Command( const std::string&  command,
                     std::string&        output,
                     const std::string&  mode = "r")
{
    // Create the stringstream
    std::stringstream sout;

    // Run Popen
    FILE *in;
    char buff[512];

    // Test output
#ifdef __linux__
    //linux code goes here
    if(!(in = popen(command.c_str(), mode.c_str()))) return 1;
#elif _WIN32
    // windows code goes here
    if(!(in = _popen(command.c_str(), mode.c_str()))) return 1;
#else

#endif

    // Parse output
    while(fgets(buff, sizeof(buff), in)!=NULL){
        sout << buff;
    }

    // Close
#ifdef __linux__
    //linux code goes here
    int exit_code = pclose(in);
#elif _WIN32
    // windows code goes here
    int exit_code = _pclose(in);
#else

#endif
    // set output
    output = sout.str();

    // Return exit code
    return exit_code;
}

/**
 * @brief Ping
 *
 * @param[in] address Address to ping.
 * @param[in] max_attempts Number of attempts to try and ping.
 * @param[out] details Details of failure if one occurs.
 *
 * @return True if responsive, false otherwise.
 *
 * @note { I am redirecting stderr to stdout.  I would recommend
 *         capturing this information separately.}
 */
bool Ping( const std::string& address,
           const int&         max_attempts,
           std::string&       details )
{
    // Format a command string
    std::string command;
    std::string output;
#ifdef __linux__
    //linux code goes here
    command = "ping -w 1 -c " + num2str(max_attempts) + " " + address + " 2>&1";
    int code = Execute_Command( command, details );
#elif _WIN32
    // windows code goes here
    command = "ping -w 1000 -n " + num2str(max_attempts) + " " + address;
    int code = 0;
    QProcess process;
    process.start(QString::fromStdString(command));
    process.waitForFinished(-1); // will wait forever until finished

    QString cmdOut = process.readAllStandardOutput().toLower();
    if(cmdOut.contains(QString("destination host unreachable"))||
       cmdOut.contains(QString("request timed out"))){
        code = -1;
    }else
        code = 0;
//    qDebug("cmdOut: %s\r\n",cmdOut.toStdString().c_str());
#else

#endif
//    qDebug("command: %s\r\n",command.c_str());
    // Execute the ping command
    return (code == 0);
}
void ConnectionChecking::stop(){
    m_stop = true;
}

void ConnectionChecking::doWork(){
    while(m_stop == false){
        std::string details;
#ifdef __linux__
    //linux code goes here
        long mtime, seconds, useconds;
        struct timeval stop, start;
        gettimeofday(&start, NULL);
        bool result = Ping( m_address, 1, details );
        gettimeofday(&stop, NULL);
        seconds  = stop.tv_sec  - start.tv_sec;
        useconds = stop.tv_usec - start.tv_usec;
        mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
#elif _WIN32
    // windows code goes here
        QTime myTimer;
        myTimer.start();
        bool result = Ping( m_address, 1, details );
        int mtime = myTimer.elapsed();
#else

#endif
//        printf("mtime = %d\r\n",mtime);
        if(mtime < 1000){
            msleep(1000-mtime);
        }
        if (result == true ){
            stateChange("ok");
        }else{
            stateChange("lost");
        }
    }

}

void ConnectionChecking::msleep(int ms){
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
