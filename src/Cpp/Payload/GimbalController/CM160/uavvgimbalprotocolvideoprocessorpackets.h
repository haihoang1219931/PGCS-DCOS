#ifndef UAVVGIMBALPROTOCOLVIDEOPROCESSORPACKETS_H
#define UAVVGIMBALPROTOCOLVIDEOPROCESSORPACKETS_H

/**
*
* Author: hainh35
* Brief : Get and set information of video processing system
*
**/

#ifdef _WIN32
    #include <winsock.h>
    #include <windows.h>
    #include <time.h>
    #define PORT        unsigned long
    #define ADDRPOINTER   int*
    struct _INIT_W32DATA
    {
       WSADATA w;
       _INIT_W32DATA() { WSAStartup( MAKEWORD( 2, 1 ), &w ); }
    } _init_once;
#else       /* ! win32 */
    #include <unistd.h>
    #include <sys/time.h>
    #include <sys/types.h>
    #include <sys/socket.h>
    #include <netdb.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #define PORT        unsigned short
    #define SOCKET    int
    #define HOSTENT  struct hostent
    #define SOCKADDR    struct sockaddr
    #define SOCKADDR_IN  struct sockaddr_in
    #define ADDRPOINTER  unsigned int*
    #define INVALID_SOCKET -1
    #define SOCKET_ERROR   -1
#endif /* _WIN32 */

#include <stdio.h>
#include <iostream>

using namespace std;

#include <QObject>
#include <QUdpSocket>
#include <stdio.h>
#include <iostream>
#include <vector>

#include "videoprocessor/UavvChangeVideoRecordingState.h"
#include "videoprocessor/UavvCurrentImageSize.h"
#include "videoprocessor/UavvCurrentRecordingState.h"
#include "videoprocessor/UavvH264StreamParameters.h"
#include "videoprocessor/UavvModifyObjectTrack.h"
#include "videoprocessor/UavvModifyTrackIndex.h"
#include "videoprocessor/UavvNudgeTrack.h"
#include "videoprocessor/UavvOverlay.h"
#include "videoprocessor/UavvStabilisationParameters.h"
#include "videoprocessor/UavvStabiliseOnTrack.h"
#include "videoprocessor/UavvTakeSnapshot.h"
#include "videoprocessor/UavvTrackingParameters.h"
#include "videoprocessor/UavvVideoConfiguration.h"
#include "videoprocessor/UavvVideoDestination.h"

#include "system/UavvRequestResponse.h"

class UavvGimbalProtocolVideoProcessorPackets: public QObject
{
    Q_OBJECT
public:
    QUdpSocket *_udpSocket;
    UavvGimbalProtocolVideoProcessorPackets(QObject *parent = 0);
    Q_INVOKABLE void modifyObjectTrack(QString trackFlag,
                           int colCoordinate,
                           int rowCoordinate);
    Q_INVOKABLE void modifyTrackByIndex(TrackByIndexAction trackFlag,
                            unsigned char trackIndex);
    Q_INVOKABLE void movingTargetDetection();
    Q_INVOKABLE void nudgeTrack(unsigned char reverse,
                    char colPixelOffset,
                    char rowPixelOffset);
    Q_INVOKABLE void setTrackingParameters(int squareSideLength,
                               QString trackingMode);
    Q_INVOKABLE void getTrackingParameters();
    Q_INVOKABLE void setStabiliseOnTrack(bool enable);
    Q_INVOKABLE void getStabiliseOnTrack();
    Q_INVOKABLE void setEStabilisationParameters(
            bool enable,
            int reEnteringRate,
            int maxTranslationCompensation,
            int maxRotationalCompensation,
            int backgroundType);
    Q_INVOKABLE void getEStabilisationParameters();
    Q_INVOKABLE void setCurrentImageSize(unsigned char reverse,
                             unsigned short frameWidth,
                             unsigned short frameHeight);
    Q_INVOKABLE void getCurrentImageSize();
//    Q_INVOKABLE void setOverlay(bool show);
    Q_INVOKABLE void setOverlay(bool enableGimbalOverlay,
                                    bool enableLaserDevice,
                                    bool enableLimitWarning,
                                    bool enableGyroStabilization,
                                    bool enableGimbalMode,
                                    bool enableTrackingBoxes,
                                    bool enableHFOV,
                                    bool enableSlantRange,
                                    bool enableTargetLocation,
                                    bool enableTimestamp,
                                    bool enableCrosshair);
    Q_INVOKABLE void getOverlay();
    Q_INVOKABLE void setVideoDestination(
                             unsigned int destinationIP,
                             unsigned short destinationPort,
                             bool enableNetworkStream,
                             bool enableAnalogOut);
    Q_INVOKABLE void getVideoDestination();
    Q_INVOKABLE void setH264StreamParameters(
            unsigned int streamBitrate,
            unsigned char frameInterval,
            unsigned char frameStep,
            unsigned char downSampleFrame,
            unsigned char reverse);
    Q_INVOKABLE void getH264StreamParameters();
    Q_INVOKABLE void changeVideoRecordingState(bool start);
    Q_INVOKABLE void getCurrentRecordingState();
    Q_INVOKABLE void setVideoConfiguration(
            VideoConfigurationEncoderType encoderType,
            VideoConfigurationOutputFrameSize sensor0FrameSize,
            VideoConfigurationOutputFrameSize sensor1FrameSize);
    Q_INVOKABLE void getVideoConfiguration();
    Q_INVOKABLE void takeSnapshot();
public:
    bool isInitialized = false;
    SOCKET m_udpSocket;
    SOCKADDR_IN m_udpAddress;
    std::string m_ip;
};

#endif // UAVVGIMBALPROTOCOLVIDEOPROCESSORPACKETS_H
