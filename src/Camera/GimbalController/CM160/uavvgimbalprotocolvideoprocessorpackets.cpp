#include "uavvgimbalprotocolvideoprocessorpackets.h"

UavvGimbalProtocolVideoProcessorPackets::UavvGimbalProtocolVideoProcessorPackets(QObject* parent) :
    QObject(parent)
{
    /*
    m_udpSocket = socket(PF_INET, SOCK_DGRAM, 0);
    m_udpAddress.sin_family = AF_INET;
    m_udpAddress.sin_port = htons(18001);
    m_udpAddress.sin_addr.s_addr = inet_addr((const char*)("192.168.1.114"));
    */

}
void UavvGimbalProtocolVideoProcessorPackets::modifyObjectTrack(QString trackFlag,
                       int colCoordinate,
                       int rowCoordinate){
    UavvModifyObjectTrack protocol;
    if(trackFlag == "first"){
        protocol.action = ActionType::StartGimbalTracking;
    }else if(trackFlag == "second"){
        protocol.action = ActionType::AddTrackCoordinates;
    }else if(trackFlag == "center"){
        protocol.action = ActionType::AddTrackCrosshair;
    }else if(trackFlag == "exit"){
        protocol.action = ActionType::KillAllTracks;
    }
    protocol.Column = (unsigned short)colCoordinate;
    protocol.Row = (unsigned short)rowCoordinate;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    //sendto(m_udpSocket,packet.data(),packet.size(),0,(struct sockaddr *)&m_udpAddress,sizeof(m_udpAddress));
    // Data send:0x24 0x40 0x05 0x6c 0x88 0x03 0xe4 0x01 0x0b 0x17
    _udpSocket->write((const char*)packet.data(),packet.size());
    // Data send:0x24 0x40 0x05 0x6c 0x88 0x02 0xa8 0x01 0x68 0xf6
    //_udpSocket->writeDatagram((const char*)packet.data(),packet.size(),QHostAddress("192.168.1.114"),18002);
}
void UavvGimbalProtocolVideoProcessorPackets::modifyTrackByIndex(TrackByIndexAction trackFlag,
                        unsigned char trackIndex){
    UavvModifyTrackIndex protocol;
    protocol.Action = trackFlag;
    protocol.Index = trackIndex;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolVideoProcessorPackets::movingTargetDetection(){}
void UavvGimbalProtocolVideoProcessorPackets::nudgeTrack(unsigned char reverse,
                char colPixelOffset,
                char rowPixelOffset){
    UavvNudgeTrack protocol;
    protocol.Column = colPixelOffset;
    protocol.Row = rowPixelOffset;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolVideoProcessorPackets::setTrackingParameters(int squareSideLength,
                           QString trackingMode){
    UavvTrackingParameters protocol;
    protocol.Acquisition = (unsigned char)squareSideLength;
    printf("setTrackingParameters [%d-%s]\r\n",
           protocol.Acquisition,trackingMode.toStdString().c_str());
    if(trackingMode == "stationary"){
        protocol.Mode = TrackingParametersAction::StationaryObject;
    }else if(trackingMode == "nochange"){
        protocol.Mode = TrackingParametersAction::NoChange;
    }else if(trackingMode == "moving"){
        protocol.Mode = TrackingParametersAction::MovingObjects;
    }
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolVideoProcessorPackets::getTrackingParameters(){
    printf("get Tracking Parameters\r\n");
    UavvRequestResponse protocol;
    protocol.PacketID = (unsigned char) UavvGimbalProtocol::SetTrackingParameters;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolVideoProcessorPackets::setStabiliseOnTrack(bool enable){
    UavvStabiliseOnTrack protocol;
    protocol.Enable = enable;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    printf("Send %d bytes: ",packet.size());
    for(int i=0; i< packet.size(); i++){
        printf("%02X ",packet[i]);
    }
    printf("\r\n");
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolVideoProcessorPackets::getStabiliseOnTrack(){
    UavvRequestResponse protocol;
    protocol.PacketID = (unsigned char) UavvGimbalProtocol::StabililseOnTrack;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolVideoProcessorPackets::setEStabilisationParameters(
        bool enable,
        int reEnteringRate,
        int maxTranslationCompensation,
        int maxRotationalCompensation,
        int backgroundType){
    UavvStabilisationParameters protocol;
    protocol.Enable = enable==true?0x01:0x00;
    protocol.Rate = (unsigned char)reEnteringRate;
    protocol.MaxTranslation = (unsigned char)maxTranslationCompensation;
    protocol.MaxRotational = (unsigned char)maxRotationalCompensation;
    protocol.Background = (unsigned char)backgroundType;
    vector<unsigned char> packet = protocol.Encode().encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolVideoProcessorPackets::getEStabilisationParameters(){
    UavvRequestResponse protocol;
    protocol.PacketID = (unsigned char) UavvGimbalProtocol::SetStabilisationParameters;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolVideoProcessorPackets::setCurrentImageSize(unsigned char reverse,
                         unsigned short frameWidth,
                         unsigned short frameHeight){

}
void UavvGimbalProtocolVideoProcessorPackets::getCurrentImageSize(){
    UavvRequestResponse protocol;
    protocol.PacketID = (unsigned char) UavvGimbalProtocol::ImageSize;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolVideoProcessorPackets::setOverlay(
                                            bool enableGimbalOverlay,
                                            bool enableLaserDevice,
                                            bool enableLimitWarning,
                                            bool enableGyroStabilization,
                                            bool enableGimbalMode,
                                            bool enableTrackingBoxes,
                                            bool enableHFOV,
                                            bool enableSlantRange,
                                            bool enableTargetLocation,
                                            bool enableTimestamp,
                                            bool enableCrosshair){
    UavvOverlay protocol;
    protocol.Reserved = 1;
    protocol.EnableGlobalOverlays = enableGimbalOverlay;
    protocol.EnableLaserArm = enableLaserDevice;
    protocol.EnableLimitWarning = enableLimitWarning;
    protocol.EnableGyroStabilisationMode = enableGyroStabilization;
    protocol.EnableGimbalMode = enableGimbalMode;
    protocol.EnableTrackingBoxes = enableTrackingBoxes;
    protocol.EnableHorizontalField = enableHFOV;
    protocol.EnableSlantRange = enableSlantRange;
    protocol.EnableTargetLocation = enableTargetLocation;
    protocol.EnableTimeStamp = enableTimestamp;
    protocol.EnableCrosshair = enableCrosshair;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
//void UavvGimbalProtocolVideoProcessorPackets::setOverlayItem(
//        QString itemName, bool visible){
//    UavvOverlay protocol;
//    protocol.Reserved = 1;
//    protocol.EnableGlobalOverlays = true;
//    protocol.EnableLimitWarning = false;
//    protocol.EnableGyroStabilisationMode = false;
//    protocol.EnableGimbalMode = false;
//    protocol.EnableTrackingBoxes = itemName;
//    protocol.EnableHorizontalField = false;
//    protocol.EnableSlantRange = false;
//    protocol.EnableTargetLocation = false;
//    protocol.EnableTimeStamp = false;
//    protocol.EnableCrosshair = false;
//    GimbalPacket payload = protocol.Encode();
//    vector<unsigned char> packet = payload.encode();
//    _udpSocket->write((const char*)packet.data(),packet.size());
//}
void UavvGimbalProtocolVideoProcessorPackets::getOverlay(){
    UavvRequestResponse protocol;
    protocol.PacketID = (unsigned char) UavvGimbalProtocol::SetOverlayMode;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolVideoProcessorPackets::setVideoDestination(
                         unsigned int destinationIP,
                         unsigned short destinationPort,
                         bool enableNetworkStream,
                         bool enableAnalogOut){
    UavvVideoDestination protocol;
    protocol.IPAddress = destinationIP;
    protocol.Port = destinationPort;
    protocol.NetworkStream = enableNetworkStream == true?0x01:0x00;
    protocol.IPAddress = enableAnalogOut == true?0x01:0x00;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolVideoProcessorPackets::getVideoDestination(){
    UavvRequestResponse protocol;
    protocol.PacketID = (unsigned char) UavvGimbalProtocol::SetVideoDestination;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolVideoProcessorPackets::setH264StreamParameters(
        unsigned int streamBitrate,
        unsigned char frameInterval,
        unsigned char frameStep,
        unsigned char downSampleFrame,
        unsigned char reverse){
    UavvH264StreamParameters protocol;
    protocol.BitRate = streamBitrate;
    protocol.FrameInterval = frameInterval;
    protocol.FrameStep = frameStep;
    protocol.DownSample = downSampleFrame;
    protocol.Reserved = reverse;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolVideoProcessorPackets::getH264StreamParameters(){
    UavvRequestResponse protocol;
    protocol.PacketID = (unsigned char) UavvGimbalProtocol::SetH264Parameters;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolVideoProcessorPackets::changeVideoRecordingState(
        bool start){
    UavvChangeVideoRecordingState protocol;
    protocol.Recording = start == true?0x01:0x00;
    protocol.Reserved = 0x00;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolVideoProcessorPackets::getCurrentRecordingState(){
    UavvRequestResponse protocol;
    protocol.PacketID = (unsigned char) UavvGimbalProtocol::RecordingStatus;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolVideoProcessorPackets::setVideoConfiguration(
    VideoConfigurationEncoderType encoderType,
    VideoConfigurationOutputFrameSize sensor0FrameSize,
    VideoConfigurationOutputFrameSize sensor1FrameSize){
    UavvVideoConfiguration protocol;
    protocol.EncoderType = encoderType;
    protocol.Sensor0 = sensor0FrameSize;
    protocol.Sensor1 = sensor1FrameSize;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolVideoProcessorPackets::getVideoConfiguration(){
    UavvRequestResponse protocol;
    protocol.PacketID = (unsigned char) UavvGimbalProtocol::VideoConfiguration;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
void UavvGimbalProtocolVideoProcessorPackets::takeSnapshot(){
    UavvTakeSnapshot protocol;
    GimbalPacket payload = protocol.Encode();
    vector<unsigned char> packet = payload.encode();
    _udpSocket->write((const char*)packet.data(),packet.size());
}
