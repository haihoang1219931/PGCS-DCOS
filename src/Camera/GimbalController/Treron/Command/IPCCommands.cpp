#include "IPCCommands.h"
IPCCommands::IPCCommands(QObject *parent) : QObject(parent) {}
IPCCommands::~IPCCommands() {}
void IPCCommands::changeTrackSize(int size)
{
    //    printf("Change track size to %d\r\n",size);
    Eye::KLV packet;
    Eye::TrackSize data;
    data.setSize((data_type)size);
    packet.key = (key_type)EyePhoenixProtocol::TrackingSize;
    packet.data = data.toByte();
    vector<byte> packetEncoded = packet.encode();
    m_buffer->add(packetEncoded);
    m_buffer->send();
}
void IPCCommands::changeStreamProfile(QString profile)
{
    Eye::KLV packet;
    Eye::StreamingProfile data;

    if (profile == "1080_4M") {
        data.setProfile((byte)Status::StreamingProfile::PROFILE_1080_4M);
    } else if (profile == "1080_2M") {
        data.setProfile((byte)Status::StreamingProfile::PROFILE_1080_2M);
    } else if (profile == "720_2M") {
        data.setProfile((byte)Status::StreamingProfile::PROFILE_720_2M);
    } else if (profile == "STOP") {
        data.setProfile((byte)Status::StreamingProfile::PROFILE_STOP);
    } else if (profile == "SYNC") {
        data.setProfile((byte)Status::StreamingProfile::PROFILE_SYNC);
    }

    packet.key = (key_type)EyePhoenixProtocol::StreamingProfile;
    packet.data = data.toByte();
    vector<byte> packetEncoded = packet.encode();
    m_buffer->add(packetEncoded);
    m_buffer->send();
}
void IPCCommands::changeSensorID(QString sensorID)
{
    Eye::KLV packet;
    SensorID data;

    if (sensorID == "EO") {
        data.setSensorId((byte)Status::SensorMode::EO);
    } else if (sensorID == "IR") {
        data.setSensorId((byte)Status::SensorMode::IR);
    }

    packet.key = (key_type)EyePhoenixProtocol::SensorChange;
    packet.data = data.toByte();
    vector<byte> packetEncoded = packet.encode();
    m_buffer->add(packetEncoded);
    m_buffer->send();
}
void IPCCommands::changeLockMode(QString lockMode, QString geoLocation)
{
    Eye::KLV packet;
    Eye::LockMode data;
    byte _geoLocationMode, _lockMode;
    QString tmpLockMode = "FREE";

    if (geoLocation == "GEOLOCATION_ON") {
        _geoLocationMode = (byte)Status::GeolocationMode::GEOLOCATION_ON;
    } else {
        _geoLocationMode = (byte)Status::GeolocationMode::GEOLOCATION_OFF;
    }

    if (lockMode == "LOCK_VISUAL") {
        _lockMode = (byte)Status::LockMode::LOCK_VISUAL;
        tmpLockMode = "VISUAL";
    } else if (lockMode == "LOCK_TRACK") {
        _lockMode = (byte)Status::LockMode::LOCK_TRACK;
        tmpLockMode = "TRACK";
    } else if (lockMode == "LOCK_GEO") {
        _lockMode = (byte)Status::LockMode::LOCK_GEO;
        tmpLockMode = "GEO";
    } else if (lockMode == "LOCK_TARGET") {
        _lockMode = (byte)Status::LockMode::LOCK_TARGET;
        tmpLockMode = "TARGET";
    } else if (lockMode == "LOCK_FREE") {
        _lockMode = (byte)Status::LockMode::LOCK_OFF;
        tmpLockMode = "FREE";
    }

    data.setLockMode(_lockMode, _geoLocationMode);
    packet.key = (key_type)EyePhoenixProtocol::LockModeChange;
    packet.data = data.toByte();
    vector<byte> packetEncoded = packet.encode();
    m_buffer->add(packetEncoded);
    m_buffer->send();
}
void IPCCommands::takeSnapshot(QString mode, int frameID)
{
    Eye::KLV packet;
    Eye::Snapshot data;

    if (mode == "SNAP_ONE") {
        data.setMode((byte)Status::SnapShotMode::SNAP_ONE);
    } else if (mode == "SNAP_THREE") {
        data.setMode((byte)Status::SnapShotMode::SNAP_THREE);
    } else if (mode == "SNAP_TEN") {
        data.setMode((byte)Status::SnapShotMode::SNAP_TEN);
    } else if (mode == "SNAP_FIFTY") {
        data.setMode((byte)Status::SnapShotMode::SNAP_FIFTY);
    }

    data.setFrameID((index_type)frameID);
    packet.key = (key_type)EyePhoenixProtocol::TakeSnapShot;
    packet.data = data.toByte();
    vector<byte> packetEncoded = packet.encode();
    m_buffer->add(packetEncoded);
    m_buffer->send();
}
void IPCCommands::changeSensorColor(QString mode)
{
    Eye::KLV packet;
    Eye::SensorColor data;

    if (mode == "IR_WHITE_HOT") {
        data.setMode((byte)Status::SensorColorMode::IR_WHITE_HOT);
    } else if (mode == "IR_BLACK_HOT") {
        data.setMode((byte)Status::SensorColorMode::IR_BLACK_HOT);
    } else if (mode == "IR_REDDISK") {
        data.setMode((byte)Status::SensorColorMode::IR_REDDISK);
    } else if (mode == "IR_COLOR") {
        data.setMode((byte)Status::SensorColorMode::IR_COLOR);
    } else if (mode == "EO_AUTO") {
        data.setMode((byte)Status::SensorColorMode::EO_AUTO);
    } else if (mode == "EO_COLOR") {
        data.setMode((byte)Status::SensorColorMode::EO_COLOR);
    } else if (mode == "EO_DAWN") {
        data.setMode((byte)Status::SensorColorMode::EO_DAWN);
    } else if (mode == "EO_DAYNIGHT") {
        data.setMode((byte)Status::SensorColorMode::EO_DN);
    }

    packet.key = (key_type)EyePhoenixProtocol::ChangeSensorColor;
    packet.data = data.toByte();
    vector<byte> packetEncoded = packet.encode();
    m_buffer->add(packetEncoded);
    m_buffer->send();
}
void IPCCommands::takeRFRequest(QString mode, int frameID)
{
    Eye::KLV packet;
    RFRequest data;
    data.setFrameID((index_type)frameID);

    if (mode == "RF_NONE") {
        data.setMode(RF_NONE);
    } else if (mode == "RF_LRF") {
        data.setMode(RF_LRF);
    } else if (mode == "RF_GPS") {
        data.setMode(RF_GPS);
    } else if (mode == "RF_UAV") {
        data.setMode(RF_UAV);
    }

    packet.key = (key_type)EyePhoenixProtocol::RFRequest;
    packet.data = data.toByte();
    vector<byte> packetEncoded = packet.encode();
    m_buffer->add(packetEncoded);
    m_buffer->send();
}
void IPCCommands::changeGimbalMode(QString mode)
{
    Eye::KLV packet;
    Eye::GimbalMode data;

    if (mode == "GIMBAL_ON") {
        data.setMode((byte)Status::GimbalMode::ON);
    } else if (mode == "GIMBAL_SLEEP") {
        data.setMode((byte)Status::GimbalMode::SLEEP);
    } else if (mode == "GIMBAL_SECURE") {
        data.setMode((byte)Status::GimbalMode::SECURE);
    } else if (mode == "GIMBAL_OFF") {
        data.setMode((byte)Status::GimbalMode::OFF);
    }

    packet.key = (key_type)EyePhoenixProtocol::GimbalMode;
    packet.data = data.toByte();
    vector<byte> packetEncoded = packet.encode();
    m_buffer->add(packetEncoded);
    m_buffer->send();
}
void IPCCommands::changeRecordMode(QString mode, int type, int frameID)
{
    Eye::KLV packet;
    GimbalRecord data;
    data.setFrameID((index_type)frameID);

    if (mode == "RECORD_FULL") {
        data.setMode(RECORD_FULL);
    } else if (mode == "RECORD_OFF") {
        data.setMode(RECORD_OFF);
    } else if (mode == "RECORD_VIEW") {
        data.setMode(RECORD_VIEW);
    } else if (mode == "RECORD_TYPE_A") {
        data.setMode(RECORD_TYPE_A);
    } else if (mode == "RECORD_TYPE_B") {
        data.setMode(RECORD_TYPE_B);
    }

    data.setType(0x00);
    packet.key = (key_type)EyePhoenixProtocol::GimbalRecord;
    packet.data = data.toByte();
    vector<byte> packetEncoded = packet.encode();
    m_buffer->add(packetEncoded);
    m_buffer->send();
    //    m_gimbalModel->gimbal()->m_gimbalRecord = mode.toStdString();
}
void IPCCommands::setClickPoint(int frameID, double x, double y, double width,
                                double height, double objW, double objH)
{
    Eye::KLV packet;

    //    printf("\n====>Send TrackObj: %d, %f, %f, %f, %f, %f, %f", frameID, x, y,
    //           width, height, objW, objH);
    QString lockMode = m_gimbalModel->m_lockMode;
    if(lockMode == "VISUAL"){
        XPoint data(frameID,x,y,width,height);
        packet.key = (key_type)EyePhoenixProtocol::SceneSteering;
        packet.data = data.toByte();
    }else if(lockMode == "TRACK"){
        TrackObject data(frameID, x - objW/2, y - objH/2, width, height, objW, objH);
        packet.key = (key_type)EyePhoenixProtocol::Tracking;
        packet.data = data.toByte();
    }

    vector<byte> packetEncoded = packet.encode();
    m_buffer->add(packetEncoded);
    m_buffer->send();
}
void IPCCommands::setZoomPosition(QString mode, int factor)
{
    Eye::KLV packet;
    ZoomData data;

    if (mode == "ZOOM_IN") {
        data.setZoomData(ZOOM_IN, 0);
    } else if (mode == "ZOOM_OUT") {
        data.setZoomData(ZOOM_OUT, 0);
    } else if (mode == "ZOOM_STOP") {
        data.setZoomData(ZOOM_STOP, 0);
    } else if (mode == "ZOOM_FACTOR") {
        data.setZoomData(ZOOM_FACTOR, (data_type)factor);
    }

    packet.key = (key_type)EyePhoenixProtocol::EOOpticalZoom;
    packet.data = data.toByte();
    vector<byte> packetEncoded = packet.encode();
    m_buffer->add(packetEncoded);
    m_buffer->send();
}

void IPCCommands::enableImageStab(QString mode, double cropRatio)
{
    Eye::KLV packet;
    Eye::ImageStab data;
    bool _mode;
    if (mode == "ISTAB_ON") {
        data.setStabMode((byte)Status::StabMode::ON);
        _mode = true;
    } else{
        _mode = false;
        data.setStabMode((byte)Status::StabMode::OFF);
    }

    data.setCropRatio(cropRatio);
    packet.key = (key_type)EyePhoenixProtocol::ImageStab;
    packet.data = data.toByte();
    vector<byte> packetEncoded = packet.encode();
    m_buffer->add(packetEncoded);
    m_buffer->send();
    m_gimbalModel->m_videoStabMode = _mode;
}

void IPCCommands::doSceneSteering(int _index)
{
    Eye::KLV packet;
    Eye::SceneSteering data(_index);
    packet.key = (key_type)EyePhoenixProtocol::SceneSteering;
    packet.data = data.toByte();
    vector<byte> packetEncoded = packet.encode();
    m_buffer->add(packetEncoded);
    m_buffer->send();
    m_gimbalModel->m_lockMode = "VISUAL";
}
void IPCCommands::configCamera(QString funcCode, QString data)
{
    Eye::KLV packet;
    printf("configCamera [%s][%s]\r\n", funcCode.toStdString().c_str(),
           data.toStdString().c_str());
    std::string cmd = HConfigMessage::makeSetCommand(funcCode.toStdString(),
                      data.toStdString());
    HConfigMessage message(funcCode.toStdString(), cmd);
    packet.key = (key_type)EyePhoenixProtocol::HitachiConfig;
    packet.data = message.toByte();
    std::vector<unsigned char> packetEncoded = packet.encode();
    m_buffer->add(packetEncoded);
    m_buffer->send();
}
