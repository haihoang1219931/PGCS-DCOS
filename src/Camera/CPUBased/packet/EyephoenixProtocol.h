#ifndef EYEPHOENIXPROTOCOL_H
#define EYEPHOENIXPROTOCOL_H

enum class EyePhoenixProtocol{
    //MotionC
    SetPanTiltRateFactor = 0x0001,
    SetPanTiltAngle = 0x000E,
    SetPanTiltAngleDiff = 0xFFF1,
    SetRapidView = 0x0005,
    GimbalStab = 0x000B,
    SetCalib = 0x000D,
    // IPC
    SensorChange = 0x0006,
    LockModeChange = 0x0007,
    TakeSnapShot = 0x0004,
    ChangeSensorColor = 0x0008,
    RFRequest = 0xFFF2,
    GimbalRecord = 0x0009,
    ScreenPress = 0x0002,
    EOOpticalZoom = 0x0000,
    E0DigitalZoom = 0xFFF3,
    EOZoomStatus = 0x0100,
    RFResponse = 0xFFF4,
    GimbalMode = 0x000A,
    ImageStab = 0xFFF5,
    GPSData = 0x0003,
    SceneSteering = 0x000F,
    Tracking = 0x0010,
    GeoLocation = 0x0011,
    TrackingSize = 0x0012,

    // System
    RequestResponse = 0xFFF6,
    CameraInstall = 0xFFF7,
    SystemStatus = 0x000C,
    Confirm = 0x010A,
    ErrorMessage = 0x010B,
    StreamingProfile = 0x010C,
    GLValid = 0x010D,
    GLInvalid = 0x010E,

    // Control Pan/Tilt
    MotionRate = 0x0401,
    MotionAngle = 0x0402,
    MotionStab = 0x0403,
    MotionCalib = 0x0404,
    MotionRequest = 0x0405,

    // IPC Data
    IPCStatusResponse = 0x0200,
    MotionDataResponse = 0x0201,
    SteeringResponse = 0x0202,

    //AP data
    ExternalAttitude = 0xBD,
    ExternalPosition = 0xBE,
    Telemetry = 0x0101,

    //GCS
    TargetPosition = 0x0500,
    MotionCStatus = 0x0501,
    GeolockInfo = 0x0502,

    //Hitachi Configuration
    HitachiConfig = 0x0606
};

#endif // EYEPHOENIXPROTOCOL_H
