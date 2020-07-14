#ifndef EYEPHOENIXPROTOCOL_H
#define EYEPHOENIXPROTOCOL_H

enum class EyePhoenixProtocol
{
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
    Joystick = 0x0013,
    JoystickFree = 0x0014,
    JoystickVisual = 0x0015,
    JoystickTrack = 0x0016,
    JoystickGeo = 0x0017,

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
    MotionAngle = 0x040A,
    MotionStab = 0x0403,
    MotionCalib = 0x0404,
    MotionRequest = 0x0405,
    MotionAnglePointing = 0x0406,
    MotionAngleSteering = 0x0407,
    MotionAngleGeoLock = 0x0408,
    MotionAngleTracking = 0x0409,
    MotionLimitAngle = 0x0402,
    MotionSetParams = 0x040B,
    MotionPanSetParams = 0x040C,
    MotionTiltSetParams = 0x040D,

    // IPC Data
    IPCStatusResponse = 0x0200,
    EOMotionDataResponse = 0x0201,
    IRMotionDataResponse = 0x0204,
    EOSteeringResponse = 0x0202,
    IRSteeringResponse = 0x0205,
    EOTrackingResponse = 0x0203,
    IRTrackingResponse = 0x0206,

    //AP data
    ExternalAttitude = 0xBD,
    ExternalPosition = 0xBE,
    Telemetry = 0x0101,

    //GCS
    TargetPosition = 0x0500,
    MotionCStatus = 0x0501,
    GeolockInfo = 0x0502,

    //Hitachi Configuration
    HitachiConfig = 0x0606,
    GLConfig = 0x0607,

    //IMU
    IMUData = 0x0701,
    IMUReset = 0x0761,
    IMUCalib = 0x0763
};

#endif // EYEPHOENIXPROTOCOL_H
