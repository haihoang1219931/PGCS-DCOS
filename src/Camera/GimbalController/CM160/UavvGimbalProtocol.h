//#pragma once
#ifndef UAVVGIMBALPROTOCOL_H
#define UAVVGIMBALPROTOCOL_H
enum class UavvGimbalProtocol
{
    GyroStablisation = 0,
    MessageAcknowledgement = 1,
    EnableMessageAcknowledgment = 2,
    QueryGimbalSerial = 3,
    GimbalStatus = 4,
    PositionStreaming = 5,
    Version = 6,
    ProtocolVersion = 7,
    DetailedVersion = 8,
    RequestPacket = 9,
    ConfigurePacketRates = 10,
    StowConfiguration = 11,
    SetStowMode = 12,
    InitialiseGimbal = 13,
    SetUnixTime = 14,
    Reset = 15,
    NetworkConfiguration = 16,
    SceneSteering = 17,
    SceneSteeringConfig = 18,
    CombinedPositionVelocityState = 19,
    SetPanPosition = 20,
    GimbalMovement = 21,
    SetPanandTiltPosition = 22,
    SetPanandTilVelocity = 23,
    SetTiltPosition = 25,
    MaxVelocityDemand = 26,
    PanTrim = 27,
    StepStareParameters = 28,
    StepStareCommand = 29,
    SensorEnable = 30,
    EnableDigitalZoom = 31,
    SetDigitalZoomSeparateMode = 32,
    SetCameraDigitalZoomPosition = 33,
    SetCameraDigitalZoomVelocity = 34,
    SetCameraZoomVelocity = 35,
    SetCameraZoomPosition = 37,
    EnableAutoFocus = 40,
    ICRMode = 41,
    HighSensitivityMode = 42,
    GenericSonycommand = 43,
    SensorDefog = 44,
    SetFocus = 45,
    EnableManualIris = 50,
    SetIris = 55,
    EOFStop = 56,
    EnableLensStabilisation = 60,
    EoExposureMode = 61,
    SetInvertPiture = 65,
    SetShutterSpeed = 70,
    SetCameraGain = 75,
    SetEOSensorVideoMode = 76,

    LaserRange = 80,
    LaserRangeStart = 81,
    LaserRangeStatus = 82,
    ArmLaserDevice = 0x55,
    FireLaserDevice = 0x56,
    LaserDeviceStatus = 0x57,

    SetPanVelocity = 90,
    SetTiltVelocity = 95,
    EnableManualShutter = 100,
    EnableAutoExposure = 105,


    VideoConfiguration = 107,
    ModifyObjectTrack = 108,
    ModifyTrackByIndex = 109,
    EnableTracker = 110,
    NudgeTrack = 111,
    SetTrackingParameters = 112,
    StabililseOnTrack = 113,
    SetStabilisationParameters = 114,
    ImageSize = 115,
    EnableMotionDetectionParameters = 116,
    SmallTargetDetectionParameters = 117,
    SetOverlayMode = 118,
    SetVideoOut = 119,
    SetVideoDestination = 120,
    SetH264Parameters = 121,
    ChangeRecordingStatus = 122,
    RecordingStatus = 123,
    TrackingStatus = 124,
    SetupSnapshots = 125,
    TakeSnapshot = 126,

    QueryPanPosition = 0x82,
    GimbalSerialNumberReply = 0x83,
    QueryTiltPosition = 0x87,
    SensorCurrentFocus = 136,
    SensorZoom = 137,  //(0x89)
    SensorFocus = 138,
    SensorFieldOfView = 139,
    QueryZoomPosition = 0x8C,
    CameraTemperature = 0x8D,
    IRSensorTemperature = 0x8E,
    QueryPanVelocity = 0x91,
    SetIRVideoStandard = 0x92,
    Flirpassthrough = 0x94,
    SetIRZoom = 0x95,
    QueryTiltVelocity = 0x96,
    ZoomPositionResponse = 0x9B,
    SetIRAGCMode = 0x9C,
    SetIRBrightness = 0x9D,
    EnableIsotherm = 0x9E,
    SetIsothermThresholds = 0x9F,
    CameraInitialisation = 0xA0,
    ResetIRCamera = 0xA1,
    SetIRFCCMode = 0xA2,
    PerformFFC = 0xA3,
    SetIRFCCTemperature = 0xA4,
    SetIRZoomDeprecated = 0xA5, //deprecated
    SetIRFreezeFrameDeprecated = 0xA6, //deprecated
    SetIRPalette = 0xA7,
    SetMWIRTempPreset = 0x86,
    SetIRVideoOrientation = 0xA8,
    GetIRTemperature = 0xA9, //deprecated
    ToggleVideoOutput = 171,
    SetIRContrast = 0xAC,
    SetIRBrightnessBias = 0xAD,
    SetIRPlateauLevel = 0xAE,
    SetIRITTMid = 0xAF,
    SetIRMAXAGC = 0xB0,
    SetIRGainMode = 0xB1,
    SetIRDDE = 0xB2,

    PointOfInterest = 182,

    MagneticCalibration = 183,
    MagneticCalibrationProgress = 184,
    SetGeolock = 185,
    CurrentGeolockSetpoint = 186,
    CurrentTargetLocation = 187,
    CurrentCornerLocations = 188,
    ExternalAltitude = 189,
    ExternalPosition = 190,
    SeedTerrainHeight = 192,

    ImuStatus = 193,
    GPSSatillities = 194,
    PlatformOrientation= 195,
    PlatformPosition= 196,
    UTCTime= 197,
    PlatformOrientationOffset= 198,
    AltitudeOffset= 199,
    IMUTranslationOffset= 200,
    GimbalMisalignmentOffset = 201,
    TargetHeightOffset,
    SaveOffsets = 203,
    AhrsConfiguration = 204,
    AhrsAlignment = 205,
    RTCMCorrections = 206,
    GimbalControlState = 207,
    RawSensorData = 208,
    DirectMotorDrive = 209,
    GvpConfig = 210,
    GeoRefInputs = 211,
    FrameTranslation = 212,
    ObjectTrackingTelemetry = 214,

    PassthroughConfiguration = 220,
    Passthrough = 221,

    FactorySettingsSetHardwareVersion = 229,
    FactorySettingsSensorPayload = 230,
    FactorySettingsMovementLimits = 231,
    FactorySettingsEncoderSettings = 232,
    FactorySettingsMotorSettings = 233,
    FactorySettingsGyroSettings = 234,
    FactorySettingsSetSerialNumber = 235,
    FactorySettingsUserSettings = 236,
    FactorySettingsGains = 237,

    BoresightCalibrationRecord = 238,
    BoresightCalibrationLookupTable = 239,
    BoresightCalibrationAckNack = 240,
    ReadGyroTemperatureCalibrationRecord = 241,
    EnableVideoProcessor = 242,
    FlashSaveResult = 243,
    GyroStream = 244,
    ObjectTrackerGains = 245,
    GyroTemperatureCalibrationRecord = 246,
    EnterFactoryMode = 247,
    SensorCalibrationCommands = 248,
    SetStartUpOptions = 249,
    RawDataStream = 250,
    Gains = 251,
    EnterBootMode = 254,
    BootLoader = 255
};
enum class CurrentGimbalMode
{
    Unarmed,
    ObjectTracking,
    Arming,
    RateControl,
    PositionControl,
    Stowed,
    SceneSteering,
    Geolocking,
    Reserved,
    PerformingFFC,
};
enum class ActionType
{
    StartGimbalTracking = 0x80,
    AddTrackCoordinates = 0x00,
    AddTrackCrosshair = 0x01,
    DesignatesTrack = 0x02,
    AddTrackPrimary = 0x03,
    StopGimbalTrack = 0x04,
    KillTrackNearest = 0x05,
    KillAllTracks = 0x06,
    KillAllPrimary = 0x07,
    KillAllCoordinates = 0x08,
    KillAllCrosshair = 0x09
};
enum class GimbalAxis
{
    Pan = 0,
    Tilt,
    Roll
};

enum class PayloadSerialPort : unsigned char
{
    None,
    SerialPort1,
    SerialPort2,
    SerialPort3,
    SerialPort4,
    SerialPort5,
    SerialPort6,
    SerialPort7,
    SerialPort8,
    VideoProcessorAPort1,
    VideoProcessorAPort2,
    VideoProcessorBPort1,
    VideoProcessorBPort2
};

enum class PayloadSerialBaudRate : unsigned char
{
    Baud4800N1,
    Baud9600N1,
    Baud38400N1,
    Baud57600N1,
    Baud115200N1,
    Baud230400N1,
    Baud468000N1
};

enum class SensorPayloadType : unsigned char
{
    None = 0,
    SonyH11 = 1,
    Sony980 = 2,
    Sony3400 = 3,
    Sony6300 = 4,
    HitachiSC110 = 5,
    HitachiSC120 = 6,
    SonyEV7500 = 7,
    SonyEV7500_1080 = 8,
    SonyEV7500_720 = 9,

    FlirPhoton = 10,
    FlirTau = 11,
    FlirTau2 = 12,
    FlirQuark19mm = 13,
    FlirQuark2 = 14,
    FlirQuark25mm,
    FlirNeutrino = 20,
    FlirTauTamronSc100 = 21,
    FlirTauTamronLens = 21
};

enum class SensorLensType : unsigned char
{
    None = 0,
    Stingray2351A01 = 1,
    OphirSupir100mmFixedFocus = 2,
    TamronSC001 = 3,
    FlirFixedFocus19mm = 4,
    FlirFixedFocus25mm = 5,
    FlirFixedFocus35mm = 6,
    FlirFixedFocus50mm = 7,
    FlirFixedFocus100mm = 8,

    Navitar2xTeleExtender119721 = 30
};

enum class LaserRangeFinderType : unsigned char
{
    None = 0,
    TruSenseS100 = 1,
    TruSenseS200 = 2
};

enum class LaserIlluminatorType : unsigned char
{
    None = 0,
};

enum class LaserDesignatorType : unsigned char
{
    None = 0,
};


enum class LaserPointerType : unsigned char
{
    None = 0,
    Nanopoint100 = 1
};

enum class GimbalType : unsigned char
{
    GD170,
    CM160,
    CM100,
    CM202,
    ST100,
    Unknown = 255
};

enum class ControlModeType : unsigned char
{
    Position,
    Velocity,
    GyroStabilised,
    Direct,
    Stow,
    Uninitialised,
    FFC,
    Undefined
};

//region GyroCalibration
enum class GyroCalibrationChannel
{
    PanBias = 0,
    TiltBias,
    PanVariance,
    TiltVariance,
    NumPanDataPoints,
    NumTiltDataPoints,
    StartTemperature,
    CalibrationTableSize
};

enum class GyroCalibrationCommand
{
    StartGyroCalibration = 0,
    StopGyroCalibration,
    WriteGyroCalibrationBiasToFlash,
    StartMisalignmentCalibration,
    MotorCalibration = 5
};
//endregion

/// <summary>
/// Different baud rates gibmal can be set to communicate on
/// </summary>
enum class BaudRateOptions : unsigned char
{
    Baud9600 = 0x00,
    Baud19200,
    Baud38400,
    Baud57600,
    Baud115200,
    Baud230400,
    Baud256000,
    Baud460800,
    Baud512000
};

/// <summary>
/// Different options which can be changed from default values
/// </summary>
enum class StartupOptions : unsigned char
{
    ResetDefaults = 0x00,
    AutoBoot = 0x01,
    BaudRate = 0x02,
    EncoderResolution = 0x03,
    ControlMode = 0x04,
    PanLimit = 0x05,
    TiltLimit,
    StowPosition,
    SetUserSettings = 0xfe,
    WriteUserSettings = 0xff
};

enum class GainTypes : unsigned char
{
    PanGyro = 0,
    TiltGyro,
    PanVelocity,
    TiltVelocity,
    PanPosition,
    TiltPosition,
    ObjectTracking,
    RollGyro,
    RollVelocity,
    RollPosition
};
enum class ImageSensorType  : unsigned char
{
    EOSensor = 0x00,
    IRSensor = 0x01
};
enum class ZoomStatus
{
    Stop,
    Zoomin,
    Zoomout
};
enum class SensorStatus
{
    Disable,
    Enable
};
enum class FlagFog
{
    FDisable=0x00,
    EnableAuto=0x01,
    EnableManual=0x02
};

enum class StrengthFog
{
    SDisable=0x00,
    Low=0x01,
    Medium=0x02,
    High=0x03
};
enum class DigitalZoomStatus
{
    Disable,
    Enable
};
enum class InfraredCutStatus
{
    Disable,
    Enable
};
enum class LensStabilizationStatus
{
    Disable,
    Enable
};
enum class ManualIrisStatus
{
    Disable,
    Enable
};
enum class AutoFocusStatus
{
    Disable,
    Enable
};
enum class ExposureMode
{
    Automatic,
    IrisPriority,
    ShutterPriority,
    Manual
};
enum class InvertMode
{
    Normal,
    Invert
};
enum class ManualShutterStatus
{
    Disable,
    Enable
};
enum class ExposureStatus
{
    Disable,
    Enable
};
enum class DefogFlag
{
    DisableDefog,
    EnableAuto,
    EnableManual,
};
enum class DefogStrength
{
    Disabled,
    LowStrength,
    MediumStrength,
    HighStrength,
};
enum class TrackingParametersAction
{
    NoChange = 0x00,
    StationaryObject = 0x01,
    MovingObjects = 0x02
};
enum class VideoConfigurationEncoderType
{
    H264Legacy = 0x00,
    H264 = 0x01,
    Mpeg4 = 0x02
};

enum class VideoConfigurationOutputFrameSize
{
    SD = 0x00,
    Size960x720 = 0x01,
    Size720 = 0x02
};
enum class VideoModulation
{
    PAL,
    NTSC
};
enum class ZoomFlag
{
    x1 = 1,
    x1Freeze = 2,
    x2 = 3,
    x2Freeze = 4,
    x4 = 5,
    x4Freeze = 6,
    x8 = 7,
    x8Freeze = 8
};
enum class AGCMode
{
    Auto,
    OnceBright,
    AutoBright,
    Manual,
    Linear
};
enum class IsothermStatus
{
    Disable,
    Enable
};
enum class FFCMode
{
    Sensor,
    Operator
};
enum class PaletteMode
{
    Whitehot,
    Blackhot,
    Fusion,
    Rain
};
enum class VideoOrientationMode
{
    Normal,
    Invert,
    Revert,
    BothIR
};
enum class GainMode
{
    Auto,
    Low,
    High
};
enum class ManualDDEStatus
{
    Disable,
    Enable
};
enum class GeoLockActionFlag{
    DisableGeoLock = 0x00,
    EnableGeoLockAtCrossHair = 0x01,
    EnableGeoLockAtCoordinateGimbal = 0x02
};
struct UserSettings
{
public:
    bool autoBoot;
    bool AutoEnableGyroPan;
    bool AutoEnableGyroTilt;
    bool AutoEnablePositionStreaming;
    BaudRateOptions Baud;
    bool EnableSceneSteeringOnBoot;
    bool EnableAutoSceneSteeringOnBoot;

    //public ControlMode controlMode; //not yet implemented
    double PanLimitLeft;
    double PanLimitRight;
    double TiltLimitUp;
    double TiltLimitDown;
    double StowPositionTilt;
    double StowPositionPan;
    bool invertSensor0;
    bool invertSensor1;
};



enum class VideoResolution : unsigned char
{
    PAL_1080i_60,
    PAL_1080p_60,
    NTSC_1080i_59_94,
    NTSC_1080p_59_94,
    PAL_1080i_50,
    PAL_1080p_50,
    PAL_1080p_30,
    NTSC_1080p_29_97,
    PAL_1080p_25,
    PAL_720p_60,
    NTSC_720p_59_94,
    PAL_720p_50,
    PAL_720p_30,
    NTSC_720p_29_97,
    PAL_720p_25,
    NTSC_Crop,
    NTSC_Squeeze,
    PAL_Crop,
    PAL_Squeeze
};

#endif
