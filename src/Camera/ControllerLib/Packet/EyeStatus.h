#ifndef EYESTATUS_H
#define EYESTATUS_H

#include "Common_type.h"

using namespace Eye;
namespace Status
{
    // IPC Status
    enum class SensorMode : unsigned char {
        EO = 0,
        IR = 1
    };

    enum class SensorColorMode : unsigned char {
        IR_WHITE_HOT = 0,
        IR_BLACK_HOT = 1,
        IR_REDDISK = 2,
        IR_COLOR = 3,
        EO_AUTO = 4,
        EO_COLOR = 5,
        EO_DAWN = 6,
        EO_DN = 7
    };

    enum class VideoStabMode : unsigned char {
        OFF = 0,
        ON = 1
    };

    enum class SnapShotMode : unsigned char {
        SNAP_ONE = 0,
        SNAP_THREE = 1,
        SNAP_TEN = 2,
        SNAP_FIFTY = 3
    };

    enum class CropMode : unsigned char {
        CROP_OFF = 0,
        CROP_5 =  1,
        CROP_10 =  2,
        CROP_15 = 3,
        CROP_20 = 4,
    };

    enum class InstallMode : unsigned char {
        MOUNT_BELL = 0,
        MOUNT_NOSE = 1,
    };

    typedef struct {
        data_type width;
        data_type height;
    } ImageSize;

    typedef struct {
        data_type MAX_HFOV = 63.4 / 180.0 * M_PI;
        data_type MAX_VFOV = 37.3 / 180.0 * M_PI;
        data_type MIN_HFOV = 2.3 / 180.0 * M_PI;
        data_type MIN_VFOV = 1.3 / 180.0 * M_PI;
        data_type freeMemory;
        data_type totalMemory;
        data_type hfov;
        data_type vfov;
        SensorColorMode sensorColorMode;
        SensorMode sensorMode = SensorMode::EO;
        VideoStabMode videoStabEn;
        SnapShotMode snapShotMode;
        ImageSize imageResize;
        ImageSize resolution;
        data_type cropMode; // percentage of image size that has been cropped : 0%, 5%, 10%, 20%
    } IPCStatus;


    // MotionC Status
    enum class StabMode : unsigned char {
        OFF = 0,
        ON = 1
    };

    enum class PresetMode : unsigned char {
        FREE = 0,
        FRONT = 1,
        RIGHT_WING = 2,
        NADIR = 3
    };

    typedef struct {
        data_type MAX_RATE = 120.0 / 180.0 * M_PI;
        StabMode panStabMode;
        StabMode tiltStabMode;
        data_type panCurrentVelocity;
        data_type tiltCurrentVelocity;
        data_type panCurrentPosition;
        data_type tiltCurrentPosition;
        PresetMode presetMode;
    } MotionCStatus;

    // System Status
    enum class StreamingProfile : unsigned char {
        PROFILE_STOP = 0,
        PROFILE_1080_4M = 1,
        PROFILE_1080_2M = 2,
        PROFILE_720_2M = 3,
        PROFILE_SYNC = 4
    };

    enum class LockMode : unsigned char {
        LOCK_OFF = 0,
        LOCK_GEO = 1,
        LOCK_TARGET = 2,
        LOCK_TRACK = 3,
        LOCK_VISUAL = 4
    };

    enum class GeolocationMode : unsigned char {
        GEOLOCATION_ON = 1,
        GEOLOCATION_OFF = 0
    };

    enum class RecordMode : unsigned char {
        RECORD_OFF = 0,
        RECORD_FULL = 1,
        RECORD_VIEW = 2,
        RECORD_TYPE_A = 3,
        RECORD_TYPE_B = 4
    };

    enum class MeasurementMode : unsigned char {
        RF_NONE = 0,
        RF_LRF = 1,
        RF_GPS = 2,
        RF_UAV = 3
    };

    enum class GimbalMode : unsigned char {
        OFF = 0,
        ON = 1,
        SLEEP = 2,
        SECURE = 3
    };

    typedef struct {
        LockMode lockMode;
        GeolocationMode geolocationMode;
        RecordMode recordMode;
        MeasurementMode measurementMode;
        GimbalMode gimbalMode;
        InstallMode installMode;
        StreamingProfile streamingProfile;
    } SystemStatus;

    typedef struct {
        data_type roll;
        data_type pitch;
        data_type yaw;
        data_type pn;
        data_type pd;
        data_type pe;
    } TelemetryStatus;

    class EyeStatus
    {
        public:
            EyeStatus() {}
            ~EyeStatus() {}

        public:
            IPCStatus IPC;
            MotionCStatus MotionC;
            SystemStatus System;
            TelemetryStatus Telemetry;
    };
}

#endif // EYESTATUS_H
