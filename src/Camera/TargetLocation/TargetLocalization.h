/**
*
* @clacc TargetLocalization
*
* Class estimate the gps position of target from the camera image.
*
* 2020 created by Dongnt11 @viettel.com.vn
*/

#ifndef TARGETLOCALIZATION_H
#define TARGETLOCALIZATION_H
#include <math.h>
#define FLT_MAX         3.402823466e+38F        /* max value */
#define PI  3.14159265358979323846f
#define PI2  PI*0.5f
#define RAD_2_DEG  180.0f / PI
#define DEG_2_RAD  PI / 180.0f

/*
* Input of this function is the coordination of object by (m) on image with the origin of coordination at center of image
*/

class TargetLocalization
{
public:
    // GPS Position
    struct GpsPosition
    {
        double Latitude;
        double Longitude;
        float Altitude;

        int _latitude;
        int _longitude;
    };

    // Target Position in the NED coordination of UAV
    struct UavNEDcoord
    {
        float x_local;
        float y_local;
        float z_local;
    };

    // check valid data for a number
    static bool isValid (float f)
    {
        return (fabsf(f) < FLT_MAX);
    }
    static bool assure (float& f, float altValue)
    {
        if (isValid (f))
            return true;

        f = altValue;
        return false;
    }

    // Data from UAv for calculation
    struct UavData
    {
        struct GpsPosition UavPosition, TargetPosition;

        // attitude of UAV
        float Roll_rad;
        float Pitch_rad;
        float Yaw_rad;

        // attitude of Camera
        float Pan_rad;
        float Tilt_rad;

        // image depth
        float imageDepth;
        float imageWidth;	// width of image frame [pixel]
        float imageHeight;  // height of image frame [pixel]

        // Pixel Size
        float pixelSize; // [mm]

        // optical length
        float opticalLength; // [m]
        float hfov; // [rad]

        // Ti le dieu chinh anh phuong X va Y
        float Sx; // [mm/pixel]
        float Sy; // [mm/pixel]

        // Offsets of optical center of camera
        int offset_X; // [pixels]
        int offset_Y; // [pixels]

        // target position in uav and camera coordination
        struct UavNEDcoord targetUav, targetCam;

        int initialized;
    };

    struct UavData currentUavData;
    void targetLocationMain (float x_im, float y_im,
                             float hfov,
                             float roll_rad,
                             float pitch_rad,
                             float yaw_rad,
                             float pan_rad,
                             float tilt_rad,
                             double uav_lat,
                             double uav_long,
                             float uav_alt,
                             double &target_lat, double &target_long);

private:
    struct transformationMatrix3x3
    {
        float R11;
        float R12;
        float R13;
        float R21;
        float R22;
        float R23;
        float R31;
        float R32;
        float R33;
    };

    struct vector3x1
    {
        float V11;
        float V21;
        float V31;
    };

    static const float scaleFactor;     // scale factor for internal use
    static const int scaleFactorI;      //  scale factor for internal use - used in expressions with int type to avoid conversion
    static const double scaleFactorD;   //  scale factor for internal use - used in expressions with double type to avoid conversion
    static const float EARTH_RADIUS;    //  Earth radius in meters.

    struct vector3x1 positionCam, positionNED, positionImg; // variables in system

    void targetLocationInit (struct UavData* curUavData);
    void rotationMatrixCalculationIB (double Roll_rad, double Pitch_rad, double Yaw_rad, struct transformationMatrix3x3 *output, int transform);
    void transposeMatrix3x3(struct transformationMatrix3x3 input, struct transformationMatrix3x3 *output);
    void mutipleMatrix3x1 (struct vector3x1 input, struct transformationMatrix3x3 rotationMatrix, struct vector3x1 *output);
    void mutipleMatrix3x3 (struct transformationMatrix3x3 A, struct transformationMatrix3x3 B, struct transformationMatrix3x3 *output);
    void calculateCalibrationMatrix(struct UavData* curUavData, struct transformationMatrix3x3 *calibrationMatrix, struct transformationMatrix3x3 *invCalibrationMatrix);
    void movePositionFlatEarth (struct GpsPosition origPos, float pTrack_rad, float pDistance_m, GpsPosition &newPos);
    void averageFilter (int n,GpsPosition *PosBuffer, GpsPosition &afterPos);
};

#endif  // TARGETLOCALIZATION_H
