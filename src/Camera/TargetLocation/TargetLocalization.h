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

	// Params for uav data
	struct visionViewParam
	{
		// Pixel Size
		float pixelSize; // [mm]
		
		// Camera image scaler
		float Sx; // [mm/pixel]
		float Sy; // [mm/pixel]

		// Offsets of optical center of camera
		int offset_X; // [pixels]
		int offset_Y; // [pixels]

		// image size
		float imageWidth;	// width of image frame [pixel]
		float imageHeight;  // height of image frame [pixel]
		float maxImageDepth;

	};

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

		// optical length
		float opticalLength; // [m]
		float hfov; // [rad]

	};
	
	struct UavData currentUavData;
	struct visionViewParam UavParams;

	void targetLocationMain (float x_ip, float y_ip, float hfov_rad, float roll_rad, float pitch_rad, float yaw_rad, 
		float pan_rad, float tilt_rad, double uav_lat_deg, double uav_lon_deg, float uav_alt_m, float uav_terrain_m, double *targetPosition);
	void visionViewMain (float hfov_rad, float roll_rad, float pitch_rad, float yaw_rad, float pan_rad, float tilt_rad,
		double *uav_position, float uav_alt_m, float uav_terrain_m, double *centerPoint, double *leftTopPoint, double *rightTopPoint, double *leftBottomPoint, double *rightBottomPoint);

	void visionViewInit (float pixel_size, float image_col_pixel, float image_row_pixel);
	float distanceFlatEarth (struct GpsPosition fromPos, struct GpsPosition toPos);

private:
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

	// min max of 2 float
	static float compareMaxF (float in1, float in2)
    {
        return (in1 > in2) ? in1 : in2;
    } 
    
    static float compareMinF (float in1, float in2)
    {
        return (in1 < in2) ? in1 : in2;
    }
	// rotation matrix
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

	int initializedVisionView;
	int initializedTargetLocation;
	struct transformationMatrix3x3 calibMatrix, invCalibMatrix;
	
	struct vector3x1 positionCam, positionNED, positionImg; // variables in system

	void targetLocationInit (struct UavData* curUavData);
	void rotationMatrixCalculationIB (double Roll_rad, double Pitch_rad, double Yaw_rad, struct transformationMatrix3x3 *output, int transform);
	void transposeMatrix3x3(struct transformationMatrix3x3 input, struct transformationMatrix3x3 *output);
	void mutipleMatrix3x1 (struct vector3x1 input, struct transformationMatrix3x3 rotationMatrix, struct vector3x1 *output);
	void mutipleMatrix3x3 (struct transformationMatrix3x3 A, struct transformationMatrix3x3 B, struct transformationMatrix3x3 *output);
	void calculateCalibrationMatrix(float opticalLength, struct transformationMatrix3x3 *calibrationMatrix, struct transformationMatrix3x3 *invCalibrationMatrix);
	void movePositionFlatEarth (struct GpsPosition origPos, float pTrack_rad, float pDistance_m, struct GpsPosition &newPos);
	void averageFilter (int n,GpsPosition *PosBuffer, GpsPosition &afterPos);

	void convexPolygons (struct GpsPosition &leftTopPoint, struct GpsPosition &leftBottomPoint, struct GpsPosition &rightTopPoint, struct GpsPosition &rightBottomPoint);
	void correctCenterPoint (struct GpsPosition leftTopPoint, struct GpsPosition leftBottomPoint, struct GpsPosition rightTopPoint, struct GpsPosition rightBottomPoint, struct GpsPosition &centerPoint);

	void targetLocationCompute (float target_x_ip, float target_y_ip, struct UavData* curUavData, float maxImageDepth, struct GpsPosition &targetPoint);
};

#endif  // TARGETLOCALIZATION_H
