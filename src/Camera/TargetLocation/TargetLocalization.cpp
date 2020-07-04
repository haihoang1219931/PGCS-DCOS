/**
*                                                                   
* @clacc TargetLocalization                                                      
*                                                                   
* Class estimate the gps position of target from the camera image.                                  
*                                                                   
* 2020 created by Dongnt11 @viettel.com.vn                                   
*/
#include "stdio.h"
#include "TargetLocalization.h"

// Multiplication data in degrees factor to store data in int type.
const float TargetLocalization::scaleFactor = 10000000.0f;
const int TargetLocalization::scaleFactorI = 10000000;
const double TargetLocalization::scaleFactorD = TargetLocalization::scaleFactor;  // Types conversion.
const float TargetLocalization::EARTH_RADIUS = 6372797.560856f;
//const float PI = 3.14159265358979323846f;
//const float PI2 = PI * 0.5f;


void TargetLocalization::visionViewInit (float pixel_size, float image_col_pixel, float image_row_pixel)
{
	initializedVisionView = 1;
	UavParams.pixelSize = pixel_size;
	UavParams.imageHeight = image_row_pixel;
	UavParams.imageWidth = image_col_pixel;

	// calculate the parameters
	float diagonal = sqrtf(UavParams.imageWidth * UavParams.imageWidth + UavParams.imageHeight * UavParams.imageHeight);
	if (pixel_size == 0.0f)
	{
		UavParams.Sx = 0.0000038;//curUavData->pixelSize / diagonal;			//0.0039f;
		UavParams.Sy = 0.0000038;//curUavData->pixelSize / diagonal;		//0.0039f;
	}
	else
	{
		UavParams.Sx = UavParams.pixelSize / diagonal;
		UavParams.Sy = UavParams.pixelSize / diagonal;
	}
    UavParams.offset_X = UavParams.imageWidth/2.0f;
    UavParams.offset_Y = UavParams.imageHeight/2.0f;
	// limit the target distance to UAV 25km
	UavParams.maxImageDepth = 25000.0f;
}

void TargetLocalization::rotationMatrixCalculationIB (double Roll_rad, double Pitch_rad, double Yaw_rad, struct transformationMatrix3x3 *output, int transform)
{
	// Calculate the tranformation matrix from inertial frame to body frame
    output->R11 =  cosf(Yaw_rad) * cosf(Pitch_rad);
    output->R21 =  sinf(Roll_rad) * sinf(Pitch_rad) * cosf(Yaw_rad) - cosf(Roll_rad) * sinf(Yaw_rad);
    output->R31 =  cosf(Roll_rad) * sinf(Pitch_rad) * cosf(Yaw_rad) + sinf(Roll_rad) * sinf(Yaw_rad);

    output->R12 =  sinf(Yaw_rad) * cosf(Pitch_rad);
    output->R22 =  cosf(Yaw_rad) * cosf(Roll_rad) + sinf(Roll_rad) * sinf(Pitch_rad) * sinf(Yaw_rad);
    output->R32 = -sinf(Roll_rad) * cosf(Yaw_rad) + sinf(Yaw_rad) * sinf(Pitch_rad) * cosf(Roll_rad);

    output->R13 = -sinf(Pitch_rad);
    output->R23 =  sinf(Roll_rad) * cosf(Pitch_rad);
    output->R33 =  cosf(Pitch_rad) * cosf(Roll_rad);

	// Calculate the tranformation matrix from body frame to inertial frame
	if (transform == 1)
	{
		float a12 = output->R12;
		float a13 = output->R13;

		output->R12 = output->R21;
		output->R21 = a12;
		output->R13 = output->R31;
		output->R31 = a13;

		float a23 = output->R23;
		output->R23 = output->R32;
		output->R32 = a23;
	}
}

/*
*  initialize the UAV data
*/

void TargetLocalization::targetLocationInit (struct UavData* curUavData)
{
	// init variables
	curUavData->imageDepth = 0.1f;
	curUavData->opticalLength = 1.0f;
	curUavData->Pan_rad = 0.0f;
	curUavData->Tilt_rad = 0.0f;

	curUavData->Roll_rad = 0.0f;
	curUavData->Pitch_rad = 0.0f;
	curUavData->Yaw_rad = 0.0f;

	// init the position of target
	curUavData->TargetPosition.Altitude = 0.0f;
	curUavData->TargetPosition.Latitude = 21.031156f;
	curUavData->TargetPosition.Longitude = 105.494485f;

	curUavData->UavPosition.Altitude = 0.0f;
	curUavData->UavPosition.Latitude = 21.031156f;
	curUavData->UavPosition.Longitude = 105.494485f;

}
/*
*      Calibration Matrix of Camera
*/

void TargetLocalization::calculateCalibrationMatrix(float opticalLength, struct transformationMatrix3x3 *calibrationMatrix, struct transformationMatrix3x3 *invCalibrationMatrix)
{
	float fx = opticalLength / UavParams.Sx;
	float fy = opticalLength / UavParams.Sy;
	// calibration matrix
	calibrationMatrix->R11 = 0.0f;
	calibrationMatrix->R12 = fx;
	calibrationMatrix->R13 = UavParams.offset_X;

	calibrationMatrix->R21 = -fy;
	calibrationMatrix->R22 = 0.0f;
	calibrationMatrix->R23 = UavParams.offset_Y;

	calibrationMatrix->R31 = 0.0f;
	calibrationMatrix->R32 = 0.0f;
	calibrationMatrix->R33 = 1.0f;

	// invert of calibration Matrix
	invCalibrationMatrix->R11 = 0.0f;
	invCalibrationMatrix->R12 = -1/fy;
	invCalibrationMatrix->R13 = UavParams.offset_Y/fy;

	invCalibrationMatrix->R21 = 1/fx;
	invCalibrationMatrix->R22 = 0.0f;
	invCalibrationMatrix->R23 = -UavParams.offset_X/fx;

	invCalibrationMatrix->R31 = 0.0f;
	invCalibrationMatrix->R32 = 0.0f;
	invCalibrationMatrix->R33 = 1.0f;
}
/*
*  output = A' in case of rotation matrix inv(A) = A'
*/

void TargetLocalization::transposeMatrix3x3(struct transformationMatrix3x3 input, struct transformationMatrix3x3 *output)
{
	*output = input;
	output->R12 = input.R21;
	output->R13 = input.R31;
	
	output->R21 = input.R12;
	output->R23 = input.R32;

	output->R31 = input.R13;
	output->R32 = input.R23;
}

/*
*             output = A * B
*/

void TargetLocalization::mutipleMatrix3x3 (struct transformationMatrix3x3 A, struct transformationMatrix3x3 B, struct transformationMatrix3x3 *output)
{
	output->R11 = A.R11 * B.R11 + A.R12 * B.R21 + A.R13 * B.R31;
	output->R12 = A.R11 * B.R12 + A.R12 * B.R22 + A.R13 * B.R32;
	output->R13 = A.R11 * B.R13 + A.R12 * B.R23 + A.R13 * B.R33;

	output->R21 = A.R21 * B.R11 + A.R22 * B.R21 + A.R23 * B.R31;
	output->R22 = A.R21 * B.R12 + A.R22 * B.R22 + A.R23 * B.R32;
	output->R23 = A.R21 * B.R13 + A.R22 * B.R23 + A.R23 * B.R33;

	output->R31 = A.R31 * B.R11 + A.R32 * B.R21 + A.R33 * B.R31;
	output->R32 = A.R31 * B.R12 + A.R32 * B.R22 + A.R33 * B.R32;
	output->R33 = A.R31 * B.R13 + A.R32 * B.R23 + A.R33 * B.R33;
}

/*
*         output = R * input
*/
void TargetLocalization::mutipleMatrix3x1 (struct vector3x1 input, struct transformationMatrix3x3 R, struct vector3x1 *output)
{
	output->V11 = input.V11 * R.R11 + input.V21 * R.R12 + input.V31 * R.R13;
	output->V21 = input.V11 * R.R21 + input.V21 * R.R22 + input.V31 * R.R23;
	output->V31 = input.V11 * R.R31 + input.V21 * R.R32 + input.V31 * R.R33;
}

/*
*      Move gps position from 1 point to another point
*/
void TargetLocalization::movePositionFlatEarth (struct GpsPosition origPos, float pTrack_rad, float pDistance_m, struct GpsPosition &newPos)
{
	// calculate distance in lat and lon axis
	float distLat = pDistance_m * cosf (pTrack_rad); 
    float distLon = pDistance_m * sinf (pTrack_rad);

	static const float RD1 = DEG_2_RAD / scaleFactor;
    static const float DR3 = RAD_2_DEG * (scaleFactor / EARTH_RADIUS);
    static const float MAX_LAT = 89.0f * scaleFactor;

    //  Limitation to the latitude range <-89,+89>
    if (origPos._latitude > MAX_LAT || origPos._latitude < -MAX_LAT)
    {
        newPos = origPos;
    }

    //  latitude in radians
    float lat0 = static_cast<float>(origPos._latitude) * RD1;

    //  Meters conversion to degrees and scale
    int dLat = static_cast<int>(distLat * DR3);
    int dLon = static_cast<int>((distLon * DR3) / cosf (lat0));

	//  correctness control of the float
    bool ret1 = assure(distLat, 0.0f);
    bool ret2 = assure(distLon, 0.0f);

    // new coordinates
    newPos._longitude = origPos._longitude + dLon;
    newPos._latitude = origPos._latitude + dLat;

	newPos.Latitude = (double)(newPos._latitude) / scaleFactor;
	newPos.Longitude = (double)(newPos._longitude) / scaleFactor;
}

/*
* moving average filter
*/

void TargetLocalization::averageFilter(int n, GpsPosition *PosBuffer, GpsPosition &afterPos)
{
	for (int i = 1; i <= n; i++)
	{
		afterPos.Altitude = afterPos.Altitude + (*PosBuffer).Altitude / (double)(n);
		afterPos.Latitude = afterPos.Latitude + (*PosBuffer).Latitude / (double)(n);
		afterPos.Longitude = afterPos.Longitude + (*PosBuffer).Longitude / (float)(n);
		afterPos._latitude = afterPos._latitude + (*PosBuffer)._latitude / (int)(n);
		afterPos._longitude = afterPos._longitude + (*PosBuffer)._longitude / (int)(n);
		PosBuffer += 2;
	}
}

/*
* correct a convex polygons
*/
void TargetLocalization::convexPolygons (struct GpsPosition &leftTopPoint, struct GpsPosition &leftBottomPoint, struct GpsPosition &rightTopPoint, struct GpsPosition &rightBottomPoint)
{
	double lat_average, lon_average;
	lat_average = (leftTopPoint.Latitude + leftBottomPoint.Latitude + rightTopPoint.Latitude + rightBottomPoint.Latitude)/4.0f;
	lon_average = (leftTopPoint.Longitude + leftBottomPoint.Longitude + rightTopPoint.Longitude + rightBottomPoint.Longitude)/4.0f;

	GpsPosition Point[4];
	Point[0] = leftTopPoint;
	Point[1] = leftBottomPoint;
	Point[2] = rightTopPoint;
	Point[3] = rightBottomPoint;

	// correct a convex polygons
	for(int i = 0; i++; i<4)
	{
		if((Point[i].Latitude >= lat_average) && (Point[i].Longitude <= lon_average))
			leftTopPoint = Point[i];
		if((Point[i].Latitude >= lat_average) && (Point[i].Longitude >= lon_average))
			rightTopPoint = Point[i];
		if((Point[i].Latitude <= lat_average) && (Point[i].Longitude >= lon_average))
			rightBottomPoint = Point[i];
		if((Point[i].Latitude <= lat_average) && (Point[i].Longitude <= lon_average))
			rightTopPoint = Point[i];
	}

}

/*
* correct the center point in a convex polygons
*/
void TargetLocalization::correctCenterPoint (struct GpsPosition leftTopPoint, struct GpsPosition leftBottomPoint, struct GpsPosition rightTopPoint, struct GpsPosition rightBottomPoint, struct GpsPosition &centerPoint)
{
	double lat_average, lon_average;
	lat_average = (leftTopPoint.Latitude + leftBottomPoint.Latitude + rightTopPoint.Latitude + rightBottomPoint.Latitude)/4.0f;
	lon_average = (leftTopPoint.Longitude + leftBottomPoint.Longitude + rightTopPoint.Longitude + rightBottomPoint.Longitude)/4.0f;

	float x1 = distanceFlatEarth(leftTopPoint, rightBottomPoint);
	float x2 = distanceFlatEarth(leftBottomPoint, rightTopPoint);

	float x = compareMaxF(x1, x2);
	
	float y1 = distanceFlatEarth(leftTopPoint, centerPoint);
	float y2 = distanceFlatEarth(leftBottomPoint, centerPoint);
	float y3 = distanceFlatEarth(rightTopPoint, centerPoint);
	float y4 = distanceFlatEarth(rightBottomPoint, centerPoint);

	float y12 = compareMaxF(y1, y2);
	float y34 = compareMaxF(y3, y4);

	float y = compareMaxF(y12, y34);

	if (y > x)
	{
		centerPoint.Latitude = lat_average;
		centerPoint.Longitude = lon_average;
		centerPoint._latitude = (int)(centerPoint.Latitude * scaleFactorD);
		centerPoint._longitude = (int)(centerPoint.Longitude * scaleFactorD);
	}
}

/*
* Distance between to point in map
*/
float TargetLocalization::distanceFlatEarth(struct GpsPosition fromPos, struct GpsPosition toPos)
{
	static const float RD1 = DEG_2_RAD / scaleFactor;
    static const float RD2 = (EARTH_RADIUS * DEG_2_RAD) / scaleFactor;

    float dLon = static_cast<float>(toPos._longitude - fromPos._longitude);
    float dLat = static_cast<float>(toPos._latitude - fromPos._latitude);

    float lat0 = static_cast<float>(fromPos._latitude) * RD1;
	float cLat0 = cosf (lat0);
    float d = RD2 * sqrtf (dLat*dLat + dLon*dLon*cLat0*cLat0);

    return d;
}

/*
* Target location compute
*/
void TargetLocalization::targetLocationCompute (float target_x_ip, float target_y_ip, struct UavData* curUavData, float maxImageDepth, struct GpsPosition &targetPoint)
{
	float Zcc = curUavData->UavPosition.Altitude;
	float Zobj = 0.0f;
    float dTrack = 0.0f;
    float dDistance = 0.0f;

	// init data
	struct transformationMatrix3x3 Rib, Rbg, Rgc;

	// calculate the calibration matrix of camera
	calculateCalibrationMatrix(curUavData->opticalLength, &calibMatrix, &invCalibMatrix);

	// calculate the rotation matrix
	rotationMatrixCalculationIB(curUavData->Roll_rad, curUavData->Pitch_rad, curUavData->Yaw_rad, &Rib, 0);
	rotationMatrixCalculationIB(0.0f,curUavData->Tilt_rad, curUavData->Pan_rad, &Rbg, 0);
	rotationMatrixCalculationIB(0.0f, PI2, 0.0f, &Rgc, 0); // from gimbal frame to camera fram: x-> z, y-> y, z-> -x

	// calculate the rotation matrix from inertial to camera
	struct transformationMatrix3x3 Ric;
    //from inertial frame to gimbal frame
    //mutipleMatrix3x3(Rib, Rbg, &Ric);
    mutipleMatrix3x3(Rgc, Rbg, &Ric);
	// from gimbal frame to camera frame
    //mutipleMatrix3x3(Ric, Rgc, &Ric);
    mutipleMatrix3x3(Ric, Rib, &Ric);
	
	// calculate the position of target in image frame
	struct vector3x1 TargetIm;
	TargetIm.V11 = target_x_ip;
	TargetIm.V21 = target_y_ip;
	TargetIm.V31 = 1.0f;

	// calculate the position of target in camera frame by calibration matrix
	// C * TargetIm = TargetC
	struct vector3x1 TargetC;
	mutipleMatrix3x1(TargetIm, invCalibMatrix, &TargetC);

	// calculate the image depth
	// lamda = -zcc/(zobj - zcc)
    // zcc = -h uav altitude
	// zobj = R31ci * xobjc + R32ci * yobjc + R33ci * zobjc - h
	struct transformationMatrix3x3 Rci;
	transposeMatrix3x3(Ric, &Rci);
    float zobj = (Rci.R31 * TargetC.V11 + Rci.R32 * TargetC.V21 + Rci.R33 * TargetC.V31) - Zcc;
    curUavData->imageDepth = Zcc/(zobj + Zcc);

	// limit the range of imagedepth
	if((curUavData->imageDepth <= 0) || (curUavData->imageDepth > maxImageDepth))
	{
		curUavData->imageDepth = maxImageDepth;
	}

	// calculate the position of target in the local coordination
	struct vector3x1 TargetI;
	mutipleMatrix3x1(TargetC, Rci, &TargetI);

	float Target_x = (float)(curUavData->imageDepth * TargetI.V11);
	float Target_y = (float)(curUavData->imageDepth * TargetI.V21);

	dDistance = sqrtf(Target_x * Target_x + Target_y * Target_y);
	dTrack = atan2(Target_y, Target_x);

	//  correctness control of the float
    bool ret1 = assure(dDistance, 0.0f);
    bool ret2 = assure(dTrack, 0.0f);

	// calculate the position of target
    movePositionFlatEarth(curUavData->UavPosition, dTrack, dDistance, targetPoint);
}
/*
* Target location main
*/

void TargetLocalization::targetLocationMain (float x_ip, float y_ip, float hfov_rad, float roll_rad, float pitch_rad, float yaw_rad,
    float pan_rad, float tilt_rad, double uav_lat_deg, double uav_lon_deg, float uav_alt_m, float uav_terrain_m, double *targetPosition)
{
	struct UavData* UavDataState = &currentUavData;
	targetLocationInit(UavDataState);

	// update current data of uav
	UavDataState->hfov = hfov_rad;
	UavDataState->opticalLength = UavParams.Sx * UavParams.imageWidth / (2.0f * tanf(hfov_rad / 2.0f));
	UavDataState->Pan_rad = pan_rad;
	UavDataState->Tilt_rad = -tilt_rad;

	UavDataState->Roll_rad = roll_rad;
	UavDataState->Pitch_rad = pitch_rad;
	UavDataState->Yaw_rad = yaw_rad;

	UavDataState->UavPosition.Altitude = uav_alt_m - uav_terrain_m;
    UavDataState->UavPosition.Latitude = uav_lat_deg;
    UavDataState->UavPosition.Longitude = uav_lon_deg;

	UavDataState->UavPosition._latitude = (int)(uav_lat_deg * scaleFactorD);
	UavDataState->UavPosition._longitude = (int)(uav_lon_deg * scaleFactorD);

	// calculate the position of target
	targetLocationCompute(x_ip, y_ip, UavDataState, UavParams.maxImageDepth, UavDataState->TargetPosition);

	// update the data
	currentUavData.imageDepth = UavDataState->imageDepth;
	currentUavData.opticalLength = UavDataState->opticalLength;
	currentUavData.Pan_rad = UavDataState->Pan_rad;
	currentUavData.Pitch_rad = UavDataState->Pitch_rad;
	currentUavData.Roll_rad = UavDataState->Roll_rad;
	currentUavData.TargetPosition = UavDataState->TargetPosition;
	currentUavData.Tilt_rad = UavDataState->Tilt_rad;
	currentUavData.UavPosition = UavDataState->UavPosition;
	currentUavData.Yaw_rad = UavDataState->Yaw_rad;

	// update the new location of target
    targetPosition[0] = UavDataState->TargetPosition.Latitude;
    targetPosition[1] = UavDataState->TargetPosition.Longitude;
}

/*
* Optical view main
*/
void TargetLocalization::visionViewMain (float hfov_rad, float roll_rad, float pitch_rad, float yaw_rad, float pan_rad, float tilt_rad,
		double *uav_position, float uav_alt_m, float uav_terrain_m, double *centerPoint, double *leftTopPoint, double *rightTopPoint, double *leftBottomPoint, double *rightBottomPoint)
{
	// update current data of uav
	struct UavData* UavDataState = &currentUavData;

	UavDataState->hfov = hfov_rad;
	UavDataState->opticalLength = UavParams.Sx * UavParams.imageWidth / (2.0f * tanf(hfov_rad / 2.0f));
	UavDataState->Pan_rad = pan_rad;
	UavDataState->Tilt_rad = -tilt_rad;

	UavDataState->Roll_rad = roll_rad;
	UavDataState->Pitch_rad = pitch_rad;
	UavDataState->Yaw_rad = yaw_rad;

	UavDataState->UavPosition.Altitude = uav_alt_m - uav_terrain_m;
    UavDataState->UavPosition.Latitude = uav_position[0];
    UavDataState->UavPosition.Longitude = uav_position[1];

	UavDataState->UavPosition._latitude = (int)(uav_position[0] * scaleFactorD);
	UavDataState->UavPosition._longitude = (int)(uav_position[1] * scaleFactorD);

	// calculate the position of vision view
	struct GpsPosition leftTopView, rightTopView, leftBottomView, rightBottomView;
	targetLocationCompute(0.0f, 0.0f, UavDataState, 30000.0f, leftTopView);
	targetLocationCompute(UavParams.imageWidth, 0.0f, UavDataState, 30000.0f, rightTopView);
	targetLocationCompute(0.0f, UavParams.imageHeight, UavDataState, 20000.0f, leftBottomView);
	targetLocationCompute(UavParams.imageWidth, UavParams.imageHeight, UavDataState, 20000.0f, rightBottomView);

	// arrange 4 points into a convex polygons
	convexPolygons(leftTopView, leftBottomView, rightTopView, rightBottomView);

	// calculate the center point of optical line of sight
	struct GpsPosition centerOpiticalPoint;
	targetLocationCompute(UavParams.imageWidth * 0.5f, UavParams.imageHeight * 0.5f, UavDataState, 25000.0f, centerOpiticalPoint);

	// modify the center point
	correctCenterPoint(leftTopView, leftBottomView, rightTopView, rightBottomView, centerOpiticalPoint);

	// update the data
	currentUavData.imageDepth = UavDataState->imageDepth;
	currentUavData.opticalLength = UavDataState->opticalLength;
	currentUavData.Pan_rad = UavDataState->Pan_rad;
	currentUavData.Pitch_rad = UavDataState->Pitch_rad;
	currentUavData.Roll_rad = UavDataState->Roll_rad;
	currentUavData.TargetPosition = UavDataState->TargetPosition;
	currentUavData.Tilt_rad = UavDataState->Tilt_rad;
	currentUavData.UavPosition = UavDataState->UavPosition;
	currentUavData.Yaw_rad = UavDataState->Yaw_rad;

	// update the location of center point
    centerPoint[0] = centerOpiticalPoint.Latitude;
    centerPoint[1] = centerOpiticalPoint.Longitude;

	// update the location of left top point
    leftTopPoint[0] = leftTopView.Latitude;
    leftTopPoint[1] = leftTopView.Longitude;

	// update the location of right top point
    rightTopPoint[0] = rightTopView.Latitude;
    rightTopPoint[1] = rightTopView.Longitude;

	// update the location of left bottom point
    leftBottomPoint[0] = leftBottomView.Latitude;
    leftBottomPoint[1] = leftBottomView.Longitude;

	// update the location of right bottom point
    rightBottomPoint[0] = rightBottomView.Latitude;
    rightBottomPoint[1] = rightBottomView.Longitude;
}
