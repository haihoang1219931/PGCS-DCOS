#ifndef GIMBALDATA_H
#define GIMBALDATA_H

#include <QObject>
#include <QVariant>
#include <QPointF>
#include <QGeoCoordinate>
#include <math.h>
#define MAX_SENSOR 10
class GimbalData : public QObject
{
    Q_OBJECT
public:
    explicit GimbalData(QObject *parent = nullptr);

Q_SIGNALS:
    void NotifyPropertyChanged(QString name);
public Q_SLOTS:
    void setVideoDestination(QString videoIP, int videoPort){
        m_videoIP = videoIP;
        m_videoPort = videoPort;
        Q_EMIT NotifyPropertyChanged("VideoDestination");
    }
    void setGimbalMode(QString gimbalMode){
        m_gimbalMode = gimbalMode;
    }
    void setStabGimbal(bool stabPan, bool stabTilt){
        m_enableGyroStabilisationPan = stabPan;
        m_enableGyroStabilisationTilt = stabTilt;
    }
    void setStabDigital(bool videoStabMode){
        m_videoStabMode = videoStabMode;
    }
    void setRecordingStatus(bool recording){
        m_recording = recording;
    }
    ////////////////////////////////
    /// \brief setGimbalInfo
    /// \param panPos
    /// \param panVel
    /// \param tiltPos
    /// \param tiltVel
    ///
    void setGimbalInfo(float panPos,float panVel,float tiltPos,float tiltVel){
        m_panPosition = panPos;
        m_panVelocity = panVel;
        m_tiltPosition = tiltPos;
        m_tiltVelocity = tiltVel;
    }
    ////////////////////////////////
    /// \brief setPlatformPosition
    /// \param lat
    /// \param lon
    /// \param alt
    ///
    void setPlatformPosition(float lat, float lon, float alt){
        m_latitude = lat;
        m_longtitude = lon;
        m_altitudeOffset = alt;
        char data[256];
        // pn,pe,pd,roll,pitch,yaw,al,ez,targetLat,targetLon;
        sprintf(data,"%f,%f,%f,%f,%f,%f,%f,%f,%f,%f",
                m_latitude,m_longtitude,m_altitudeOffset,
                m_rollOffset,m_pitchOffset,m_yawOffset,
                m_panPosition,m_tiltPosition,
                m_centerLat,m_centerLon);
        //        printf("setPlatformPosition %s\r\n",data);
        //        FileControler::addLine(m_logFile,data);
        Q_EMIT NotifyPropertyChanged("position");
    }

    void setPlatformOrientation(float roll, float pitch, float yaw){
        m_rollOffset = roll;
        m_pitchOffset = pitch;
        m_yawOffset = yaw;
        Q_EMIT NotifyPropertyChanged("orientation");
    }
    void setCurrentTargetLocation(float lat, float lon,float slr){
        m_targetLat = lat;
        m_targetLon = lon;
        m_targetSlr = slr;
    }
    void setGeolockLocation(float lat, float lon,float alt){
        m_geoLat = lat;
        m_geoLon = lon;
        m_geoAlt = alt;
    }
    void setCorners(float lat01, float lon01,
                    float lat02, float lon02,
                    float lat03, float lon03,
                    float lat04, float lon04,
                    float centerLat, float centerLon){
        m_cornerLat[0] = lat01;
        m_cornerLon[0] = lon01;
        m_cornerLat[1] = lat02;
        m_cornerLon[1] = lon02;
        m_cornerLat[2] = lat03;
        m_cornerLon[2] = lon03;
        m_cornerLat[3] = lat04;
        m_cornerLon[3] = lon04;
        m_centerLat = centerLat;
        m_centerLon = centerLon;
        Q_EMIT NotifyPropertyChanged("corners");
    }
    void setLaserRangeState(QString  state){
        if(state == "START"){
            m_lrfInValidCount = 0;
            m_lrfValidCount = 0;
            m_laserRangeStart = 0;
        }else if(state == "STOPPED"){
            m_laserRangeStart = -1;
        }
        //        printf("setLaserRangeState = %d\r\n",m_laserRangeStart);
    }
    void setLaserRange(float _laserRange,int _valid){
        m_laserRange = _laserRange;
        if(m_lrfCurrent == -1)
            m_lrfCurrent = _valid;
        if(_valid == 0 && m_laserRangeStart == 0){
            m_lrfInValidCount ++;
        }else if(_valid == 1 && m_laserRangeStart == 0){
            m_lrfValidCount ++;
        }
        if(_valid != m_lrfCurrent){
            m_lrfInValidCount = 0;
            m_lrfValidCount = 0;
            m_lrfCurrent = _valid;
        }
        //        printf("m_lrfInValidCount = %d\r\n",m_lrfInValidCount);
        //        printf("m_lrfValidCount = %d\r\n",m_lrfValidCount);
        if(m_lrfInValidCount == 10 && m_laserRangeStart == 0){
            Q_EMIT NotifyPropertyChanged("LaserRangeInValid");
            m_laserRangeStart = -1;
            m_lrfInValidCount = 0;
        }else if(m_lrfValidCount == 10  && m_laserRangeStart == 0){
            Q_EMIT NotifyPropertyChanged("LaserRangeValid");
            m_laserRangeStart = -1;
            m_lrfValidCount = 0;
        }
    }
    void setTrackParams(int _trackSize,QString _trackMode = ""){
        m_trackSize = _trackSize;
        m_trackMode = _trackMode;
        Q_EMIT NotifyPropertyChanged("TrackParams");
    }
    void setSensorID(int sensorID){
        m_sensorID = sensorID;
    }

    void setFOVSensor(int sensorID,float _hFOV, float _vFOV){
        set_float(&m_hfov[sensorID],_hFOV,"HFOV");
        set_float(&m_vfov[sensorID],_vFOV,"VFOV");
        float zoom =  tan(m_hfovMax[sensorID]/2/180*M_PI) / tan(m_hfov[sensorID]/2/180*M_PI) ;
        set_float(&m_zoom[sensorID],zoom,"ZoomSensor");
//        printf("Sensor[%d] hfov:[%.02f] vfov:[%.02f] zoom[%.02f]\r\n",sensorID,_hFOV,_vFOV,zoom);
    }
    void setVersion(QString gimbalSerial,
                    QString firmwareVersion,
                    QString hardwareVersion,
                    QString protocolVersion){
        m_gimbalSerialNumber = gimbalSerial;
        m_firmwareVersion = firmwareVersion;
        m_hardwareVersion = hardwareVersion;
        m_protocolVersion = protocolVersion;
        Q_EMIT NotifyPropertyChanged(QString("Version"));
    }

    Q_INVOKABLE QVariant getData(int index){
        QVariant res;
        QVariantMap map;
        // vesion
        map.insert("GIMBAL_SERIAL", (m_gimbalSerialNumber));
        map.insert("FIRMWARE_VERSION", (m_firmwareVersion));
        map.insert("HARDWARE_VERSION", (m_hardwareVersion));
        map.insert("PROTOCOL_VERSION", (m_protocolVersion));
        map.insert("id", index);
        map.insert("SystemMode", (m_lockMode));
        map.insert("hfov", m_hfov[m_sensorID]);
        map.insert("panPos", m_panPosition);
        map.insert("panVel", m_panVelocity);
        map.insert("tiltPos", m_tiltPosition);
        map.insert("tiltVel", m_tiltVelocity);

        map.insert("pn", m_latitude);
        map.insert("pe", m_longtitude);
        map.insert("pd", m_altitudeOffset);
        map.insert("roll", m_rollOffset);
        map.insert("pitch", m_pitchOffset);
        map.insert("yaw", m_yawOffset);

        map.insert("speedNorth",m_speedNorth);
        map.insert("speedEast", m_speedEast);
        map.insert("gpsAlt", m_gpsAlt);
        map.insert("takeOffAlt", m_gpstakeOffAlt);
        map.insert("px", 0);
        map.insert("py", 0);
        // Target
        map.insert("TARGET_LAT", m_targetLat);
        map.insert("TARGET_LON", m_targetLon);
        map.insert("TARGET_SLR", m_targetSlr);
        // Geo Lock
        map.insert("GEO_LON", m_geoLon);
        map.insert("GEO_LAT", m_geoLat);
        map.insert("GEO_ALT", m_geoAlt);
        // IPC Status
        QString  sensor = m_sensorID == 0?"EO":"IR";
        map.insert("SENSOR", sensor);
        QVariantMap mapFov;
        mapFov.insert("EO",m_hfov[0]);
        mapFov.insert("IR",m_hfov[1]);
        map.insert("HFOV", mapFov);
        QVariantMap mapIRFov;
        mapIRFov.insert("HFOV",m_hfov[1]);
        mapIRFov.insert("VFOV",m_vfov[1]);
        map.insert("IR", mapIRFov);
        QVariantMap mapZoom;
        mapZoom.insert("EO",m_zoom[0]);
        mapZoom.insert("IR",m_zoom[1]);
        map.insert("ZOOM", mapZoom);
        map.insert("STAB_DIGITAL", m_videoStabMode);
        map.insert("LOCK_MODE", (m_lockMode));
        // Motion C Status
        map.insert("STAB_GIMBAL",
                   m_enableGyroStabilisationPan &
                   m_enableGyroStabilisationTilt);
        // Target
        map.insert("CORNER01", QPointF(m_cornerLat[0],m_cornerLon[0]));
        map.insert("CORNER02", QPointF(m_cornerLat[1],m_cornerLon[1]));
        map.insert("CORNER03", QPointF(m_cornerLat[2],m_cornerLon[2]));
        map.insert("CORNER04", QPointF(m_cornerLat[3],m_cornerLon[3]));
        map.insert("CENTER", QPointF(m_centerLat,m_centerLon));
        map.insert("UAV", QPointF(m_latitude,m_longtitude));
        map.insert("RECORD",m_recording);
        map.insert("GIMBAL_MODE",(m_gimbalMode));
        map.insert("TRACK_SIZE",m_trackSize);
        map.insert("TRACK_MODE",(m_trackMode));
        map.insert("LASER_RANGE", m_laserRange);
        map.insert("VIDEO_IP", (m_videoIP));
        map.insert("VIDEO_PORT", m_videoPort);
        map.insert("GCS_SHARED",m_gcsShare);
        map.insert("PRESET",m_presetMode);
        //        printf("m_enableGyroStabilisationPan = %s\r\n",m_enableGyroStabilisationPan==true?"ON":"OFF");
        //        printf("m_enableGyroStabilisationTilt = %s\r\n",m_enableGyroStabilisationTilt==true?"ON":"OFF");
        //        printf("STAB_DIGITAL = %s\r\n",m_videoStabMode==true?"ON":"OFF");
        //        printf("SENSOR = %d\r\n",m_sensorID);
        //        printf("GIMBAL = %s\r\n",m_gimbalMode.c_str());
        res = QVariant(map);
        return res;
    }
public:
    QString m_gimbalSerialNumber = "";
    QString m_firmwareVersion = "";
    QString m_hardwareVersion = "";
    QString m_protocolVersion = "";
    QString m_logFile;
    float m_panPosition = 0;
    float m_tiltPosition = 0;
    float m_zoomPosition = 0;
    float m_rollPosition = 0;
    float m_panVelocity = 0;
    float m_tiltVelocity = 0;
    float m_rollVelocity = 0;
    float m_panOffset = 0;
    float m_tiltOffset = 0;
    float m_latitude = 0;
    float m_longtitude = 0;
    float m_altitudeOffset = 0;
    float m_rollOffset = 0;
    float m_pitchOffset = 0;
    float m_yawOffset = 0;
    float m_cornerLat[4];
    float m_cornerLon[4];    
    float m_centerLat = 0;
    float m_centerLon = 0;
    int m_sensorID = 0;
    float m_hfov[MAX_SENSOR],m_vfov[MAX_SENSOR],m_zoom[MAX_SENSOR];
    float m_hfovMax[MAX_SENSOR];
    float m_hfovMin[MAX_SENSOR];
    float m_zoomMax[MAX_SENSOR];
    float m_zoomMin[MAX_SENSOR];
    float m_digitalZoomMax[MAX_SENSOR];
    float m_zoomTarget[MAX_SENSOR];
    float m_zoomCalculated[MAX_SENSOR];
    bool m_enableGyroStabilisationPan;
    bool m_enableGyroStabilisationTilt;
    bool m_sceneSteeringEnabled;
    bool m_sceneSteeringAutoEnabled;
    bool m_sceneSteerOnObjectTrackExit;
    bool m_sceneSteerOnGeolockExit;
    bool m_sceneSteerOnZeroRateExit;
    bool m_sceneSteerOnPositionMoveCompleteExit;
    bool m_enableStowTimeout;
    unsigned short m_stowTimeoutPeriod;
    float m_stowPositionTilt;
    float m_stowPositionPan;
    float m_encoderOffsetTilt;
    float m_encoderOffsetPan;
    float m_encoderResolution;
    bool m_invertEncoderPan;
    bool m_invertEncoderTilt;
    unsigned short m_motorStartSpeedTilt;
    unsigned short m_motorStartSpeedPan;
    bool m_invertMotorPan;
    bool m_invertMotorTilt;
    bool m_invertGyroPan;
    bool m_invertGyroTilt;
    float m_mechanicalOffsetPan;
    float m_mechanicalOffsetTilt;
    float m_maxTorquePan;
    float m_maxTorqueTilt;
    bool m_autoBoot;
    bool m_autoEnableGyroPan;
    bool m_autoEnableGyroTilt;
    int m_baud;
    float m_panLimitLeft;
    float m_panLimitRight;
    float m_tiltLimitDown;
    float m_tiltLimitUp;
    bool m_invertedPayload;
    bool m_enableSceneSteeringOnBoot;
    bool m_enableAutoSceneSteeringOnBoot;
    bool m_autoEnablePositionStreaming;
    QString m_panUpdateRate;
    QString m_tiltUpdateRate;
    bool m_enableAutoStabilisationBias;
    bool m_enableManualStabilisationBias;
    bool m_enableSceneSteering;
    bool m_enableRateRotationCorrection;
    int m_sceneSteeringTimeOut;
    QString m_gyroPanBias;
    QString m_gyroTiltBias;
    QString m_stowStatus;
    float m_positionEncoderPan;
    float m_positionEncoderTilt;
    float m_GyroPanPosition;
    float m_gyroTiltPosition;
    float m_velocityEncoderPan;
    float m_velocityEncoderTilt;
    QString m_controlModePan;
    QString m_controlModeTilt;
    float m_positionDemandedPan;
    float m_positionDemandedTilt;
    float m_velocityDemandedPan;
    float m_velocityDemandedTilt;
    float m_errorDeltaPan;
    float m_errorDeltaTilt;
    float m_errorPan;
    float m_errorTilt;
    float m_errorSumPan;
    float m_errorSumTilt;
    float m_gyroTemperaturePan;
    float m_gyroTemperatureTilt;
    float m_powerPan;
    float m_powerTilt;
    float m_timeStamp;
    QString m_lockMode = "FREE";
    bool m_videoStabMode = true;
    bool m_gcsShare = true;
    int m_lrfValidCount = 0;
    int m_lrfInValidCount = 0;
    int m_lrfCurrent = -1;
    float m_laserRange = -1;
    int m_laserRangeStart = -1;
    int m_trackSize = -1;
    QString m_gimbalMode = "NA";
    QString m_trackMode = "";
    QString m_presetMode = "OFF";
    float m_geoLon,m_geoLat,m_geoAlt;
    float m_targetLon,m_targetLat,m_targetSlr;
    float m_speedNorth,m_speedEast,m_gpsAlt,m_gpstakeOffAlt;
    bool m_recording = true;
    QString m_videoIP = "";
    int m_videoPort = 0;
    void set_ushort(unsigned short *val, unsigned short dat, const char* str)
    {
        if(*val != dat){
            *val = dat;
            Q_EMIT NotifyPropertyChanged(QString(str));
        }
    }
    void set_float(float *val, float dat, const  char* str)
    {
        if(*val != dat){
            *val = dat;
            Q_EMIT NotifyPropertyChanged(QString(str));
        }
    }

    void set_bool(bool *val, bool dat, const  char* str)
    {
        if(*val != dat){
            *val = dat;
            Q_EMIT NotifyPropertyChanged(QString(str));
        }
    }
    void set_int(int *val, int dat, const char* str)
    {
        if(*val != dat){
            *val = dat;
            Q_EMIT NotifyPropertyChanged(QString(str));
        }
    }
    void set_string(QString* val, QString dat, const char* str)
    {
        if(*val != dat){
            *val = dat;
            Q_EMIT NotifyPropertyChanged(QString(str));
        }
    }
};

#endif // GIMBALDATA_H
