#ifndef GIMBALCONTEXT_H
#define GIMBALCONTEXT_H
#include "../Cache/Cache.h"
#include "Buffer/RollBuffer.h"
#include "Packet/MotionImage.h"
#include "Packet/SystemStatus.h"
#include "Packet/TrackResponse.h"
#include "Packet/XPoint.h"
#include "UavvGimbalProtocol.h"
#include <QObject>
#include <QPointF>
#include <QVariant>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <vector>
#define NUM_SENSOR 2
using namespace std;

class GimbalContext : public QObject
{
        Q_OBJECT
    public:
        GimbalContext(QObject *parent = 0);
        virtual ~GimbalContext();

        RollBuffer<Eye::SystemStatus> *m_rbSystem;
        RollBuffer<Eye::MotionImage> *m_rbIPCEO;
        RollBuffer<Eye::MotionImage> *m_rbIPCIR;
        RollBuffer<Eye::TrackResponse> *m_rbTrackResEO;
        RollBuffer<Eye::XPoint> *m_rbXPointEO;
        RollBuffer<Eye::TrackResponse> *m_rbTrackResIR;
        RollBuffer<Eye::XPoint> *m_rbXPointIR;

        float m_pn, m_pe, m_pd;
        float m_pnMeasured, m_peMeasured, m_pdMeasured;
        float m_roll, m_pitch, m_yaw;
        float m_panVelocity, m_panAngle, m_tiltVelocity, m_tiltAngle;
        float m_freeMemory, m_totalMemory;
        bool m_videoStabMode, m_gimbalStabMode;
        float m_targetLat, m_targetLon, m_targetAlt, m_targetSlr;
        float m_geoLat, m_geoLon, m_geoAlt;
        string m_snapShotMode, m_sensorMode, m_sensorColorMode;
        bool m_gimbalRecord = false;
        string m_gimbalMode = "NA";
        string m_lockMode = "FREE";
        int m_sensorID = 0;
        float m_hfov[NUM_SENSOR], m_vfov[NUM_SENSOR], m_zoom[NUM_SENSOR];
        float m_zoomMax[NUM_SENSOR];
        float m_zoomMin[NUM_SENSOR];
        float m_cornerLat[4];
        float m_cornerLon[4];
        float m_centerLat = 0;
        float m_centerLon = 0;
        int m_trackSize = 50;
        string m_trackMode = "";
        int m_lrfValidCount = 0;
        int m_lrfInValidCount = 0;
        int m_lrfCurrent = -1;
        float m_laserRange = -1;
        int m_laserRangeStart = -1;
        // tuna21 - for joystick
        bool m_usingJoystick = false;
        float m_joystickX = 0, m_joystickY = 0;
        float m_steeringXPrev = 0, m_steeringYPrev = 0;

        void set_ushort(unsigned short *val, unsigned short dat, const char *str)
        {
            if (*val != dat) {
                *val = dat;
                Q_EMIT NotifyPropertyChanged(QString(str));
            }
        }

        void set_float(float *val, float dat, const char *str)
        {
            if (*val != dat) {
                *val = dat;
                Q_EMIT NotifyPropertyChanged(QString(str));
            }
        }

        void set_bool(bool *val, bool dat, const char *str)
        {
            if (*val != dat) {
                *val = dat;
                Q_EMIT NotifyPropertyChanged(QString(str));
            }
        }
        void set_int(int *val, int dat, const char *str)
        {
            if (*val != dat) {
                *val = dat;
                Q_EMIT NotifyPropertyChanged(QString(str));
            }
        }
        void set_string(string *val, string dat, const char *str)
        {
            if (*val != dat) {
                *val = dat;
                Q_EMIT NotifyPropertyChanged(QString(str));
            }
        }

        ////////////////////////////////

        //------------------
        void updateGPSData(float _pn, float _pe, float _pd, float _roll, float _pitch,
                           float _yaw)
        {
            if ((m_pn != _pn) || (m_pe != _pe) || (m_pd != _pd) || (m_roll != _roll) ||
                (m_pitch != _pitch) || (m_yaw != _yaw)) {
                m_pn = _pn;
                m_pe = _pe;
                m_pd = _pd;
                m_roll = _roll;
                m_pitch = _pitch;
                m_yaw = _yaw;
            }

            Q_EMIT NotifyPropertyChanged("GPSData");
        }
        void updateGLMeasured(float _pnMeasured, float _peMeasured,
                              float _pdMeasured)
        {
            if ((m_pnMeasured != _pnMeasured) || (m_peMeasured != _peMeasured) ||
                (m_pdMeasured != _pdMeasured)) {
                m_pnMeasured = _pnMeasured;
                m_peMeasured = _peMeasured;
                m_pdMeasured = _pdMeasured;
            }

            Q_EMIT NotifyPropertyChanged("GLValid");
        }
        void updateIPCState(Eye::IPCStatusResponse ipcStatusResponse)
        {
            data_type hfov = ipcStatusResponse.getHFOV() / M_PI * 180.0;
            data_type vfov = hfov / 16.0 * 9.0;
            bool digitalStab = false;
            byte bDigitalStab = ipcStatusResponse.getVideoStabEn();

            if (bDigitalStab == (byte)Status::VideoStabMode::ON) {
                digitalStab = true;
            }

            string lockMode;

            switch (ipcStatusResponse.getLockMode()) {
            case (byte)Status::LockMode::LOCK_GEO:
                lockMode = "GEO";
                break;

            case (byte)Status::LockMode::LOCK_VISUAL:
                lockMode = "VISUAL";
                break;

            case (byte)Status::LockMode::LOCK_OFF:
                lockMode = "FREE";
                break;

            case (byte)Status::LockMode::LOCK_TRACK:
                lockMode = "TRACK";
                break;

            default:
                break;
            }

            m_sensorID = ipcStatusResponse.getSensorMode();
            updateIPCState(hfov, vfov, digitalStab, lockMode);
            setTrackParams((int)ipcStatusResponse.getTrackSize(), "");
            m_gimbalRecord = ipcStatusResponse.getRecordMode();
            //            printf("gimbalRecord = %x\r\n", ipcStatusResponse.getRecordMode());
        }
        void setLaserRangeState(QString state)
        {
            if (state == "START") {
                m_lrfInValidCount = 0;
                m_lrfValidCount = 0;
                m_laserRangeStart = 0;
            } else if (state == "STOPPED") {
                m_laserRangeStart = -1;
            }

            //        printf("setLaserRangeState = %d\r\n",m_laserRangeStart);
        }
        void setLaserRange(float _laserRange, int _valid)
        {
            m_laserRange = _laserRange;

            if (m_lrfCurrent == -1) {
                m_lrfCurrent = _valid;
            }

            if (_valid == 0 && m_laserRangeStart == 0) {
                m_lrfInValidCount++;
            } else if (_valid == 1 && m_laserRangeStart == 0) {
                m_lrfValidCount++;
            }

            if (_valid != m_lrfCurrent) {
                m_lrfInValidCount = 0;
                m_lrfValidCount = 0;
                m_lrfCurrent = _valid;
            }

            //        printf("m_lrfInValidCount = %d\r\n",m_lrfInValidCount);
            //        printf("m_lrfValidCount = %d\r\n",m_lrfValidCount);
            if (m_lrfInValidCount == 10 && m_laserRangeStart == 0) {
                Q_EMIT NotifyPropertyChanged("LaserRangeInValid");
                m_laserRangeStart = -1;
                m_lrfInValidCount = 0;
            } else if (m_lrfValidCount == 10 && m_laserRangeStart == 0) {
                Q_EMIT NotifyPropertyChanged("LaserRangeValid");
                m_laserRangeStart = -1;
                m_lrfValidCount = 0;
            }
        }
        void setTrackParams(int _trackSize, string _trackMode = "")
        {
            m_trackSize = _trackSize;
            m_trackMode = _trackMode;
            Q_EMIT NotifyPropertyChanged("TrackParams");
        }
        void setSensorID(int sensorID)
        {
            m_sensorID = sensorID;
        }
        void SetFOVSensor(int sensorID, float _hFOV, float _vFOV)
        {
            set_float(&m_hfov[sensorID], _hFOV, "HFOV");
            set_float(&m_vfov[sensorID], _vFOV, "VFOV");
            float zoom = tan(m_zoomMax[sensorID] * M_PI / 180.0 / 2.0) /
                         tan(m_hfov[sensorID] * M_PI / 180.0 / 2.0);
            zoom = (float)((int)zoom);

            if (zoom < 1) {
                zoom = 1;
            }

            set_float(&m_zoom[sensorID], zoom, "ZoomSensor");
        }
        void updateIPCState(float _hfov, float _vfov, bool videoStab,
                            string lockMode)
        {
            {
                if (m_videoStabMode != videoStab) {
                    m_videoStabMode = videoStab;
                }

                if (m_lockMode != lockMode) {
                    m_lockMode = lockMode;
                }

                SetFOVSensor(0, _hfov, _vfov);
                Q_EMIT NotifyPropertyChanged("IPCStatus");
            }
        }

        void setGeolockLocation(float lat, float lon, float alt)
        {
            m_geoLat = lat;
            m_geoLon = lon;
            m_geoAlt = alt;
        }
        void setCorners(float lat01, float lon01, float lat02, float lon02,
                        float lat03, float lon03, float lat04, float lon04,
                        float centerLat, float centerLon)
        {
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
        }
        void updateMotionCStatus(Eye::MotionCStatus motionCStatus)
        {
            updateMotionCStatus(
                motionCStatus.getPanStabMode() == (byte)Status::StabMode::ON &&
                motionCStatus.getTiltStabMode() == (byte)Status::StabMode::ON,
                (double)motionCStatus.getPanPos(), (double)motionCStatus.getTiltPos(),
                (double)motionCStatus.getPanVelo(),
                (double)motionCStatus.getTiltVelo());
        }
        void updateMotionCStatus(bool gimbalStab, double panPos, double tiltPos,
                                 double panVel, double tiltVel)
        {
            m_panAngle = panPos;
            m_tiltAngle = tiltPos;
            m_panVelocity = panVel;
            m_tiltVelocity = tiltVel;
            m_gimbalStabMode = gimbalStab;
            Q_EMIT NotifyPropertyChanged("MotionCStatus");
        }
        Q_INVOKABLE QVariant getData(int index)
        {
            QVariant res;
            QVariantMap map;
            Eye::SystemStatus state;

            if (index > 0 && m_rbSystem->size() > 0) {
                state = m_rbSystem->getElementById((index_type)index);

                if (state.getIndex() != index || index == -1) {
                    state = m_rbSystem->last();
                }
            } else {
                Eye::Telemetry tmpTelemetry;
                Eye::IPCStatusResponse tmpIPCStatus;
                Eye::MotionCStatus tmpMotionCStatus;
                tmpTelemetry.setGPSAlt(0);
                tmpTelemetry.setPn(m_pn);
                tmpTelemetry.setPe(m_pe);
                tmpTelemetry.setPd(m_pd);
                tmpTelemetry.setRoll(m_roll);
                tmpTelemetry.setPitch(m_pitch);
                tmpTelemetry.setYaw(m_yaw);
                tmpTelemetry.setGPSAlt(m_pd);
                tmpTelemetry.setSpeedEast(0);
                tmpTelemetry.setSpeedNorth(0);
                tmpTelemetry.setTakeOffAlt(0);
                tmpIPCStatus.m_lockMode = (byte)Status::LockMode::LOCK_OFF;
                tmpIPCStatus.m_hfovEO = (data_type)m_hfov[m_sensorID] * M_PI / 180.0;
                tmpIPCStatus.m_sensorMode = (byte)Status::SensorMode::EO;
                tmpMotionCStatus.setPanPos(m_panAngle);
                tmpMotionCStatus.setTiltPos(m_tiltAngle);
                tmpMotionCStatus.setPanVelo(m_panVelocity);
                tmpMotionCStatus.setTiltVelo(m_tiltVelocity);
                state.setTelemetry(tmpTelemetry);
                state.setIPCStatus(tmpIPCStatus);
                state.setMotionCStatus(tmpMotionCStatus);
            }

            map.insert("id", index);
            map.insert("SystemMode", QString::fromStdString(m_lockMode));
            map.insert("hfov", state.getIPCStatus().getHFOV() / M_PI * 180.0);
            map.insert("panPos", state.getMotionCStatus().getPanPos() / M_PI * 180.0);
            map.insert("panVel", state.getMotionCStatus().getPanVelo() /*/M_PI*180.0*/);
            map.insert("tiltPos", state.getMotionCStatus().getTiltPos() / M_PI * 180.0);
            map.insert("tiltVel", state.getMotionCStatus().getTiltVelo() /*/M_PI*180.0*/);
            map.insert("pn", state.getTelemetry().getPn());
            map.insert("pe", state.getTelemetry().getPe());
            map.insert("pd", state.getTelemetry().getPd());
            map.insert("roll", state.getTelemetry().getRoll());
            map.insert("pitch", state.getTelemetry().getPitch());
            map.insert("yaw", state.getTelemetry().getYaw());
            map.insert("speedNorth", state.getTelemetry().getSpeedNorth());
            map.insert("speedEast", state.getTelemetry().getSpeedEast());
            map.insert("gpsAlt", state.getTelemetry().getGPSAlt());
            map.insert("takeOffAlt", state.getTelemetry().getTakeOffAlt());
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
            QString sensor = (m_sensorID == 0) ? "EO" : "IR";
            map.insert("SENSOR", sensor);
            QVariantMap mapEOFov;
            mapEOFov.insert("HFOV", m_hfov[0]);
            mapEOFov.insert("VFOV", m_vfov[0]);
            map.insert("EO", mapEOFov);
            QVariantMap mapIRFov;
            mapIRFov.insert("HFOV", m_hfov[1]);
            mapIRFov.insert("VFOV", m_vfov[1]);
            map.insert("IR", mapIRFov);
            QVariantMap mapZoom;
            mapZoom.insert("EO", m_zoom[0]);
            mapZoom.insert("IR", m_zoom[1]);
            map.insert("ZOOM", mapZoom);
            map.insert("STAB_DIGITAL", m_videoStabMode);
            map.insert("LOCK_MODE", QString::fromStdString(m_lockMode));
            // Motion C Status
            map.insert("STAB_GIMBAL", m_gimbalStabMode);
            // Target
            map.insert("CORNER01", QPointF(m_cornerLat[0], m_cornerLon[0]));
            map.insert("CORNER02", QPointF(m_cornerLat[1], m_cornerLon[1]));
            map.insert("CORNER03", QPointF(m_cornerLat[2], m_cornerLon[2]));
            map.insert("CORNER04", QPointF(m_cornerLat[3], m_cornerLon[3]));
            map.insert("CENTER", QPointF(m_centerLat, m_centerLon));
            map.insert("GIMBAL_RECORD", m_gimbalRecord);
            map.insert("GIMBAL_MODE", QString::fromStdString(m_gimbalMode));
            map.insert("TRACK_SIZE", m_trackSize);
            map.insert("TRACK_MODE", QString::fromStdString(m_trackMode));
            map.insert("LASER_RANGE", m_laserRange);
            res = QVariant(map);
            return res;
        }

        Q_INVOKABLE void setJoystickState(bool _state)
        {
            m_usingJoystick = _state;
        }
        Q_INVOKABLE void setJoystickPos(float _joystickX, float _joystickY)
        {
            m_joystickX = _joystickX;
            m_joystickY = _joystickY;
        }

    Q_SIGNALS:
        void NotifyPropertyChanged(QString name);
};

#endif // GIMBALCONTEXT_H
