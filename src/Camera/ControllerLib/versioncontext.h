#ifndef VERSIONCONTEXT_H
#define VERSIONCONTEXT_H

#include <stdio.h>
#include <iostream>
#include <QObject>
#include "UavvGimbalProtocol.h"
using namespace std;

class VersionNumber2Value
{
public:
    int Major;
    int Minor;
    VersionNumber2Value(){}
    ~VersionNumber2Value(){}
    VersionNumber2Value(int major, int minor)
    {
        Major = major;
        Minor = minor;
    }

    string ToString()
    {

        return string("{")+std::to_string(Major)+string("}.") +
                string("{")+std::to_string(Minor)+string("}.");
    }
};
class VersionNumber4Value
{
public:
    int Major;
    int Minor;
    int Revision;
    int Build;
    VersionNumber4Value(){}
    ~VersionNumber4Value(){}
    VersionNumber4Value(int major, int minor, int revision, int build)
    {
        Major = major;
        Minor = minor;
        Revision = revision;
        Build = build;
    }

    string ToString()
    {
        return string("{")+std::to_string(Major)+string("}.") +
                string("{")+std::to_string(Minor)+string("}.") +
                string("{")+std::to_string(Revision)+string("}.") +
                string("{")+std::to_string(Build)+string("}.") ;
    }
};
enum AppBits : int
{
    HDOut = 0x01,
    Stabilisation = 0x02,
    H264 = 0x04,
    MTD = 0x08,
    Tracking = 0x10,
    MTI = 0x20,
    Telemetry = 0x40,
    Enhancement = 0x80,
    Blending = 0x100,
    HDIn = 0x200, //formerly Mjeg streaming pre 2.20
    Recording = 0x400,
    KLV = 0x800
};
class ApplicationBits
{
    //[Flags]

public:
    bool IsEStabUnlocked;
    bool IsH264Unlocked;
    bool IsMTDUnlocked;
    bool IsTrackingUnlocked;
    bool IsMTIUnlocked;
    bool IsTelemetryUnlocked;
    bool IsEnhancementUnlocked;
    bool IsBlendingUnlocked;
    bool IsHdInUnlocked;
    bool IsRecordingUnlocked;
    bool IsKlvUnlocked;
    bool IsHdOutUnlocked;
    ApplicationBits(){}
    ~ApplicationBits(){}
    ApplicationBits(uint value)
    {
        IsEStabUnlocked = (value & (int)AppBits::Stabilisation) == (int)AppBits::Stabilisation ? true : false;
        IsH264Unlocked = (value & (int)AppBits::H264) == (int)AppBits::H264 ? true : false;
        IsMTDUnlocked = (value & (int)AppBits::MTD) == (int)AppBits::MTD ? true : false;
        IsTrackingUnlocked = (value & (int)AppBits::Tracking) == (int)AppBits::Tracking ? true : false;
        IsMTIUnlocked = (value & (int)AppBits::MTI) == (int)AppBits::MTI ? true : false;
        IsTelemetryUnlocked = (value & (int)AppBits::Telemetry) == (int)AppBits::Telemetry ? true : false;
        IsEnhancementUnlocked = (value & (int)AppBits::Enhancement) == (int)AppBits::Enhancement ? true : false;
        IsBlendingUnlocked = (value & (int)AppBits::Blending) == (int)AppBits::Blending ? true : false;
        IsHdInUnlocked = ((AppBits)value)&(AppBits::HDIn) == 0 ? false: true;
        IsRecordingUnlocked = (value & (int)AppBits::Recording) == (int)AppBits::Recording ? true : false;
        IsKlvUnlocked = (value & (int)AppBits::KLV) == (int)AppBits::KLV ? true : false;
        IsHdOutUnlocked = ((AppBits)value)&(AppBits::HDOut) == 0 ? false: true;
    }
};
class VersionContext: public QObject
{
    Q_OBJECT
public:
    VersionNumber4Value gimbalFirmwareVersion;
    VersionNumber2Value gimbalHardwareVersion;
    VersionNumber2Value gimbalProtocolVersion;
    uint ahrsSerialNumberPartA;
    uint ahrsSerialNumberPartB;
    uint ahrsSerialNumberPartC;
    int topBoardFirmwareVersionMajor;
    int topBoardFirmwareVersionMinor;
    int topBoardFirmwareVersionRevision;
    int topBoardFirmwareVersionBuild;
    int topBoardHardwareVersionMajor;
    int panBmcFirmwareVersionMajor;
    int panBmcFirmwareVersionMinor;
    int panBmcFirmwareVersionRevision;
    int panBmcFirmwareVersionBuild;
    int panBmcHardwareVersionMajor;
    int panBmcHardwareVersionMinor;
    int tiltBmcFirmwareVersionMajor;
    int tiltBmcFirmwareVersionMinor;
    int tiltBmcFirmwareVersionRevision;
    int tiltBmcFirmwareVersionBuild;
    int tiltBmcHardwareVersionMajor;
    int tiltBmcHardwareVersionMinor;
    int videoProcoessorSoftwareVersionMajor;
    int videoProcessorSoftwareVersionMinor;
    int videoProcessorSoftwareVersionRevision;
    ulong _videoProcessorHardwareID;
    int videoProcoessor2SoftwareVersionMajor;
    int videoProcessor2SoftwareVersionMinor;
    int videoProcessor2SoftwareVersionRevision;
    string _topBoardFirmwareVersion;
    bool _isversionPacketReceived;
    GimbalType gimbalModel;
    ulong _videoProcessor2HardwareID;
    ApplicationBits video1ApplicationBits;
    ApplicationBits video2ApplicationBits;
    string _ahrsSerialNumber;
    int _ahrsFirmwareVersion;
    int _ahrsHardwareVersion;
    int _gimbalSerialNumber;
    string _videoFirmwareVersion;
    string _video2FirmwareVersion;
    string _topBoardVersionNumber;
    string _panMBCFirmwareVersion;
    string _tiltMBCFirmwareVersion;
    bool _isEStabUnlocked;
    bool _isMTDUnlocked;
    bool _isTrackingUnlocked;
    bool _isMTIUnlocked;
    bool _isTelemetryUnlocked;
    bool _isEnhancementUnlocked;
    bool _isBlendingUnlocked;
    bool _isHdInUnlocked;
    bool _isRecordingUnlocked;
    bool _isKlvUnlocked;
    bool _isHdOutUnlocked;
    bool _isH264Unlocked;
    bool _isVideo2MTDUnlocked;
    bool _isVideo2TrackingUnlocked;
    bool _isVideo2MTIUnlocked;
    bool _isVideo2TelemetryUnlocked;
    bool _isVideo2EnhancementUnlocked;
    bool _isVideo2BlendingUnlocked;
    bool _isVideo2HdInUnlocked;
    bool _isVideo2RecordingUnlocked;
    bool _isVideo2KlvUnlocked;
    bool _isVideo2HdOutUnlocked;
    bool _isVideo2H264Unlocked;
    bool _isVideo2EStabUnlocked;
    VersionContext(QObject* parent = 0);
    virtual ~VersionContext();
    void GimbalFirmwareVersion(VersionNumber4Value value)
    {
        gimbalFirmwareVersion = value;
        Q_EMIT NotifyPropertyChanged("GimbalFirmwareVersion");
    }

    void IsversionPacketReceived(bool value)
    {
        _isversionPacketReceived = value;
        Q_EMIT NotifyPropertyChanged("IsversionPacketReceived");

    }

    void GimbalHardwareVersion(VersionNumber2Value value)
    {
        gimbalHardwareVersion = value;
        Q_EMIT NotifyPropertyChanged("GimbalHardwareVersion");
    }

    void GimbalProtocolVersion(VersionNumber2Value value)
    {
        gimbalProtocolVersion = value;
        Q_EMIT NotifyPropertyChanged("GimbalProtocolVersion");
    }

    void AhrsSerialNumberPartA(uint value)
    {
        ahrsSerialNumberPartA = value;
        Q_EMIT NotifyPropertyChanged("AhrsSerialNumberPartA");
    }
    void AhrsSerialNumberPartB(uint value)
    {
        ahrsSerialNumberPartB = value;
        Q_EMIT NotifyPropertyChanged("AhrsSerialNumberPartB");
    }
    void AhrsSerialNumberPartC(uint value)
    {
        ahrsSerialNumberPartC = value;
        Q_EMIT NotifyPropertyChanged("AhrsSerialNumberPartC");
    }
    void TopBoardFirmwareVersionMajor(int value)
    {
        topBoardFirmwareVersionMajor = value;
        Q_EMIT NotifyPropertyChanged("TopBoardFirmwareVersionMajor");
    }
    void TopBoardFirmwareVersionMinor(int value)
    {
        topBoardFirmwareVersionMinor = value;
        Q_EMIT NotifyPropertyChanged("TopBoardFirmwareVersionMinor");
    }
    void TopBoardFirmwareVersionRevision(int value)
    {
        topBoardFirmwareVersionRevision = value;
        Q_EMIT NotifyPropertyChanged("TopBoardFirmwareVersionRevision");
    }
    void TopBoardFirmwareVersionBuild(int value)
    {
        topBoardFirmwareVersionBuild = value;
        Q_EMIT NotifyPropertyChanged("TopBoardFirmwareVersionBuild");
    }
    void TopBoardHardwareVersionMajor(int value)
    {
        topBoardHardwareVersionMajor = value;
        Q_EMIT NotifyPropertyChanged("TopBoardHardwareVersionMajor");
    }
   void PanBmcFirmwareVersionMajor(int value)
    {
       panBmcFirmwareVersionMajor = value;
       Q_EMIT NotifyPropertyChanged("PanBmcFirmwareVersionMajor");
    }
    void PanBmcFirmwareVersionMinor(int value)
    {
        panBmcFirmwareVersionMinor = value;
        Q_EMIT NotifyPropertyChanged("PanBmcFirmwareVersionMinor");
    }
    void PanBmcFirmwareVersionRevision(int value)
    {
        panBmcFirmwareVersionRevision = value;
        Q_EMIT NotifyPropertyChanged("PanBmcFirmwareVersionRevision");
    }

    void PanBmcFirmwareVersionBuild(int value)
    {
        panBmcFirmwareVersionBuild = value;
        Q_EMIT NotifyPropertyChanged("PanBmcFirmwareVersionBuild");
    }

    void PanBmcHardwareVersionMajor(int value)
    {
        panBmcHardwareVersionMajor = value;
        Q_EMIT NotifyPropertyChanged("PanBmcHardwareVersionMajor");
    }

    void PanBmcHardwareVersionMinor(int value)
    {
        panBmcHardwareVersionMinor = value;
        Q_EMIT NotifyPropertyChanged("PanBmcHardwareVersionMinor");
    }

    void TiltBmcFirmwareVersionMajor(int value)
    {
        tiltBmcFirmwareVersionMajor = value;
        Q_EMIT NotifyPropertyChanged("TiltBmcFirmwareVersionMajor");
    }

    void TiltBmcFirmwareVersionMinor(int value)
    {
        tiltBmcFirmwareVersionMinor = value;
        Q_EMIT NotifyPropertyChanged("TiltBmcFirmwareVersionMinor");
    }
    void TiltBmcFirmwareVersionRevision(int value)
    {
        tiltBmcFirmwareVersionRevision = value;
        Q_EMIT NotifyPropertyChanged("TiltBmcFirmwareVersionRevision");
    }
    void TiltBmcFirmwareVersionBuild(int value)
    {
        tiltBmcFirmwareVersionBuild = value;
        Q_EMIT NotifyPropertyChanged("TiltBmcFirmwareVersionBuild");
    }
    void TiltBmcHardwareVersionMajor(int value)
    {
        tiltBmcHardwareVersionMajor = value;
        Q_EMIT NotifyPropertyChanged("TiltBmcHardwareVersionMajor");
    }

    void TiltBmcHardwareVersionMinor(int value)
    {
        tiltBmcHardwareVersionMinor = value;
        Q_EMIT NotifyPropertyChanged("TiltBmcHardwareVersionMinor");
    }

    void VideoProcoessorSoftwareVersionMajor(int value)
    {
        videoProcoessorSoftwareVersionMajor = value;
        Q_EMIT NotifyPropertyChanged("VideoProcoessorSoftwareVersionMajor");
    }

    void VideoProcessorSoftwareVersionMinor(int value)
    {
        videoProcessorSoftwareVersionMinor = value;
        Q_EMIT NotifyPropertyChanged("VideoProcessorSoftwareVersionMinor");
    }

    void VideoProcessorSoftwareVersionRevision(int value)
    {
        videoProcessorSoftwareVersionRevision = value;
        Q_EMIT NotifyPropertyChanged("VideoProcessorSoftwareVersionRevision");
    }


    void VideoProcoessor2SoftwareVersionMajor(int value)
    {
        videoProcoessor2SoftwareVersionMajor = value;
        Q_EMIT NotifyPropertyChanged("VideoProcoessor2SoftwareVersionMajor");
    }

    void VideoProcessor2SoftwareVersionMinor(int value)
    {
        videoProcessor2SoftwareVersionMinor = value;
        Q_EMIT NotifyPropertyChanged("VideoProcessor2SoftwareVersionMinor");
    }
    void VideoProcessor2SoftwareVersionRevision(int value)
    {
        videoProcessor2SoftwareVersionRevision = value;
        Q_EMIT NotifyPropertyChanged("VideoProcessor2SoftwareVersionRevision");
    }
    void TopBoardFirmwareVersion(string value)
    {
        _topBoardFirmwareVersion = value;
        Q_EMIT NotifyPropertyChanged("_topBoardFirmwareVersion");
    }
    void VideoProcessorSerialNumber(ulong value)
    {
        _videoProcessorHardwareID = value;
        Q_EMIT NotifyPropertyChanged("VideoProcessorHardwareID");
    }
    void VideoProcessor2HardwareID(ulong value)
    {
        _videoProcessor2HardwareID = value;
        Q_EMIT NotifyPropertyChanged("VideoProcessor2HardwareID");
    }
    void IsEStabUnlocked(bool value)
    {
        _isEStabUnlocked = value;
        Q_EMIT NotifyPropertyChanged("IsEStabUnlocked");
    }
    void IsH264Unlocked(bool value)
    {
        _isH264Unlocked = value;
        Q_EMIT NotifyPropertyChanged("IsH264Unlocked");
    }

    void IsMTDUnlocked(bool value)
    {
        _isMTDUnlocked = value;
        Q_EMIT NotifyPropertyChanged("IsMTDUnlocked");
    }
   void IsTrackingUnlocked(bool value)
    {
       _isTrackingUnlocked = value;
       Q_EMIT NotifyPropertyChanged("IsTrackingUnlocked");
    }
    void IsMTIUnlocked(bool value)
    {
        _isMTIUnlocked = value;
        Q_EMIT NotifyPropertyChanged("IsMTIUnlocked");
    }
    void IsTelemetryUnlocked(bool value)
    {
        _isTelemetryUnlocked = value;
        Q_EMIT NotifyPropertyChanged("IsTelemetryUnlocked");
    }
    void IsEnhancementUnlocked(bool value)
    {
        _isEnhancementUnlocked = value;
        Q_EMIT NotifyPropertyChanged("IsEnhancementUnlocked");
    }


    void IsBlendingUnlocked(bool value)
    {
        _isBlendingUnlocked = value;
        Q_EMIT NotifyPropertyChanged("IsBlendingUnlocked");
    }
    void IsHdInUnlocked(bool value)
    {
        _isHdInUnlocked = value;
        Q_EMIT NotifyPropertyChanged("IsHdInUnlocked");
    }
    void IsRecordingUnlocked(bool value)
    {
        _isRecordingUnlocked = value;
        Q_EMIT NotifyPropertyChanged("IsRecordingUnlocked");
    }
    void IsKlvUnlocked(bool value)
    {
        _isKlvUnlocked = value;
        Q_EMIT NotifyPropertyChanged("IsKlvUnlocked");
    }

    void IsHdOutUnlocked(bool value)
    {
        _isHdOutUnlocked = value;
        Q_EMIT NotifyPropertyChanged("IsHdOutUnlocked");
    }



    void IsVideo2EStabUnlocked(bool value)
    {
        _isVideo2EStabUnlocked = value;
        Q_EMIT NotifyPropertyChanged("IsVideo2EStabUnlocked");
    }
    void IsVideo2H264Unlocked(bool value)
    {
        _isVideo2H264Unlocked = value;
        Q_EMIT NotifyPropertyChanged("IsVideo2H264Unlocked");
    }


    void IsVideo2MTDUnlocked(bool value)
    {
        _isVideo2MTDUnlocked = value;
        Q_EMIT NotifyPropertyChanged("IsVideo2MTDUnlocked");
    }
    void IsVideo2TrackingUnlocked(bool value)
    {
        _isVideo2TrackingUnlocked = value;
        Q_EMIT NotifyPropertyChanged("IsVideo2TrackingUnlocked");
    }
    void IsVideo2MTIUnlocked(bool value)
    {
        _isVideo2MTIUnlocked = value;
        Q_EMIT NotifyPropertyChanged("IsVideo2MTIUnlocked");
    }
    void IsVideo2TelemetryUnlocked(bool value)
    {
        _isVideo2TelemetryUnlocked = value;
        Q_EMIT NotifyPropertyChanged("IsVideo2TelemetryUnlocked");
    }
    void IsVideo2EnhancementUnlocked(bool value)
    {
        _isVideo2EnhancementUnlocked = value;
        Q_EMIT NotifyPropertyChanged("IsVideo2EnhancementUnlocked");
    }


    void IsVideo2BlendingUnlocked(bool value)
    {
        _isVideo2BlendingUnlocked = value;
        Q_EMIT NotifyPropertyChanged("IsVideo2BlendingUnlocked");
    }
    void IsVideo2HdInUnlocked(bool value)
    {
        _isVideo2HdInUnlocked = value;
        Q_EMIT NotifyPropertyChanged("IsVideo2HdInUnlocked");
    }
    void IsVideo2RecordingUnlocked(bool value)
    {
        _isVideo2RecordingUnlocked = value;
        Q_EMIT NotifyPropertyChanged("IsVideo2RecordingUnlocked");
    }
    void IsVideo2KlvUnlocked(bool value)
    {
        _isVideo2KlvUnlocked = value;
        Q_EMIT NotifyPropertyChanged("IsVideo2KlvUnlocked");
    }

    void IsVideo2HdOutUnlocked(bool value)
    {
        _isVideo2HdOutUnlocked = value;
        Q_EMIT NotifyPropertyChanged("IsVideo2HdOutUnlocked");
    }

    void GimbalModel(GimbalType value)
    {
        gimbalModel = value;
        Q_EMIT NotifyPropertyChanged("GimbalModel");
    }
    void AhrsSerialNumber(string value)
    {
        _ahrsSerialNumber = value;
        Q_EMIT NotifyPropertyChanged("AhrsSerialNumber");
    }
    void AhrsFirmwareVersion(int value)
    {
        _ahrsFirmwareVersion = value;
        Q_EMIT NotifyPropertyChanged("AhrsFirmwareVersion");
    }
    void AhrsHardwareVersion(int value)
    {
        _ahrsHardwareVersion = value;
        Q_EMIT NotifyPropertyChanged("AhrsFirmwareVersion");
    }
    void GimbalSerialNumber(int value)
    {
        _gimbalSerialNumber = value;
        Q_EMIT NotifyPropertyChanged("GimbalSerialNumber");
    }

    void VideoFirmwareVersion(string value)
    {
        _videoFirmwareVersion = value;
        Q_EMIT NotifyPropertyChanged("VideoFirmwareVersion");
    }
    void Video2FirmwareVersion(string value)
    {
        _video2FirmwareVersion = value;
        Q_EMIT NotifyPropertyChanged("Vide2FirmwareVersion");
    }

    void PanBMCFirmwareVersion(string value)
    {
        _panMBCFirmwareVersion = value;
        Q_EMIT NotifyPropertyChanged("PanBMCFirmwareVersion");
    }

    void TiltBMCFirmwareVersion(string value)
    {
        _tiltMBCFirmwareVersion = value;
        Q_EMIT NotifyPropertyChanged("TiltBMCFirmwareVersion");
    }
Q_SIGNALS:
    void NotifyPropertyChanged(QString name);
};

#endif // VERSIONCONTEXT_H
