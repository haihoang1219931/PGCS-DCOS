#ifndef UAVVVERSION_H
#define UAVVVERSION_H

#include "../UavvPacket.h"
class VideoProcessorFeatures
{
public:
	enum AppBits : int
	{
		Reserved = 0x01,
		Stabilisation = 0x02,
		H264 = 0x04,
		MTD = 0x08,
		Tracking = 0x10,
		MTI = 0x20,
		Telemetry = 0x40,
		Enhancement = 0x80,
		Blending = 0x100,
		Mjpeg = 0x200,
		Recording = 0x400,
		KLV = 0x800
	};
	unsigned int ApplicationBits;
	bool IsEStabUnlocked;
	bool IsH264Unlocked;
	bool IsMTDUnlocked;
	bool IsTrackingUnlocked;
	bool IsMTIUnlocked;
	bool IsTelemetryUnlocked;
	bool IsEnhancementUnlocked;
	bool IsBlendingUnlocked;
	bool IsMjpegUnlocked;
	bool IsRecordingUnlocked;
	bool IsKlvUnlocked;
	VideoProcessorFeatures();
	~VideoProcessorFeatures();
	VideoProcessorFeatures(unsigned int videoFeatures);


};

class UavvVersion
{
public:
    unsigned int Length = 47;
    int GimbalSerialNumber;
    int GimbalFirmwareVersionMajor;
    int GimbalFirmwareVersionMinor;
    int GimbalFirmwareVersionRevision;
    int GimbalFirmwareVersionBuild;
    int GimbalHardwareVersionMajor;
    int GimbalHardwareVersionMinor;
    GimbalType GimbalModel;
    int GimbalProtocolVersionMajor;
    int GimbalProtocolVersionMinor;
    long double VideoProcessorSerialNumber;
    int VideoProcessorFirmwareVersionMajor;
    int VideoProcessorFirmwareVersionMinor;
    int VideoProcessorFirmwareVersionRevision;
    VideoProcessorFeatures VideoProcessor;
    unsigned int GNSSSerialNumberPartA;
    unsigned int GNSSSerialNumberPartB;
    unsigned int GNSSSerialNumberPartC;
    unsigned int GNSSFirmwareVersion;
    unsigned int GNSSHardwareVersion;
    UavvVersion(int gimbalSerialNumber, int gimbalFirmwareVersionMajor, int gimbalFirmwareVersionMinor, int gimbalFirmwareVersionRevision, int gimbalFirmwareVersionBuild, int gimbalHardwareVersionMajor, int gimbalHardwareVersionMinor, GimbalType gimbalModel, int gimbalProtocolVersionMajor, int gimbalProtocolVersionMinor,
    long double videoProcessorSerialNumber, int videoProcessorFirmwareVersionMajor, int videoProcessorFirmwareVersionMinor, int videoProcessorFirmwareVersionRevision, unsigned int videoProcessorFeatures,
    unsigned int gNSSSerialNumberPartA, unsigned int gNSSSerialNumberPartB, unsigned int gNSSSerialNumberPartC, unsigned int gNSSFirmwareVersion, unsigned int gNSSHardwareVersion);
public:
    UavvVersion();
    ~UavvVersion();
    static ParseResult TryParse(GimbalPacket packet, UavvVersion *result);
};

#endif
