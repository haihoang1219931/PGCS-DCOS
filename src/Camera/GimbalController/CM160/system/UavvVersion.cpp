#include "UavvVersion.h"

VideoProcessorFeatures::VideoProcessorFeatures() {}
VideoProcessorFeatures::~VideoProcessorFeatures() {}
VideoProcessorFeatures::VideoProcessorFeatures(unsigned int videoFeatures)
{
	ApplicationBits = videoFeatures;
	IsEStabUnlocked = (videoFeatures & (int)Stabilisation) == (int)Stabilisation ? true : false;
	IsH264Unlocked = (videoFeatures & (int)H264) == (int)H264 ? true : false;
	IsMTDUnlocked = (videoFeatures & (int)MTD) == (int)MTD ? true : false;
	IsTrackingUnlocked = (videoFeatures & (int)Tracking) == (int)Tracking ? true : false;
	IsMTIUnlocked = (videoFeatures & (int)MTI) == (int)MTI ? true : false;
	IsTelemetryUnlocked = (videoFeatures & (int)Telemetry) == (int)Telemetry ? true : false;
	IsEnhancementUnlocked = (videoFeatures & (int)Enhancement) == (int)Enhancement ? true : false;
	IsBlendingUnlocked = (videoFeatures & (int)Blending) == (int)Blending ? true : false;
	IsMjpegUnlocked = (videoFeatures & (int)Mjpeg) == (int)Mjpeg ? true : false;
	IsRecordingUnlocked = (videoFeatures & (int)Recording) == (int)Recording ? true : false;
	IsKlvUnlocked = (videoFeatures & (int)KLV) == (int)KLV ? true : false;
}

UavvVersion::UavvVersion() {}
UavvVersion::~UavvVersion() {}
UavvVersion::UavvVersion(int gimbalSerialNumber, int gimbalFirmwareVersionMajor, int gimbalFirmwareVersionMinor, int gimbalFirmwareVersionRevision, int gimbalFirmwareVersionBuild, int gimbalHardwareVersionMajor, int gimbalHardwareVersionMinor, GimbalType gimbalModel, int gimbalProtocolVersionMajor, int gimbalProtocolVersionMinor,
	long double videoProcessorSerialNumber, int videoProcessorFirmwareVersionMajor, int videoProcessorFirmwareVersionMinor, int videoProcessorFirmwareVersionRevision, unsigned int videoProcessorFeatures,
	unsigned int gNSSSerialNumberPartA, unsigned int gNSSSerialNumberPartB, unsigned int gNSSSerialNumberPartC, unsigned int gNSSFirmwareVersion, unsigned int gNSSHardwareVersion)
{
	GimbalSerialNumber = gimbalSerialNumber;
	GimbalFirmwareVersionMajor = gimbalFirmwareVersionMajor;
	GimbalFirmwareVersionMinor = gimbalFirmwareVersionMinor;
	GimbalFirmwareVersionRevision = gimbalFirmwareVersionRevision;
	GimbalFirmwareVersionBuild = gimbalFirmwareVersionBuild;
	GimbalHardwareVersionMajor = gimbalHardwareVersionMajor;
	GimbalHardwareVersionMinor = gimbalHardwareVersionMinor;
	GimbalModel = gimbalModel;
	GimbalProtocolVersionMajor = gimbalProtocolVersionMajor;
	GimbalProtocolVersionMinor = gimbalProtocolVersionMinor;
	VideoProcessorSerialNumber = videoProcessorSerialNumber;
	VideoProcessorFirmwareVersionMajor = videoProcessorFirmwareVersionMajor;
	VideoProcessorFirmwareVersionMinor = videoProcessorFirmwareVersionMinor;
	VideoProcessorFirmwareVersionRevision = videoProcessorFirmwareVersionRevision;

    VideoProcessor = VideoProcessorFeatures(videoProcessorFeatures);

	GNSSSerialNumberPartA = gNSSSerialNumberPartA;
	GNSSSerialNumberPartB = gNSSSerialNumberPartB;
	GNSSSerialNumberPartC = gNSSSerialNumberPartC;
	GNSSFirmwareVersion = gNSSFirmwareVersion;
	GNSSHardwareVersion = gNSSHardwareVersion;
}

ParseResult UavvVersion::TryParse(GimbalPacket packet, UavvVersion *result)
{
    if (packet.Data.size() < result->Length)
	{
        return ParseResult::InvalidLength;
	}
	int gimbalSerialNumber;
	int gimbalFirmwareVersionMajor;
	int gimbalFirmwareVersionMinor;
	int gimbalFirmwareVersionRevision;
	int gimbalFirmwareVersionBuild;
	int gimbalHardwareVersionMajor;
	int gimbalHardwareVersionMinor;
	GimbalType gimbalModel;
	int gimbalProtocolVersionMajor;
	int gimbalProtocolVersionMinor;
	double videoProcessorSerialNumber;
	int videoProcessorFirmwareVersionMajor;
	int videoProcessorFirmwareVersionMinor;
	int videoProcessorFirmwareVersionRevision;
	unsigned int videoProcessorFeatures;
	unsigned int gNSSSerialNumberPartA;
	unsigned int gNSSSerialNumberPartB;
	unsigned int gNSSSerialNumberPartC;
	unsigned int gNSSFirmwareVersion;
	unsigned int gNSSHardwareVersion;
    gimbalSerialNumber = ByteManipulation::ToUInt16(packet.Data.data(), 0, Endianness::Big);
	gimbalFirmwareVersionMajor = packet.Data[2];
	gimbalFirmwareVersionMinor = packet.Data[3];
	gimbalFirmwareVersionRevision = packet.Data[4];
    gimbalFirmwareVersionBuild = ByteManipulation::ToUInt16(packet.Data.data(), 5, Endianness::Big);
	gimbalHardwareVersionMajor = packet.Data[7];
	gimbalHardwareVersionMinor = packet.Data[8];
	gimbalModel = (GimbalType)packet.Data[9];
	gimbalProtocolVersionMajor = packet.Data[10];
	gimbalProtocolVersionMinor = packet.Data[11];
    videoProcessorSerialNumber = ByteManipulation::ToUInt64(packet.Data.data(), 12, Endianness::Big);
	videoProcessorFirmwareVersionMajor = packet.Data[20];
	videoProcessorFirmwareVersionMinor = packet.Data[21];
	videoProcessorFirmwareVersionRevision = packet.Data[22];
    videoProcessorFeatures = ByteManipulation::ToUInt32(packet.Data.data(), 23, Endianness::Big);
    gNSSSerialNumberPartA = ByteManipulation::ToUInt32(packet.Data.data(), 27, Endianness::Big);
    gNSSSerialNumberPartB = ByteManipulation::ToUInt32(packet.Data.data(), 31, Endianness::Big);
    gNSSSerialNumberPartC = ByteManipulation::ToUInt32(packet.Data.data(), 35, Endianness::Big);
    gNSSFirmwareVersion = ByteManipulation::ToUInt32(packet.Data.data(), 39, Endianness::Big);
    gNSSHardwareVersion = ByteManipulation::ToUInt32(packet.Data.data(), 43, Endianness::Big);
	*result = UavvVersion(gimbalSerialNumber, gimbalFirmwareVersionMajor, gimbalFirmwareVersionMinor, gimbalFirmwareVersionRevision, gimbalFirmwareVersionBuild, gimbalHardwareVersionMajor, gimbalHardwareVersionMinor, gimbalModel, gimbalProtocolVersionMajor, gimbalProtocolVersionMinor,
		videoProcessorSerialNumber, videoProcessorFirmwareVersionMajor, videoProcessorFirmwareVersionMinor, videoProcessorFirmwareVersionRevision, videoProcessorFeatures,
		gNSSSerialNumberPartA, gNSSSerialNumberPartB, gNSSSerialNumberPartC, gNSSFirmwareVersion, gNSSHardwareVersion);
    return ParseResult::Success;
}
