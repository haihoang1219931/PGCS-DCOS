#pragma once
// MESSAGE TRACKER VTP by nhatdn1

#define MAVLINK_MSG_ID_TRACKER_VTP 50000

MAVPACKED(
typedef struct __mavlink_gps_tracker_vtp_t {
 uint64_t time_usec; /*< [us] Timestamp (UNIX Epoch time or time since system boot). The receiving end can infer timestamp format (since 1.1.1970 or since system boot) by checking for the magnitude the number.*/
 int32_t lat; /*< [degE7] Latitude (WGS84, EGM96 ellipsoid)*/
 int32_t lon; /*< [degE7] Longitude (WGS84, EGM96 ellipsoid)*/
 int32_t baro; /*< [mm] Altitude (MSL). Positive for up. Note that virtually all GPS modules provide the MSL altitude in addition to the WGS84 altitude.*/
 double heading;
 int16_t rssi;
 uint8_t status;
 uint32_t result1;
 uint32_t result2;

}) mavlink_tracker_vtp_t;

#define MAVLINK_MSG_ID_TRACKER_VTP_LEN 35
#define MAVLINK_MSG_ID_TRACKER_VTP_MIN_LEN 35
#define MAVLINK_MSG_ID_24_LEN 35
#define MAVLINK_MSG_ID_24_MIN_LEN 35

#define MAVLINK_MSG_ID_TRACKER_VTP_CRC 24
#define MAVLINK_MSG_ID_24_CRC 24




#if MAVLINK_COMMAND_24BIT
#define MAVLINK_MESSAGE_INFO_TRACKER_VTP { \
    50000, \
    "TRACKER_VTP", \
    9, \
    {	 { "time_usec", NULL, MAVLINK_TYPE_UINT64_T, 0, 0, offsetof(mavlink_tracker_vtp_t, time_usec) }, \
         { "lat", NULL, MAVLINK_TYPE_INT32_T, 0, 8, offsetof(mavlink_tracker_vtp_t, lat) }, \
         { "lon", NULL, MAVLINK_TYPE_INT32_T, 0, 12, offsetof(mavlink_tracker_vtp_t, lon) }, \
         { "baro", NULL, MAVLINK_TYPE_INT32_T, 0, 16, offsetof(mavlink_tracker_vtp_t, baro) }, \
         { "heading", NULL, MAVLINK_TYPE_DOUBLE, 0, 20, offsetof(mavlink_tracker_vtp_t, heading) }, \
         { "rssi", NULL, MAVLINK_TYPE_INT16_T, 0, 24, offsetof(mavlink_tracker_vtp_t, rssi) }, \
         { "status", NULL, MAVLINK_TYPE_UINT8_T, 0, 26, offsetof(mavlink_tracker_vtp_t, status) }, \
         { "result1", NULL, MAVLINK_TYPE_UINT32_T, 0, 27, offsetof(mavlink_tracker_vtp_t, result1) }, \
         { "result2", NULL, MAVLINK_TYPE_UINT32_T, 0, 31, offsetof(mavlink_tracker_vtp_t, result2) }, \
         } \
}
#else
#define MAVLINK_MESSAGE_INFO_TRACKER_VTP { \
    "TRACKER_VTP", \
    9, \
    {	 { "time_usec", NULL, MAVLINK_TYPE_UINT64_T, 0, 0, offsetof(mavlink_tracker_vtp_t, time_usec) }, \
         { "lat", NULL, MAVLINK_TYPE_INT32_T, 0, 8, offsetof(mavlink_tracker_vtp_t, lat) }, \
         { "lon", NULL, MAVLINK_TYPE_INT32_T, 0, 12, offsetof(mavlink_tracker_vtp_t, lon) }, \
         { "baro", NULL, MAVLINK_TYPE_INT32_T, 0, 16, offsetof(mavlink_tracker_vtp_t, baro) }, \
         { "heading", NULL, MAVLINK_TYPE_DOUBLE, 0, 20, offsetof(mavlink_tracker_vtp_t, heading) }, \
         { "rssi", NULL, MAVLINK_TYPE_INT16_T, 0, 24, offsetof(mavlink_tracker_vtp_t, rssi) }, \
         { "status", NULL, MAVLINK_TYPE_UINT8_T, 0, 26, offsetof(mavlink_tracker_vtp_t, status) }, \
         { "result1", NULL, MAVLINK_TYPE_UINT32_T, 0, 27, offsetof(mavlink_tracker_vtp_t, result1) }, \
         { "result2", NULL, MAVLINK_TYPE_UINT32_T, 0, 31, offsetof(mavlink_tracker_vtp_t, result2) }, \
         } \
}
#endif

/**
 * @brief Pack a TRACKER_VTP message
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param msg The MAVLink message to compress the data into
 * @param time_usec [us] Timestamp (UNIX Epoch time or time since system boot). The receiving end can infer timestamp format (since 1.1.1970 or since system boot) by checking for the magnitude the number.
 * @param lat [degE7] Latitude (WGS84, EGM96 ellipsoid)
 * @param lon [degE7] Longitude (WGS84, EGM96 ellipsoid)

 */
static inline uint16_t mavlink_msg_tracker_vtp_pack(uint8_t system_id, uint8_t component_id, mavlink_message_t* msg,
                               uint64_t time_usec, int32_t lat, int32_t lon, int32_t baro, double heading, int16_t rssi, uint8_t status, uint32_t result1, uint32_t result2)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char buf[MAVLINK_MSG_ID_TRACKER_VTP_LEN];
    _mav_put_uint64_t(buf, 0, time_usec);
    _mav_put_int32_t(buf, 8, lat);
    _mav_put_int32_t(buf, 12, lon);
    _mav_put_int32_t(buf, 16, baro);
    _mav_put_double(buf, 20, heading);
    _mav_put_int16_t(buf, 24, rssi);
    _mav_put_uint8_t(buf, 26, status);
    _mav_put_uint32_t(buf, 27, result1);
    _mav_put_uint32_t(buf, 31, result2);

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), buf, MAVLINK_MSG_ID_TRACKER_VTP_LEN);
#else
    mavlink_tracker_vtp_t packet;
    packet.time_usec = time_usec;
    packet.lat = lat;
    packet.lon = lon;
    packet.baro=baro;
    packet.heading=heading;
    packet.rssi=rssi;
    packet.status=status;
    packet.result1=result1;
    packet.result2=result2;

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), &packet, MAVLINK_MSG_ID_TRACKER_VTP_LEN);
#endif

    msg->msgid = MAVLINK_MSG_ID_TRACKER_VTP;
    return mavlink_finalize_message(msg, system_id, component_id, MAVLINK_MSG_ID_TRACKER_VTP_MIN_LEN, MAVLINK_MSG_ID_TRACKER_VTP_LEN, MAVLINK_MSG_ID_TRACKER_VTP_CRC);
}

/**
 * @brief Pack a TRACKER_VTP message on a channel
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param chan The MAVLink channel this message will be sent over
 * @param msg The MAVLink message to compress the data into
 * @param time_usec [us] Timestamp (UNIX Epoch time or time since system boot). The receiving end can infer timestamp format (since 1.1.1970 or since system boot) by checking for the magnitude the number.
 * @param lat [degE7] Latitude (WGS84, EGM96 ellipsoid)
 * @param lon [degE7] Longitude (WGS84, EGM96 ellipsoid)

 */
static inline uint16_t mavlink_msg_tracker_vtp_pack_chan(uint8_t system_id, uint8_t component_id, uint8_t chan,
        mavlink_message_t* msg, uint64_t time_usec, int32_t lat, int32_t lon, int32_t baro, double heading,
		int16_t rssi, uint8_t status, uint32_t result1, uint32_t result2)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char buf[MAVLINK_MSG_ID_TRACKER_VTP_LEN];
    _mav_put_uint64_t(buf, 0, time_usec);
    _mav_put_int32_t(buf, 8, lat);
    _mav_put_int32_t(buf, 12, lon);
    _mav_put_int32_t(buf, 16, baro);
    _mav_put_double(buf, 20, heading);
    _mav_put_int16_t(buf, 24, rssi);
    _mav_put_uint8_t(buf, 26, status);
    _mav_put_uint32_t(buf, 27, result1);
    _mav_put_uint32_t(buf, 31, result2);

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), buf, MAVLINK_MSG_ID_TRACKER_VTP_LEN);
#else
    mavlink_tracker_vtp_t packet;

    packet.time_usec = time_usec;
    packet.lat = lat;
    packet.lon = lon;
    packet.baro=baro;
    packet.heading=heading;
    packet.rssi=rssi;
    packet.status=status;
    packet.result1=result1;
    packet.result2=result2;

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), &packet, MAVLINK_MSG_ID_TRACKER_VTP_LEN);
#endif

    msg->msgid = MAVLINK_MSG_ID_TRACKER_VTP;
    return mavlink_finalize_message_chan(msg, system_id, component_id, chan, MAVLINK_MSG_ID_TRACKER_VTP_MIN_LEN, MAVLINK_MSG_ID_TRACKER_VTP_LEN, MAVLINK_MSG_ID_TRACKER_VTP_CRC);
}

/**
 * @brief Encode a TRACKER_VTP struct
 *
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param msg The MAVLink message to compress the data into
 * @param TRACKER_VTP C-struct to read the message contents from
 */
static inline uint16_t mavlink_msg_tracker_vtp_encode(uint8_t system_id, uint8_t component_id, mavlink_message_t* msg, const mavlink_tracker_vtp_t* tracker_vtp)
{
    return mavlink_msg_tracker_vtp_pack(system_id, component_id, msg, tracker_vtp->time_usec , tracker_vtp->lat, tracker_vtp->lon, tracker_vtp->baro,
    		tracker_vtp->heading, tracker_vtp->rssi, tracker_vtp->status, tracker_vtp->result1, tracker_vtp->result2);
}

/**
 * @brief Encode a TRACKER_VTP struct on a channel
 *
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param chan The MAVLink channel this message will be sent over
 * @param msg The MAVLink message to compress the data into
 * @param TRACKER_VTP C-struct to read the message contents from
 */
static inline uint16_t mavlink_msg_tracker_vtp_encode_chan(uint8_t system_id, uint8_t component_id, uint8_t chan, mavlink_message_t* msg, const mavlink_tracker_vtp_t* tracker_vtp)
{
    return mavlink_msg_tracker_vtp_pack_chan(system_id, component_id, chan, msg, tracker_vtp->time_usec , tracker_vtp->lat, tracker_vtp->lon, tracker_vtp->baro, tracker_vtp->heading, tracker_vtp->rssi, tracker_vtp->status, tracker_vtp->result1, tracker_vtp->result2);
}

/**
 * @brief Send a TRACKER_VTP message
 * @param chan MAVLink channel to send the message
 *
 * @param time_usec [us] Timestamp (UNIX Epoch time or time since system boot). The receiving end can infer timestamp format (since 1.1.1970 or since system boot) by checking for the magnitude the number.
 * @param lat [degE7] Latitude (WGS84, EGM96 ellipsoid)
 * @param lon [degE7] Longitude (WGS84, EGM96 ellipsoid)

 */
#ifdef MAVLINK_USE_CONVENIENCE_FUNCTIONS

static inline void mavlink_msg_tracker_vtp_send(mavlink_channel_t chan, uint64_t time_usec, int32_t lat, int32_t lon, int32_t baro, double heading,
		int16_t rssi, uint8_t status, uint32_t result1, uint32_t result2)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char buf[MAVLINK_MSG_ID_TRACKER_VTP_LEN];

     _mav_put_uint64_t(buf, 0, time_usec);
     _mav_put_int32_t(buf, 8, lat);
     _mav_put_int32_t(buf, 12, lon);
     _mav_put_int32_t(buf, 16, baro);
     _mav_put_double(buf, 20, heading);
     _mav_put_int16_t(buf, 24, rssi);
     _mav_put_uint8_t(buf, 26, status);
     _mav_put_uint32_t(buf, 27, result1);
     _mav_put_uint32_t(buf, 31, result2);

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_TRACKER_VTP, buf, MAVLINK_MSG_ID_TRACKER_VTP_MIN_LEN, MAVLINK_MSG_ID_TRACKER_VTP_LEN, MAVLINK_MSG_ID_TRACKER_VTP_CRC);
#else
    mavlink_tracker_vtp_t packet;

    packet.time_usec = time_usec;
    packet.lat = lat;
    packet.lon = lon;
    packet.baro=baro;
    packet.heading=heading;
    packet.rssi=rssi;
    packet.status=status;
    packet.result1=result1;
    packet.result2=result2;

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_TRACKER_VTP, (const char *)&packet, MAVLINK_MSG_ID_TRACKER_VTP_MIN_LEN, MAVLINK_MSG_ID_TRACKER_VTP_LEN, MAVLINK_MSG_ID_TRACKER_VTP_CRC);
#endif
}

/**
 * @brief Send a TRACKER_VTP message
 * @param chan MAVLink channel to send the message
 * @param struct The MAVLink struct to serialize
 */
static inline void mavlink_msg_tracker_vtp_send_struct(mavlink_channel_t chan, const mavlink_tracker_vtp_t* tracker_vtp)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    mavlink_msg_tracker_vtp_send(chan, tracker_vtp->time_usec , tracker_vtp->lat, tracker_vtp->lon, tracker_vtp->baro, tracker_vtp->heading, tracker_vtp->rssi, tracker_vtp->status, tracker_vtp->result1, tracker_vtp->result2);
#else
    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_TRACKER_VTP, (const char *)tracker_vtp, MAVLINK_MSG_ID_TRACKER_VTP_MIN_LEN, MAVLINK_MSG_ID_TRACKER_VTP_LEN, MAVLINK_MSG_ID_TRACKER_VTP_CRC);
#endif
}

#if MAVLINK_MSG_ID_TRACKER_VTP_LEN <= MAVLINK_MAX_PAYLOAD_LEN
/*
  This varient of _send() can be used to save stack space by re-using
  memory from the receive buffer.  The caller provides a
  mavlink_message_t which is the size of a full mavlink message. This
  is usually the receive buffer for the channel, and allows a reply to an
  incoming message with minimum stack space usage.
 */
static inline void mavlink_msg_tracker_vtp_send_buf(mavlink_message_t *msgbuf, mavlink_channel_t chan,  uint64_t time_usec, int32_t lat, int32_t lon, int32_t baro, double heading,
		int16_t rssi, uint8_t status, uint32_t result1, uint32_t result2)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char *buf = (char *)msgbuf;
    _mav_put_uint64_t(buf, 0, time_usec);
    _mav_put_int32_t(buf, 8, lat);
    _mav_put_int32_t(buf, 12, lon);
    _mav_put_int32_t(buf, 16, baro);
    _mav_put_double(buf, 20, heading);
    _mav_put_int16_t(buf, 24, rssi);
    _mav_put_uint8_t(buf, 26, status);
    _mav_put_uint32_t(buf, 27, result1);
    _mav_put_uint32_t(buf, 31, result2);


    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_TRACKER_VTP, buf, MAVLINK_MSG_ID_TRACKER_VTP_MIN_LEN, MAVLINK_MSG_ID_TRACKER_VTP_LEN, MAVLINK_MSG_ID_TRACKER_VTP_CRC);
#else
    mavlink_tracker_vtp_t *packet = (mavlink_tracker_vtp_t *)msgbuf;

    packet.time_usec = time_usec;
    packet.lat = lat;
    packet.lon = lon;
    packet.baro=baro;
    packet.heading=heading;
    packet.rssi=rssi;
    packet.status=status;
    packet.result1=result1;
    packet.result2=result2;

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_TRACKER_VTP, (const char *)packet, MAVLINK_MSG_ID_TRACKER_VTP_MIN_LEN, MAVLINK_MSG_ID_TRACKER_VTP_LEN, MAVLINK_MSG_ID_TRACKER_VTP_CRC);
#endif
}
#endif

#endif

// MESSAGE TRACKER_VTP UNPACKING


/**
 * @brief Get field time_usec from TRACKER_VTP message
 *
 * @return [us] Timestamp (UNIX Epoch time or time since system boot). The receiving end can infer timestamp format (since 1.1.1970 or since system boot) by checking for the magnitude the number.
 */
static inline uint64_t mavlink_msg_tracker_vtp_get_time_usec(const mavlink_message_t* msg)
{
    return _MAV_RETURN_uint64_t(msg,  0);
}


/**
 * @brief Get field lat from TRACKER_VTP message
 *
 * @return [degE7] Latitude (WGS84, EGM96 ellipsoid)
 */
static inline int32_t mavlink_msg_tracker_vtp_get_lat(const mavlink_message_t* msg)
{
    return _MAV_RETURN_int32_t(msg,  8);
}

/**
 * @brief Get field lon from TRACKER_VTP message
 *
 * @return [degE7] Longitude (WGS84, EGM96 ellipsoid)
 */
static inline int32_t mavlink_msg_tracker_vtp_get_lon(const mavlink_message_t* msg)
{
    return _MAV_RETURN_int32_t(msg,  12);
}

/**
 * @brief Get baro from TRACKER_VTP message
 *
 * @return barometter.
 */
static inline int32_t mavlink_msg_tracker_vtp_get_baro(const mavlink_message_t* msg)
{
    return _MAV_RETURN_int32_t(msg,  16);
}

/**
 * @brief Get heading from TRACKER_VTP message
 *
 * @return  tracker heading
 */
static inline double mavlink_msg_tracker_vtp_get_heading(const mavlink_message_t* msg)
{
    return _MAV_RETURN_double_t(msg,  20);
}

/**
 * @brief Get RSSI from TRACKER_VTP message
 *
 * @return RSSI
 */
static inline int16_t mavlink_msg_tracker_vtp_get_rssi(const mavlink_message_t* msg)
{
    return _MAV_RETURN_int16_t(msg,  24);
}

/**
 * @brief Get status from TRACKER_VTP message
 *
 * @return status
 */
static inline uint8_t mavlink_msg_tracker_vtp_get_status(const mavlink_message_t* msg)
{
    return _MAV_RETURN_uint8_t(msg,  26);
}

/**
 * @brief Get result1 from TRACKER_VTP message
 *
 * @return result1
 */
static inline uint32_t mavlink_msg_tracker_vtp_get_result1(const mavlink_message_t* msg)
{
    return _MAV_RETURN_uint32_t(msg,  27);
}

/**
 * @brief Get result2 from TRACKER_VTP message
 *
 * @return result2
 */
static inline uint32_t mavlink_msg_tracker_vtp_get_result2(const mavlink_message_t* msg)
{
    return _MAV_RETURN_uint32_t(msg,  31);
}


/**
 * @brief Decode a TRACKER_VTP message into a struct
 *
 * @param msg The message to decode
 * @param TRACKER_VTP C-struct to decode the message contents into
 */
static inline void mavlink_msg_tracker_vtp_decode(const mavlink_message_t* msg, mavlink_tracker_vtp_t* tracker_vtp)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
	tracker_vtp->time_usec = mavlink_msg_tracker_vtp_get_time_usec(msg);
	tracker_vtp->lat     = mavlink_msg_tracker_vtp_get_lat(msg);
	tracker_vtp->lon     = mavlink_msg_tracker_vtp_get_lon(msg);
	tracker_vtp->baro    = mavlink_msg_tracker_vtp_get_baro(msg);
	tracker_vtp->heading = mavlink_msg_tracker_vtp_get_heading(msg);
	tracker_vtp->rssi    = mavlink_msg_tracker_vtp_get_rssi(msg);
	tracker_vtp->status  = mavlink_msg_tracker_vtp_get_status(msg);
	tracker_vtp->result1 = mavlink_msg_tracker_vtp_get_result1(msg);
	tracker_vtp->result2 = mavlink_msg_tracker_vtp_get_result2(msg);

#else
        uint8_t len = msg->len < MAVLINK_MSG_ID_TRACKER_VTP_LEN? msg->len : MAVLINK_MSG_ID_TRACKER_VTP_LEN;
        memset(tracker_vtp, 0, MAVLINK_MSG_ID_TRACKER_VTP_LEN);
    memcpy(tracker_vtp, _MAV_PAYLOAD(msg), len);
#endif
}
