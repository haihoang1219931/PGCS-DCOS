#pragma once
// MESSAGE PW PACKING

#define MAVLINK_MSG_ID_PW 11026

MAVPACKED(
typedef struct __mavlink_pw_t {
 float VbattA; /*<  Voltage battery A.*/
 float IbattA; /*<  Current battery A.*/
 float EbattA; /*<  Energy battery A.*/
 float VbattB; /*<  Voltage battery B.*/
 float IbattB; /*<  Current battery B.*/
 float EbattB; /*<  Energy battery B.*/
 float Vbatt12S; /*<  Voltage battery 12S.*/
 float pw_temp; /*<  Internal temperature.*/
}) mavlink_pw_t;

#define MAVLINK_MSG_ID_PW_LEN 32
#define MAVLINK_MSG_ID_PW_MIN_LEN 32
#define MAVLINK_MSG_ID_11026_LEN 32
#define MAVLINK_MSG_ID_11026_MIN_LEN 32

#define MAVLINK_MSG_ID_PW_CRC 118
#define MAVLINK_MSG_ID_11026_CRC 118



#if MAVLINK_COMMAND_24BIT
#define MAVLINK_MESSAGE_INFO_PW { \
    11026, \
    "PW", \
    8, \
    {  { "VbattA", NULL, MAVLINK_TYPE_FLOAT, 0, 0, offsetof(mavlink_pw_t, VbattA) }, \
         { "IbattA", NULL, MAVLINK_TYPE_FLOAT, 0, 4, offsetof(mavlink_pw_t, IbattA) }, \
         { "EbattA", NULL, MAVLINK_TYPE_FLOAT, 0, 8, offsetof(mavlink_pw_t, EbattA) }, \
         { "VbattB", NULL, MAVLINK_TYPE_FLOAT, 0, 12, offsetof(mavlink_pw_t, VbattB) }, \
         { "IbattB", NULL, MAVLINK_TYPE_FLOAT, 0, 16, offsetof(mavlink_pw_t, IbattB) }, \
         { "EbattB", NULL, MAVLINK_TYPE_FLOAT, 0, 20, offsetof(mavlink_pw_t, EbattB) }, \
         { "Vbatt12S", NULL, MAVLINK_TYPE_FLOAT, 0, 24, offsetof(mavlink_pw_t, Vbatt12S) }, \
         { "pw_temp", NULL, MAVLINK_TYPE_FLOAT, 0, 28, offsetof(mavlink_pw_t, pw_temp) }, \
         } \
}
#else
#define MAVLINK_MESSAGE_INFO_PW { \
    "PW", \
    8, \
    {  { "VbattA", NULL, MAVLINK_TYPE_FLOAT, 0, 0, offsetof(mavlink_pw_t, VbattA) }, \
         { "IbattA", NULL, MAVLINK_TYPE_FLOAT, 0, 4, offsetof(mavlink_pw_t, IbattA) }, \
         { "EbattA", NULL, MAVLINK_TYPE_FLOAT, 0, 8, offsetof(mavlink_pw_t, EbattA) }, \
         { "VbattB", NULL, MAVLINK_TYPE_FLOAT, 0, 12, offsetof(mavlink_pw_t, VbattB) }, \
         { "IbattB", NULL, MAVLINK_TYPE_FLOAT, 0, 16, offsetof(mavlink_pw_t, IbattB) }, \
         { "EbattB", NULL, MAVLINK_TYPE_FLOAT, 0, 20, offsetof(mavlink_pw_t, EbattB) }, \
         { "Vbatt12S", NULL, MAVLINK_TYPE_FLOAT, 0, 24, offsetof(mavlink_pw_t, Vbatt12S) }, \
         { "pw_temp", NULL, MAVLINK_TYPE_FLOAT, 0, 28, offsetof(mavlink_pw_t, pw_temp) }, \
         } \
}
#endif

/**
 * @brief Pack a pw message
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param msg The MAVLink message to compress the data into
 *
 * @param VbattA  Voltage battery A.
 * @param IbattA  Current battery A.
 * @param EbattA  Energy battery A.
 * @param VbattB  Voltage battery B.
 * @param IbattB  Current battery B.
 * @param EbattB  Energy battery B.
 * @param Vbatt12S  Voltage battery 12S.
 * @param pw_temp  Internal temperature.
 * @return length of the message in bytes (excluding serial stream start sign)
 */
static inline uint16_t mavlink_msg_pw_pack(uint8_t system_id, uint8_t component_id, mavlink_message_t* msg,
                               float VbattA, float IbattA, float EbattA, float VbattB, float IbattB, float EbattB, float Vbatt12S, float pw_temp)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char buf[MAVLINK_MSG_ID_PW_LEN];
    _mav_put_float(buf, 0, VbattA);
    _mav_put_float(buf, 4, IbattA);
    _mav_put_float(buf, 8, EbattA);
    _mav_put_float(buf, 12, VbattB);
    _mav_put_float(buf, 16, IbattB);
    _mav_put_float(buf, 20, EbattB);
    _mav_put_float(buf, 24, Vbatt12S);
    _mav_put_float(buf, 28, pw_temp);

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), buf, MAVLINK_MSG_ID_PW_LEN);
#else
    mavlink_pw_t packet;
    packet.VbattA = VbattA;
    packet.IbattA = IbattA;
    packet.EbattA = EbattA;
    packet.VbattB = VbattB;
    packet.IbattB = IbattB;
    packet.EbattB = EbattB;
    packet.Vbatt12S = Vbatt12S;
    packet.pw_temp = pw_temp;

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), &packet, MAVLINK_MSG_ID_PW_LEN);
#endif

    msg->msgid = MAVLINK_MSG_ID_PW;
    return mavlink_finalize_message(msg, system_id, component_id, MAVLINK_MSG_ID_PW_MIN_LEN, MAVLINK_MSG_ID_PW_LEN, MAVLINK_MSG_ID_PW_CRC);
}

/**
 * @brief Pack a pw message on a channel
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param chan The MAVLink channel this message will be sent over
 * @param msg The MAVLink message to compress the data into
 * @param VbattA  Voltage battery A.
 * @param IbattA  Current battery A.
 * @param EbattA  Energy battery A.
 * @param VbattB  Voltage battery B.
 * @param IbattB  Current battery B.
 * @param EbattB  Energy battery B.
 * @param Vbatt12S  Voltage battery 12S.
 * @param pw_temp  Internal temperature.
 * @return length of the message in bytes (excluding serial stream start sign)
 */
static inline uint16_t mavlink_msg_pw_pack_chan(uint8_t system_id, uint8_t component_id, uint8_t chan,
                               mavlink_message_t* msg,
                                   float VbattA,float IbattA,float EbattA,float VbattB,float IbattB,float EbattB,float Vbatt12S,float pw_temp)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char buf[MAVLINK_MSG_ID_PW_LEN];
    _mav_put_float(buf, 0, VbattA);
    _mav_put_float(buf, 4, IbattA);
    _mav_put_float(buf, 8, EbattA);
    _mav_put_float(buf, 12, VbattB);
    _mav_put_float(buf, 16, IbattB);
    _mav_put_float(buf, 20, EbattB);
    _mav_put_float(buf, 24, Vbatt12S);
    _mav_put_float(buf, 28, pw_temp);

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), buf, MAVLINK_MSG_ID_PW_LEN);
#else
    mavlink_pw_t packet;
    packet.VbattA = VbattA;
    packet.IbattA = IbattA;
    packet.EbattA = EbattA;
    packet.VbattB = VbattB;
    packet.IbattB = IbattB;
    packet.EbattB = EbattB;
    packet.Vbatt12S = Vbatt12S;
    packet.pw_temp = pw_temp;

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), &packet, MAVLINK_MSG_ID_PW_LEN);
#endif

    msg->msgid = MAVLINK_MSG_ID_PW;
    return mavlink_finalize_message_chan(msg, system_id, component_id, chan, MAVLINK_MSG_ID_PW_MIN_LEN, MAVLINK_MSG_ID_PW_LEN, MAVLINK_MSG_ID_PW_CRC);
}

/**
 * @brief Encode a pw struct
 *
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param msg The MAVLink message to compress the data into
 * @param pw C-struct to read the message contents from
 */
static inline uint16_t mavlink_msg_pw_encode(uint8_t system_id, uint8_t component_id, mavlink_message_t* msg, const mavlink_pw_t* pw)
{
    return mavlink_msg_pw_pack(system_id, component_id, msg, pw->VbattA, pw->IbattA, pw->EbattA, pw->VbattB, pw->IbattB, pw->EbattB, pw->Vbatt12S, pw->pw_temp);
}

/**
 * @brief Encode a pw struct on a channel
 *
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param chan The MAVLink channel this message will be sent over
 * @param msg The MAVLink message to compress the data into
 * @param pw C-struct to read the message contents from
 */
static inline uint16_t mavlink_msg_pw_encode_chan(uint8_t system_id, uint8_t component_id, uint8_t chan, mavlink_message_t* msg, const mavlink_pw_t* pw)
{
    return mavlink_msg_pw_pack_chan(system_id, component_id, chan, msg, pw->VbattA, pw->IbattA, pw->EbattA, pw->VbattB, pw->IbattB, pw->EbattB, pw->Vbatt12S, pw->pw_temp);
}

/**
 * @brief Send a pw message
 * @param chan MAVLink channel to send the message
 *
 * @param VbattA  Voltage battery A.
 * @param IbattA  Current battery A.
 * @param EbattA  Energy battery A.
 * @param VbattB  Voltage battery B.
 * @param IbattB  Current battery B.
 * @param EbattB  Energy battery B.
 * @param Vbatt12S  Voltage battery 12S.
 * @param pw_temp  Internal temperature.
 */
#ifdef MAVLINK_USE_CONVENIENCE_FUNCTIONS

static inline void mavlink_msg_pw_send(mavlink_channel_t chan, float VbattA, float IbattA, float EbattA, float VbattB, float IbattB, float EbattB, float Vbatt12S, float pw_temp)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char buf[MAVLINK_MSG_ID_PW_LEN];
    _mav_put_float(buf, 0, VbattA);
    _mav_put_float(buf, 4, IbattA);
    _mav_put_float(buf, 8, EbattA);
    _mav_put_float(buf, 12, VbattB);
    _mav_put_float(buf, 16, IbattB);
    _mav_put_float(buf, 20, EbattB);
    _mav_put_float(buf, 24, Vbatt12S);
    _mav_put_float(buf, 28, pw_temp);

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_PW, buf, MAVLINK_MSG_ID_PW_MIN_LEN, MAVLINK_MSG_ID_PW_LEN, MAVLINK_MSG_ID_PW_CRC);
#else
    mavlink_pw_t packet;
    packet.VbattA = VbattA;
    packet.IbattA = IbattA;
    packet.EbattA = EbattA;
    packet.VbattB = VbattB;
    packet.IbattB = IbattB;
    packet.EbattB = EbattB;
    packet.Vbatt12S = Vbatt12S;
    packet.pw_temp = pw_temp;

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_PW, (const char *)&packet, MAVLINK_MSG_ID_PW_MIN_LEN, MAVLINK_MSG_ID_PW_LEN, MAVLINK_MSG_ID_PW_CRC);
#endif
}

/**
 * @brief Send a pw message
 * @param chan MAVLink channel to send the message
 * @param struct The MAVLink struct to serialize
 */
static inline void mavlink_msg_pw_send_struct(mavlink_channel_t chan, const mavlink_pw_t* pw)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    mavlink_msg_pw_send(chan, pw->VbattA, pw->IbattA, pw->EbattA, pw->VbattB, pw->IbattB, pw->EbattB, pw->Vbatt12S, pw->pw_temp);
#else
    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_PW, (const char *)pw, MAVLINK_MSG_ID_PW_MIN_LEN, MAVLINK_MSG_ID_PW_LEN, MAVLINK_MSG_ID_PW_CRC);
#endif
}

#if MAVLINK_MSG_ID_PW_LEN <= MAVLINK_MAX_PAYLOAD_LEN
/*
  This varient of _send() can be used to save stack space by re-using
  memory from the receive buffer.  The caller provides a
  mavlink_message_t which is the size of a full mavlink message. This
  is usually the receive buffer for the channel, and allows a reply to an
  incoming message with minimum stack space usage.
 */
static inline void mavlink_msg_pw_send_buf(mavlink_message_t *msgbuf, mavlink_channel_t chan,  float VbattA, float IbattA, float EbattA, float VbattB, float IbattB, float EbattB, float Vbatt12S, float pw_temp)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char *buf = (char *)msgbuf;
    _mav_put_float(buf, 0, VbattA);
    _mav_put_float(buf, 4, IbattA);
    _mav_put_float(buf, 8, EbattA);
    _mav_put_float(buf, 12, VbattB);
    _mav_put_float(buf, 16, IbattB);
    _mav_put_float(buf, 20, EbattB);
    _mav_put_float(buf, 24, Vbatt12S);
    _mav_put_float(buf, 28, pw_temp);

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_PW, buf, MAVLINK_MSG_ID_PW_MIN_LEN, MAVLINK_MSG_ID_PW_LEN, MAVLINK_MSG_ID_PW_CRC);
#else
    mavlink_pw_t *packet = (mavlink_pw_t *)msgbuf;
    packet->VbattA = VbattA;
    packet->IbattA = IbattA;
    packet->EbattA = EbattA;
    packet->VbattB = VbattB;
    packet->IbattB = IbattB;
    packet->EbattB = EbattB;
    packet->Vbatt12S = Vbatt12S;
    packet->pw_temp = pw_temp;

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_PW, (const char *)packet, MAVLINK_MSG_ID_PW_MIN_LEN, MAVLINK_MSG_ID_PW_LEN, MAVLINK_MSG_ID_PW_CRC);
#endif
}
#endif

#endif

// MESSAGE PW UNPACKING


/**
 * @brief Get field VbattA from pw message
 *
 * @return  Voltage battery A.
 */
static inline float mavlink_msg_pw_get_VbattA(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  0);
}

/**
 * @brief Get field IbattA from pw message
 *
 * @return  Current battery A.
 */
static inline float mavlink_msg_pw_get_IbattA(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  4);
}

/**
 * @brief Get field EbattA from pw message
 *
 * @return  Energy battery A.
 */
static inline float mavlink_msg_pw_get_EbattA(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  8);
}

/**
 * @brief Get field VbattB from pw message
 *
 * @return  Voltage battery B.
 */
static inline float mavlink_msg_pw_get_VbattB(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  12);
}

/**
 * @brief Get field IbattB from pw message
 *
 * @return  Current battery B.
 */
static inline float mavlink_msg_pw_get_IbattB(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  16);
}

/**
 * @brief Get field EbattB from pw message
 *
 * @return  Energy battery B.
 */
static inline float mavlink_msg_pw_get_EbattB(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  20);
}

/**
 * @brief Get field Vbatt12S from pw message
 *
 * @return  Voltage battery 12S.
 */
static inline float mavlink_msg_pw_get_Vbatt12S(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  24);
}

/**
 * @brief Get field pw_temp from pw message
 *
 * @return  Internal temperature.
 */
static inline float mavlink_msg_pw_get_pw_temp(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  28);
}

/**
 * @brief Decode a pw message into a struct
 *
 * @param msg The message to decode
 * @param pw C-struct to decode the message contents into
 */
static inline void mavlink_msg_pw_decode(const mavlink_message_t* msg, mavlink_pw_t* pw)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    pw->VbattA = mavlink_msg_pw_get_VbattA(msg);
    pw->IbattA = mavlink_msg_pw_get_IbattA(msg);
    pw->EbattA = mavlink_msg_pw_get_EbattA(msg);
    pw->VbattB = mavlink_msg_pw_get_VbattB(msg);
    pw->IbattB = mavlink_msg_pw_get_IbattB(msg);
    pw->EbattB = mavlink_msg_pw_get_EbattB(msg);
    pw->Vbatt12S = mavlink_msg_pw_get_Vbatt12S(msg);
    pw->pw_temp = mavlink_msg_pw_get_pw_temp(msg);
#else
        uint8_t len = msg->len < MAVLINK_MSG_ID_PW_LEN? msg->len : MAVLINK_MSG_ID_PW_LEN;
        memset(pw, 0, MAVLINK_MSG_ID_PW_LEN);
    memcpy(pw, _MAV_PAYLOAD(msg), len);
#endif
}
