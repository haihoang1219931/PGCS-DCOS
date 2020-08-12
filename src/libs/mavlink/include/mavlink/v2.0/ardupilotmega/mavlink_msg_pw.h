#pragma once
// MESSAGE PW PACKING

#define MAVLINK_MSG_ID_PW 11026

MAVPACKED(
typedef struct __mavlink_pw_t {
 float VbattA; /*<  Voltage battery A.*/
 float IbattA; /*<  Current battery A.*/
 float VbattB; /*<  Voltage battery B.*/
 float IbattB; /*<  Current battery B.*/
 float Vgen; /*<  Voltage generator.*/
 float Vavionics; /*<  Voltage avionics.*/
 float Iavionics; /*<  Current avionics.*/
 float Vpayload; /*<  Voltage payload.*/
 float Ipayload; /*<  Current payload.*/
 float Vservo; /*<  Voltage servo.*/
 float Iservo; /*<  Current servo.*/
 float V28DC; /*<  Voltage 28VDC.*/
 float I28DC; /*<  Current 28VDC.*/
 int16_t energyA; /*<  Energy battery A.*/
 int16_t energyB; /*<  Energy battery B.*/
 int8_t pw_temp; /*<  Internal temperature.*/
}) mavlink_pw_t;

#define MAVLINK_MSG_ID_PW_LEN 57
#define MAVLINK_MSG_ID_PW_MIN_LEN 57
#define MAVLINK_MSG_ID_11026_LEN 57
#define MAVLINK_MSG_ID_11026_MIN_LEN 57

#define MAVLINK_MSG_ID_PW_CRC 17
#define MAVLINK_MSG_ID_11026_CRC 17



#if MAVLINK_COMMAND_24BIT
#define MAVLINK_MESSAGE_INFO_PW { \
    11026, \
    "PW", \
    16, \
    {  { "VbattA", NULL, MAVLINK_TYPE_FLOAT, 0, 0, offsetof(mavlink_pw_t, VbattA) }, \
         { "IbattA", NULL, MAVLINK_TYPE_FLOAT, 0, 4, offsetof(mavlink_pw_t, IbattA) }, \
         { "VbattB", NULL, MAVLINK_TYPE_FLOAT, 0, 8, offsetof(mavlink_pw_t, VbattB) }, \
         { "IbattB", NULL, MAVLINK_TYPE_FLOAT, 0, 12, offsetof(mavlink_pw_t, IbattB) }, \
         { "Vgen", NULL, MAVLINK_TYPE_FLOAT, 0, 16, offsetof(mavlink_pw_t, Vgen) }, \
         { "Vavionics", NULL, MAVLINK_TYPE_FLOAT, 0, 20, offsetof(mavlink_pw_t, Vavionics) }, \
         { "Iavionics", NULL, MAVLINK_TYPE_FLOAT, 0, 24, offsetof(mavlink_pw_t, Iavionics) }, \
         { "Vpayload", NULL, MAVLINK_TYPE_FLOAT, 0, 28, offsetof(mavlink_pw_t, Vpayload) }, \
         { "Ipayload", NULL, MAVLINK_TYPE_FLOAT, 0, 32, offsetof(mavlink_pw_t, Ipayload) }, \
         { "Vservo", NULL, MAVLINK_TYPE_FLOAT, 0, 36, offsetof(mavlink_pw_t, Vservo) }, \
         { "Iservo", NULL, MAVLINK_TYPE_FLOAT, 0, 40, offsetof(mavlink_pw_t, Iservo) }, \
         { "V28DC", NULL, MAVLINK_TYPE_FLOAT, 0, 44, offsetof(mavlink_pw_t, V28DC) }, \
         { "I28DC", NULL, MAVLINK_TYPE_FLOAT, 0, 48, offsetof(mavlink_pw_t, I28DC) }, \
         { "pw_temp", NULL, MAVLINK_TYPE_INT8_T, 0, 56, offsetof(mavlink_pw_t, pw_temp) }, \
         { "energyA", NULL, MAVLINK_TYPE_INT16_T, 0, 52, offsetof(mavlink_pw_t, energyA) }, \
         { "energyB", NULL, MAVLINK_TYPE_INT16_T, 0, 54, offsetof(mavlink_pw_t, energyB) }, \
         } \
}
#else
#define MAVLINK_MESSAGE_INFO_PW { \
    "PW", \
    16, \
    {  { "VbattA", NULL, MAVLINK_TYPE_FLOAT, 0, 0, offsetof(mavlink_pw_t, VbattA) }, \
         { "IbattA", NULL, MAVLINK_TYPE_FLOAT, 0, 4, offsetof(mavlink_pw_t, IbattA) }, \
         { "VbattB", NULL, MAVLINK_TYPE_FLOAT, 0, 8, offsetof(mavlink_pw_t, VbattB) }, \
         { "IbattB", NULL, MAVLINK_TYPE_FLOAT, 0, 12, offsetof(mavlink_pw_t, IbattB) }, \
         { "Vgen", NULL, MAVLINK_TYPE_FLOAT, 0, 16, offsetof(mavlink_pw_t, Vgen) }, \
         { "Vavionics", NULL, MAVLINK_TYPE_FLOAT, 0, 20, offsetof(mavlink_pw_t, Vavionics) }, \
         { "Iavionics", NULL, MAVLINK_TYPE_FLOAT, 0, 24, offsetof(mavlink_pw_t, Iavionics) }, \
         { "Vpayload", NULL, MAVLINK_TYPE_FLOAT, 0, 28, offsetof(mavlink_pw_t, Vpayload) }, \
         { "Ipayload", NULL, MAVLINK_TYPE_FLOAT, 0, 32, offsetof(mavlink_pw_t, Ipayload) }, \
         { "Vservo", NULL, MAVLINK_TYPE_FLOAT, 0, 36, offsetof(mavlink_pw_t, Vservo) }, \
         { "Iservo", NULL, MAVLINK_TYPE_FLOAT, 0, 40, offsetof(mavlink_pw_t, Iservo) }, \
         { "V28DC", NULL, MAVLINK_TYPE_FLOAT, 0, 44, offsetof(mavlink_pw_t, V28DC) }, \
         { "I28DC", NULL, MAVLINK_TYPE_FLOAT, 0, 48, offsetof(mavlink_pw_t, I28DC) }, \
         { "pw_temp", NULL, MAVLINK_TYPE_INT8_T, 0, 56, offsetof(mavlink_pw_t, pw_temp) }, \
         { "energyA", NULL, MAVLINK_TYPE_INT16_T, 0, 52, offsetof(mavlink_pw_t, energyA) }, \
         { "energyB", NULL, MAVLINK_TYPE_INT16_T, 0, 54, offsetof(mavlink_pw_t, energyB) }, \
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
 * @param VbattB  Voltage battery B.
 * @param IbattB  Current battery B.
 * @param Vgen  Voltage generator.
 * @param Vavionics  Voltage avionics.
 * @param Iavionics  Current avionics.
 * @param Vpayload  Voltage payload.
 * @param Ipayload  Current payload.
 * @param Vservo  Voltage servo.
 * @param Iservo  Current servo.
 * @param V28DC  Voltage 28VDC.
 * @param I28DC  Current 28VDC.
 * @param pw_temp  Internal temperature.
 * @param energyA  Energy battery A.
 * @param energyB  Energy battery B.
 * @return length of the message in bytes (excluding serial stream start sign)
 */
static inline uint16_t mavlink_msg_pw_pack(uint8_t system_id, uint8_t component_id, mavlink_message_t* msg,
                               float VbattA, float IbattA, float VbattB, float IbattB, float Vgen, float Vavionics, float Iavionics, float Vpayload, float Ipayload, float Vservo, float Iservo, float V28DC, float I28DC, int8_t pw_temp, int16_t energyA, int16_t energyB)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char buf[MAVLINK_MSG_ID_PW_LEN];
    _mav_put_float(buf, 0, VbattA);
    _mav_put_float(buf, 4, IbattA);
    _mav_put_float(buf, 8, VbattB);
    _mav_put_float(buf, 12, IbattB);
    _mav_put_float(buf, 16, Vgen);
    _mav_put_float(buf, 20, Vavionics);
    _mav_put_float(buf, 24, Iavionics);
    _mav_put_float(buf, 28, Vpayload);
    _mav_put_float(buf, 32, Ipayload);
    _mav_put_float(buf, 36, Vservo);
    _mav_put_float(buf, 40, Iservo);
    _mav_put_float(buf, 44, V28DC);
    _mav_put_float(buf, 48, I28DC);
    _mav_put_int16_t(buf, 52, energyA);
    _mav_put_int16_t(buf, 54, energyB);
    _mav_put_int8_t(buf, 56, pw_temp);

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), buf, MAVLINK_MSG_ID_PW_LEN);
#else
    mavlink_pw_t packet;
    packet.VbattA = VbattA;
    packet.IbattA = IbattA;
    packet.VbattB = VbattB;
    packet.IbattB = IbattB;
    packet.Vgen = Vgen;
    packet.Vavionics = Vavionics;
    packet.Iavionics = Iavionics;
    packet.Vpayload = Vpayload;
    packet.Ipayload = Ipayload;
    packet.Vservo = Vservo;
    packet.Iservo = Iservo;
    packet.V28DC = V28DC;
    packet.I28DC = I28DC;
    packet.energyA = energyA;
    packet.energyB = energyB;
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
 * @param VbattB  Voltage battery B.
 * @param IbattB  Current battery B.
 * @param Vgen  Voltage generator.
 * @param Vavionics  Voltage avionics.
 * @param Iavionics  Current avionics.
 * @param Vpayload  Voltage payload.
 * @param Ipayload  Current payload.
 * @param Vservo  Voltage servo.
 * @param Iservo  Current servo.
 * @param V28DC  Voltage 28VDC.
 * @param I28DC  Current 28VDC.
 * @param pw_temp  Internal temperature.
 * @param energyA  Energy battery A.
 * @param energyB  Energy battery B.
 * @return length of the message in bytes (excluding serial stream start sign)
 */
static inline uint16_t mavlink_msg_pw_pack_chan(uint8_t system_id, uint8_t component_id, uint8_t chan,
                               mavlink_message_t* msg,
                                   float VbattA,float IbattA,float VbattB,float IbattB,float Vgen,float Vavionics,float Iavionics,float Vpayload,float Ipayload,float Vservo,float Iservo,float V28DC,float I28DC,int8_t pw_temp,int16_t energyA,int16_t energyB)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char buf[MAVLINK_MSG_ID_PW_LEN];
    _mav_put_float(buf, 0, VbattA);
    _mav_put_float(buf, 4, IbattA);
    _mav_put_float(buf, 8, VbattB);
    _mav_put_float(buf, 12, IbattB);
    _mav_put_float(buf, 16, Vgen);
    _mav_put_float(buf, 20, Vavionics);
    _mav_put_float(buf, 24, Iavionics);
    _mav_put_float(buf, 28, Vpayload);
    _mav_put_float(buf, 32, Ipayload);
    _mav_put_float(buf, 36, Vservo);
    _mav_put_float(buf, 40, Iservo);
    _mav_put_float(buf, 44, V28DC);
    _mav_put_float(buf, 48, I28DC);
    _mav_put_int16_t(buf, 52, energyA);
    _mav_put_int16_t(buf, 54, energyB);
    _mav_put_int8_t(buf, 56, pw_temp);

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), buf, MAVLINK_MSG_ID_PW_LEN);
#else
    mavlink_pw_t packet;
    packet.VbattA = VbattA;
    packet.IbattA = IbattA;
    packet.VbattB = VbattB;
    packet.IbattB = IbattB;
    packet.Vgen = Vgen;
    packet.Vavionics = Vavionics;
    packet.Iavionics = Iavionics;
    packet.Vpayload = Vpayload;
    packet.Ipayload = Ipayload;
    packet.Vservo = Vservo;
    packet.Iservo = Iservo;
    packet.V28DC = V28DC;
    packet.I28DC = I28DC;
    packet.energyA = energyA;
    packet.energyB = energyB;
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
    return mavlink_msg_pw_pack(system_id, component_id, msg, pw->VbattA, pw->IbattA, pw->VbattB, pw->IbattB, pw->Vgen, pw->Vavionics, pw->Iavionics, pw->Vpayload, pw->Ipayload, pw->Vservo, pw->Iservo, pw->V28DC, pw->I28DC, pw->pw_temp, pw->energyA, pw->energyB);
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
    return mavlink_msg_pw_pack_chan(system_id, component_id, chan, msg, pw->VbattA, pw->IbattA, pw->VbattB, pw->IbattB, pw->Vgen, pw->Vavionics, pw->Iavionics, pw->Vpayload, pw->Ipayload, pw->Vservo, pw->Iservo, pw->V28DC, pw->I28DC, pw->pw_temp, pw->energyA, pw->energyB);
}

/**
 * @brief Send a pw message
 * @param chan MAVLink channel to send the message
 *
 * @param VbattA  Voltage battery A.
 * @param IbattA  Current battery A.
 * @param VbattB  Voltage battery B.
 * @param IbattB  Current battery B.
 * @param Vgen  Voltage generator.
 * @param Vavionics  Voltage avionics.
 * @param Iavionics  Current avionics.
 * @param Vpayload  Voltage payload.
 * @param Ipayload  Current payload.
 * @param Vservo  Voltage servo.
 * @param Iservo  Current servo.
 * @param V28DC  Voltage 28VDC.
 * @param I28DC  Current 28VDC.
 * @param pw_temp  Internal temperature.
 * @param energyA  Energy battery A.
 * @param energyB  Energy battery B.
 */
#ifdef MAVLINK_USE_CONVENIENCE_FUNCTIONS

static inline void mavlink_msg_pw_send(mavlink_channel_t chan, float VbattA, float IbattA, float VbattB, float IbattB, float Vgen, float Vavionics, float Iavionics, float Vpayload, float Ipayload, float Vservo, float Iservo, float V28DC, float I28DC, int8_t pw_temp, int16_t energyA, int16_t energyB)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char buf[MAVLINK_MSG_ID_PW_LEN];
    _mav_put_float(buf, 0, VbattA);
    _mav_put_float(buf, 4, IbattA);
    _mav_put_float(buf, 8, VbattB);
    _mav_put_float(buf, 12, IbattB);
    _mav_put_float(buf, 16, Vgen);
    _mav_put_float(buf, 20, Vavionics);
    _mav_put_float(buf, 24, Iavionics);
    _mav_put_float(buf, 28, Vpayload);
    _mav_put_float(buf, 32, Ipayload);
    _mav_put_float(buf, 36, Vservo);
    _mav_put_float(buf, 40, Iservo);
    _mav_put_float(buf, 44, V28DC);
    _mav_put_float(buf, 48, I28DC);
    _mav_put_int16_t(buf, 52, energyA);
    _mav_put_int16_t(buf, 54, energyB);
    _mav_put_int8_t(buf, 56, pw_temp);

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_PW, buf, MAVLINK_MSG_ID_PW_MIN_LEN, MAVLINK_MSG_ID_PW_LEN, MAVLINK_MSG_ID_PW_CRC);
#else
    mavlink_pw_t packet;
    packet.VbattA = VbattA;
    packet.IbattA = IbattA;
    packet.VbattB = VbattB;
    packet.IbattB = IbattB;
    packet.Vgen = Vgen;
    packet.Vavionics = Vavionics;
    packet.Iavionics = Iavionics;
    packet.Vpayload = Vpayload;
    packet.Ipayload = Ipayload;
    packet.Vservo = Vservo;
    packet.Iservo = Iservo;
    packet.V28DC = V28DC;
    packet.I28DC = I28DC;
    packet.energyA = energyA;
    packet.energyB = energyB;
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
    mavlink_msg_pw_send(chan, pw->VbattA, pw->IbattA, pw->VbattB, pw->IbattB, pw->Vgen, pw->Vavionics, pw->Iavionics, pw->Vpayload, pw->Ipayload, pw->Vservo, pw->Iservo, pw->V28DC, pw->I28DC, pw->pw_temp, pw->energyA, pw->energyB);
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
static inline void mavlink_msg_pw_send_buf(mavlink_message_t *msgbuf, mavlink_channel_t chan,  float VbattA, float IbattA, float VbattB, float IbattB, float Vgen, float Vavionics, float Iavionics, float Vpayload, float Ipayload, float Vservo, float Iservo, float V28DC, float I28DC, int8_t pw_temp, int16_t energyA, int16_t energyB)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char *buf = (char *)msgbuf;
    _mav_put_float(buf, 0, VbattA);
    _mav_put_float(buf, 4, IbattA);
    _mav_put_float(buf, 8, VbattB);
    _mav_put_float(buf, 12, IbattB);
    _mav_put_float(buf, 16, Vgen);
    _mav_put_float(buf, 20, Vavionics);
    _mav_put_float(buf, 24, Iavionics);
    _mav_put_float(buf, 28, Vpayload);
    _mav_put_float(buf, 32, Ipayload);
    _mav_put_float(buf, 36, Vservo);
    _mav_put_float(buf, 40, Iservo);
    _mav_put_float(buf, 44, V28DC);
    _mav_put_float(buf, 48, I28DC);
    _mav_put_int16_t(buf, 52, energyA);
    _mav_put_int16_t(buf, 54, energyB);
    _mav_put_int8_t(buf, 56, pw_temp);

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_PW, buf, MAVLINK_MSG_ID_PW_MIN_LEN, MAVLINK_MSG_ID_PW_LEN, MAVLINK_MSG_ID_PW_CRC);
#else
    mavlink_pw_t *packet = (mavlink_pw_t *)msgbuf;
    packet->VbattA = VbattA;
    packet->IbattA = IbattA;
    packet->VbattB = VbattB;
    packet->IbattB = IbattB;
    packet->Vgen = Vgen;
    packet->Vavionics = Vavionics;
    packet->Iavionics = Iavionics;
    packet->Vpayload = Vpayload;
    packet->Ipayload = Ipayload;
    packet->Vservo = Vservo;
    packet->Iservo = Iservo;
    packet->V28DC = V28DC;
    packet->I28DC = I28DC;
    packet->energyA = energyA;
    packet->energyB = energyB;
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
 * @brief Get field VbattB from pw message
 *
 * @return  Voltage battery B.
 */
static inline float mavlink_msg_pw_get_VbattB(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  8);
}

/**
 * @brief Get field IbattB from pw message
 *
 * @return  Current battery B.
 */
static inline float mavlink_msg_pw_get_IbattB(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  12);
}

/**
 * @brief Get field Vgen from pw message
 *
 * @return  Voltage generator.
 */
static inline float mavlink_msg_pw_get_Vgen(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  16);
}

/**
 * @brief Get field Vavionics from pw message
 *
 * @return  Voltage avionics.
 */
static inline float mavlink_msg_pw_get_Vavionics(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  20);
}

/**
 * @brief Get field Iavionics from pw message
 *
 * @return  Current avionics.
 */
static inline float mavlink_msg_pw_get_Iavionics(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  24);
}

/**
 * @brief Get field Vpayload from pw message
 *
 * @return  Voltage payload.
 */
static inline float mavlink_msg_pw_get_Vpayload(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  28);
}

/**
 * @brief Get field Ipayload from pw message
 *
 * @return  Current payload.
 */
static inline float mavlink_msg_pw_get_Ipayload(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  32);
}

/**
 * @brief Get field Vservo from pw message
 *
 * @return  Voltage servo.
 */
static inline float mavlink_msg_pw_get_Vservo(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  36);
}

/**
 * @brief Get field Iservo from pw message
 *
 * @return  Current servo.
 */
static inline float mavlink_msg_pw_get_Iservo(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  40);
}

/**
 * @brief Get field V28DC from pw message
 *
 * @return  Voltage 28VDC.
 */
static inline float mavlink_msg_pw_get_V28DC(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  44);
}

/**
 * @brief Get field I28DC from pw message
 *
 * @return  Current 28VDC.
 */
static inline float mavlink_msg_pw_get_I28DC(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  48);
}

/**
 * @brief Get field pw_temp from pw message
 *
 * @return  Internal temperature.
 */
static inline int8_t mavlink_msg_pw_get_pw_temp(const mavlink_message_t* msg)
{
    return _MAV_RETURN_int8_t(msg,  56);
}

/**
 * @brief Get field energyA from pw message
 *
 * @return  Energy battery A.
 */
static inline int16_t mavlink_msg_pw_get_energyA(const mavlink_message_t* msg)
{
    return _MAV_RETURN_int16_t(msg,  52);
}

/**
 * @brief Get field energyB from pw message
 *
 * @return  Energy battery B.
 */
static inline int16_t mavlink_msg_pw_get_energyB(const mavlink_message_t* msg)
{
    return _MAV_RETURN_int16_t(msg,  54);
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
    pw->VbattB = mavlink_msg_pw_get_VbattB(msg);
    pw->IbattB = mavlink_msg_pw_get_IbattB(msg);
    pw->Vgen = mavlink_msg_pw_get_Vgen(msg);
    pw->Vavionics = mavlink_msg_pw_get_Vavionics(msg);
    pw->Iavionics = mavlink_msg_pw_get_Iavionics(msg);
    pw->Vpayload = mavlink_msg_pw_get_Vpayload(msg);
    pw->Ipayload = mavlink_msg_pw_get_Ipayload(msg);
    pw->Vservo = mavlink_msg_pw_get_Vservo(msg);
    pw->Iservo = mavlink_msg_pw_get_Iservo(msg);
    pw->V28DC = mavlink_msg_pw_get_V28DC(msg);
    pw->I28DC = mavlink_msg_pw_get_I28DC(msg);
    pw->energyA = mavlink_msg_pw_get_energyA(msg);
    pw->energyB = mavlink_msg_pw_get_energyB(msg);
    pw->pw_temp = mavlink_msg_pw_get_pw_temp(msg);
#else
        uint8_t len = msg->len < MAVLINK_MSG_ID_PW_LEN? msg->len : MAVLINK_MSG_ID_PW_LEN;
        memset(pw, 0, MAVLINK_MSG_ID_PW_LEN);
    memcpy(pw, _MAV_PAYLOAD(msg), len);
#endif
}
