#pragma once
// MESSAGE ECU PACKING

#define MAVLINK_MSG_ID_ECU 11027

MAVPACKED(
typedef struct __mavlink_ecu_t {
 float throttle; /*<  Throttle signal (0 to 100%).*/
 uint32_t fuelUsed; /*<  Fuel used in grams.*/
 float CHT; /*<  Cylinder Head Temperature in Celsius degrees.*/
 float fuelPressure; /*<  Fuel pressure in Bar.*/
 uint32_t hobbs; /*<  Engine run time in seconds.*/
 float cpuLoad; /*<  CPU load in percent.*/
 float chargeTemp; /*<  Charge temperature in Celsius degrees.*/
 float flowRate; /*<  Fuel flow rate in grams per minute.*/
 uint16_t rpm; /*<  Engine speed in revolutions per minute.*/
 uint16_t throttlePulse; /*<  Throttle pulse width in microseconds.*/
}) mavlink_ecu_t;

#define MAVLINK_MSG_ID_ECU_LEN 36
#define MAVLINK_MSG_ID_ECU_MIN_LEN 36
#define MAVLINK_MSG_ID_11027_LEN 36
#define MAVLINK_MSG_ID_11027_MIN_LEN 36

#define MAVLINK_MSG_ID_ECU_CRC 38
#define MAVLINK_MSG_ID_11027_CRC 38



#if MAVLINK_COMMAND_24BIT
#define MAVLINK_MESSAGE_INFO_ECU { \
    11027, \
    "ECU", \
    10, \
    {  { "throttle", NULL, MAVLINK_TYPE_FLOAT, 0, 0, offsetof(mavlink_ecu_t, throttle) }, \
         { "rpm", NULL, MAVLINK_TYPE_UINT16_T, 0, 32, offsetof(mavlink_ecu_t, rpm) }, \
         { "fuelUsed", NULL, MAVLINK_TYPE_UINT32_T, 0, 4, offsetof(mavlink_ecu_t, fuelUsed) }, \
         { "throttlePulse", NULL, MAVLINK_TYPE_UINT16_T, 0, 34, offsetof(mavlink_ecu_t, throttlePulse) }, \
         { "CHT", NULL, MAVLINK_TYPE_FLOAT, 0, 8, offsetof(mavlink_ecu_t, CHT) }, \
         { "fuelPressure", NULL, MAVLINK_TYPE_FLOAT, 0, 12, offsetof(mavlink_ecu_t, fuelPressure) }, \
         { "hobbs", NULL, MAVLINK_TYPE_UINT32_T, 0, 16, offsetof(mavlink_ecu_t, hobbs) }, \
         { "cpuLoad", NULL, MAVLINK_TYPE_FLOAT, 0, 20, offsetof(mavlink_ecu_t, cpuLoad) }, \
         { "chargeTemp", NULL, MAVLINK_TYPE_FLOAT, 0, 24, offsetof(mavlink_ecu_t, chargeTemp) }, \
         { "flowRate", NULL, MAVLINK_TYPE_FLOAT, 0, 28, offsetof(mavlink_ecu_t, flowRate) }, \
         } \
}
#else
#define MAVLINK_MESSAGE_INFO_ECU { \
    "ECU", \
    10, \
    {  { "throttle", NULL, MAVLINK_TYPE_FLOAT, 0, 0, offsetof(mavlink_ecu_t, throttle) }, \
         { "rpm", NULL, MAVLINK_TYPE_UINT16_T, 0, 32, offsetof(mavlink_ecu_t, rpm) }, \
         { "fuelUsed", NULL, MAVLINK_TYPE_UINT32_T, 0, 4, offsetof(mavlink_ecu_t, fuelUsed) }, \
         { "throttlePulse", NULL, MAVLINK_TYPE_UINT16_T, 0, 34, offsetof(mavlink_ecu_t, throttlePulse) }, \
         { "CHT", NULL, MAVLINK_TYPE_FLOAT, 0, 8, offsetof(mavlink_ecu_t, CHT) }, \
         { "fuelPressure", NULL, MAVLINK_TYPE_FLOAT, 0, 12, offsetof(mavlink_ecu_t, fuelPressure) }, \
         { "hobbs", NULL, MAVLINK_TYPE_UINT32_T, 0, 16, offsetof(mavlink_ecu_t, hobbs) }, \
         { "cpuLoad", NULL, MAVLINK_TYPE_FLOAT, 0, 20, offsetof(mavlink_ecu_t, cpuLoad) }, \
         { "chargeTemp", NULL, MAVLINK_TYPE_FLOAT, 0, 24, offsetof(mavlink_ecu_t, chargeTemp) }, \
         { "flowRate", NULL, MAVLINK_TYPE_FLOAT, 0, 28, offsetof(mavlink_ecu_t, flowRate) }, \
         } \
}
#endif

/**
 * @brief Pack a ecu message
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param msg The MAVLink message to compress the data into
 *
 * @param throttle  Throttle signal (0 to 100%).
 * @param rpm  Engine speed in revolutions per minute.
 * @param fuelUsed  Fuel used in grams.
 * @param throttlePulse  Throttle pulse width in microseconds.
 * @param CHT  Cylinder Head Temperature in Celsius degrees.
 * @param fuelPressure  Fuel pressure in Bar.
 * @param hobbs  Engine run time in seconds.
 * @param cpuLoad  CPU load in percent.
 * @param chargeTemp  Charge temperature in Celsius degrees.
 * @param flowRate  Fuel flow rate in grams per minute.
 * @return length of the message in bytes (excluding serial stream start sign)
 */
static inline uint16_t mavlink_msg_ecu_pack(uint8_t system_id, uint8_t component_id, mavlink_message_t* msg,
                               float throttle, uint16_t rpm, uint32_t fuelUsed, uint16_t throttlePulse, float CHT, float fuelPressure, uint32_t hobbs, float cpuLoad, float chargeTemp, float flowRate)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char buf[MAVLINK_MSG_ID_ECU_LEN];
    _mav_put_float(buf, 0, throttle);
    _mav_put_uint32_t(buf, 4, fuelUsed);
    _mav_put_float(buf, 8, CHT);
    _mav_put_float(buf, 12, fuelPressure);
    _mav_put_uint32_t(buf, 16, hobbs);
    _mav_put_float(buf, 20, cpuLoad);
    _mav_put_float(buf, 24, chargeTemp);
    _mav_put_float(buf, 28, flowRate);
    _mav_put_uint16_t(buf, 32, rpm);
    _mav_put_uint16_t(buf, 34, throttlePulse);

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), buf, MAVLINK_MSG_ID_ECU_LEN);
#else
    mavlink_ecu_t packet;
    packet.throttle = throttle;
    packet.fuelUsed = fuelUsed;
    packet.CHT = CHT;
    packet.fuelPressure = fuelPressure;
    packet.hobbs = hobbs;
    packet.cpuLoad = cpuLoad;
    packet.chargeTemp = chargeTemp;
    packet.flowRate = flowRate;
    packet.rpm = rpm;
    packet.throttlePulse = throttlePulse;

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), &packet, MAVLINK_MSG_ID_ECU_LEN);
#endif

    msg->msgid = MAVLINK_MSG_ID_ECU;
    return mavlink_finalize_message(msg, system_id, component_id, MAVLINK_MSG_ID_ECU_MIN_LEN, MAVLINK_MSG_ID_ECU_LEN, MAVLINK_MSG_ID_ECU_CRC);
}

/**
 * @brief Pack a ecu message on a channel
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param chan The MAVLink channel this message will be sent over
 * @param msg The MAVLink message to compress the data into
 * @param throttle  Throttle signal (0 to 100%).
 * @param rpm  Engine speed in revolutions per minute.
 * @param fuelUsed  Fuel used in grams.
 * @param throttlePulse  Throttle pulse width in microseconds.
 * @param CHT  Cylinder Head Temperature in Celsius degrees.
 * @param fuelPressure  Fuel pressure in Bar.
 * @param hobbs  Engine run time in seconds.
 * @param cpuLoad  CPU load in percent.
 * @param chargeTemp  Charge temperature in Celsius degrees.
 * @param flowRate  Fuel flow rate in grams per minute.
 * @return length of the message in bytes (excluding serial stream start sign)
 */
static inline uint16_t mavlink_msg_ecu_pack_chan(uint8_t system_id, uint8_t component_id, uint8_t chan,
                               mavlink_message_t* msg,
                                   float throttle,uint16_t rpm,uint32_t fuelUsed,uint16_t throttlePulse,float CHT,float fuelPressure,uint32_t hobbs,float cpuLoad,float chargeTemp,float flowRate)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char buf[MAVLINK_MSG_ID_ECU_LEN];
    _mav_put_float(buf, 0, throttle);
    _mav_put_uint32_t(buf, 4, fuelUsed);
    _mav_put_float(buf, 8, CHT);
    _mav_put_float(buf, 12, fuelPressure);
    _mav_put_uint32_t(buf, 16, hobbs);
    _mav_put_float(buf, 20, cpuLoad);
    _mav_put_float(buf, 24, chargeTemp);
    _mav_put_float(buf, 28, flowRate);
    _mav_put_uint16_t(buf, 32, rpm);
    _mav_put_uint16_t(buf, 34, throttlePulse);

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), buf, MAVLINK_MSG_ID_ECU_LEN);
#else
    mavlink_ecu_t packet;
    packet.throttle = throttle;
    packet.fuelUsed = fuelUsed;
    packet.CHT = CHT;
    packet.fuelPressure = fuelPressure;
    packet.hobbs = hobbs;
    packet.cpuLoad = cpuLoad;
    packet.chargeTemp = chargeTemp;
    packet.flowRate = flowRate;
    packet.rpm = rpm;
    packet.throttlePulse = throttlePulse;

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), &packet, MAVLINK_MSG_ID_ECU_LEN);
#endif

    msg->msgid = MAVLINK_MSG_ID_ECU;
    return mavlink_finalize_message_chan(msg, system_id, component_id, chan, MAVLINK_MSG_ID_ECU_MIN_LEN, MAVLINK_MSG_ID_ECU_LEN, MAVLINK_MSG_ID_ECU_CRC);
}

/**
 * @brief Encode a ecu struct
 *
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param msg The MAVLink message to compress the data into
 * @param ecu C-struct to read the message contents from
 */
static inline uint16_t mavlink_msg_ecu_encode(uint8_t system_id, uint8_t component_id, mavlink_message_t* msg, const mavlink_ecu_t* ecu)
{
    return mavlink_msg_ecu_pack(system_id, component_id, msg, ecu->throttle, ecu->rpm, ecu->fuelUsed, ecu->throttlePulse, ecu->CHT, ecu->fuelPressure, ecu->hobbs, ecu->cpuLoad, ecu->chargeTemp, ecu->flowRate);
}

/**
 * @brief Encode a ecu struct on a channel
 *
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param chan The MAVLink channel this message will be sent over
 * @param msg The MAVLink message to compress the data into
 * @param ecu C-struct to read the message contents from
 */
static inline uint16_t mavlink_msg_ecu_encode_chan(uint8_t system_id, uint8_t component_id, uint8_t chan, mavlink_message_t* msg, const mavlink_ecu_t* ecu)
{
    return mavlink_msg_ecu_pack_chan(system_id, component_id, chan, msg, ecu->throttle, ecu->rpm, ecu->fuelUsed, ecu->throttlePulse, ecu->CHT, ecu->fuelPressure, ecu->hobbs, ecu->cpuLoad, ecu->chargeTemp, ecu->flowRate);
}

/**
 * @brief Send a ecu message
 * @param chan MAVLink channel to send the message
 *
 * @param throttle  Throttle signal (0 to 100%).
 * @param rpm  Engine speed in revolutions per minute.
 * @param fuelUsed  Fuel used in grams.
 * @param throttlePulse  Throttle pulse width in microseconds.
 * @param CHT  Cylinder Head Temperature in Celsius degrees.
 * @param fuelPressure  Fuel pressure in Bar.
 * @param hobbs  Engine run time in seconds.
 * @param cpuLoad  CPU load in percent.
 * @param chargeTemp  Charge temperature in Celsius degrees.
 * @param flowRate  Fuel flow rate in grams per minute.
 */
#ifdef MAVLINK_USE_CONVENIENCE_FUNCTIONS

static inline void mavlink_msg_ecu_send(mavlink_channel_t chan, float throttle, uint16_t rpm, uint32_t fuelUsed, uint16_t throttlePulse, float CHT, float fuelPressure, uint32_t hobbs, float cpuLoad, float chargeTemp, float flowRate)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char buf[MAVLINK_MSG_ID_ECU_LEN];
    _mav_put_float(buf, 0, throttle);
    _mav_put_uint32_t(buf, 4, fuelUsed);
    _mav_put_float(buf, 8, CHT);
    _mav_put_float(buf, 12, fuelPressure);
    _mav_put_uint32_t(buf, 16, hobbs);
    _mav_put_float(buf, 20, cpuLoad);
    _mav_put_float(buf, 24, chargeTemp);
    _mav_put_float(buf, 28, flowRate);
    _mav_put_uint16_t(buf, 32, rpm);
    _mav_put_uint16_t(buf, 34, throttlePulse);

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_ECU, buf, MAVLINK_MSG_ID_ECU_MIN_LEN, MAVLINK_MSG_ID_ECU_LEN, MAVLINK_MSG_ID_ECU_CRC);
#else
    mavlink_ecu_t packet;
    packet.throttle = throttle;
    packet.fuelUsed = fuelUsed;
    packet.CHT = CHT;
    packet.fuelPressure = fuelPressure;
    packet.hobbs = hobbs;
    packet.cpuLoad = cpuLoad;
    packet.chargeTemp = chargeTemp;
    packet.flowRate = flowRate;
    packet.rpm = rpm;
    packet.throttlePulse = throttlePulse;

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_ECU, (const char *)&packet, MAVLINK_MSG_ID_ECU_MIN_LEN, MAVLINK_MSG_ID_ECU_LEN, MAVLINK_MSG_ID_ECU_CRC);
#endif
}

/**
 * @brief Send a ecu message
 * @param chan MAVLink channel to send the message
 * @param struct The MAVLink struct to serialize
 */
static inline void mavlink_msg_ecu_send_struct(mavlink_channel_t chan, const mavlink_ecu_t* ecu)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    mavlink_msg_ecu_send(chan, ecu->throttle, ecu->rpm, ecu->fuelUsed, ecu->throttlePulse, ecu->CHT, ecu->fuelPressure, ecu->hobbs, ecu->cpuLoad, ecu->chargeTemp, ecu->flowRate);
#else
    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_ECU, (const char *)ecu, MAVLINK_MSG_ID_ECU_MIN_LEN, MAVLINK_MSG_ID_ECU_LEN, MAVLINK_MSG_ID_ECU_CRC);
#endif
}

#if MAVLINK_MSG_ID_ECU_LEN <= MAVLINK_MAX_PAYLOAD_LEN
/*
  This varient of _send() can be used to save stack space by re-using
  memory from the receive buffer.  The caller provides a
  mavlink_message_t which is the size of a full mavlink message. This
  is usually the receive buffer for the channel, and allows a reply to an
  incoming message with minimum stack space usage.
 */
static inline void mavlink_msg_ecu_send_buf(mavlink_message_t *msgbuf, mavlink_channel_t chan,  float throttle, uint16_t rpm, uint32_t fuelUsed, uint16_t throttlePulse, float CHT, float fuelPressure, uint32_t hobbs, float cpuLoad, float chargeTemp, float flowRate)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char *buf = (char *)msgbuf;
    _mav_put_float(buf, 0, throttle);
    _mav_put_uint32_t(buf, 4, fuelUsed);
    _mav_put_float(buf, 8, CHT);
    _mav_put_float(buf, 12, fuelPressure);
    _mav_put_uint32_t(buf, 16, hobbs);
    _mav_put_float(buf, 20, cpuLoad);
    _mav_put_float(buf, 24, chargeTemp);
    _mav_put_float(buf, 28, flowRate);
    _mav_put_uint16_t(buf, 32, rpm);
    _mav_put_uint16_t(buf, 34, throttlePulse);

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_ECU, buf, MAVLINK_MSG_ID_ECU_MIN_LEN, MAVLINK_MSG_ID_ECU_LEN, MAVLINK_MSG_ID_ECU_CRC);
#else
    mavlink_ecu_t *packet = (mavlink_ecu_t *)msgbuf;
    packet->throttle = throttle;
    packet->fuelUsed = fuelUsed;
    packet->CHT = CHT;
    packet->fuelPressure = fuelPressure;
    packet->hobbs = hobbs;
    packet->cpuLoad = cpuLoad;
    packet->chargeTemp = chargeTemp;
    packet->flowRate = flowRate;
    packet->rpm = rpm;
    packet->throttlePulse = throttlePulse;

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_ECU, (const char *)packet, MAVLINK_MSG_ID_ECU_MIN_LEN, MAVLINK_MSG_ID_ECU_LEN, MAVLINK_MSG_ID_ECU_CRC);
#endif
}
#endif

#endif

// MESSAGE ECU UNPACKING


/**
 * @brief Get field throttle from ecu message
 *
 * @return  Throttle signal (0 to 100%).
 */
static inline float mavlink_msg_ecu_get_throttle(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  0);
}

/**
 * @brief Get field rpm from ecu message
 *
 * @return  Engine speed in revolutions per minute.
 */
static inline uint16_t mavlink_msg_ecu_get_rpm(const mavlink_message_t* msg)
{
    return _MAV_RETURN_uint16_t(msg,  32);
}

/**
 * @brief Get field fuelUsed from ecu message
 *
 * @return  Fuel used in grams.
 */
static inline uint32_t mavlink_msg_ecu_get_fuelUsed(const mavlink_message_t* msg)
{
    return _MAV_RETURN_uint32_t(msg,  4);
}

/**
 * @brief Get field throttlePulse from ecu message
 *
 * @return  Throttle pulse width in microseconds.
 */
static inline uint16_t mavlink_msg_ecu_get_throttlePulse(const mavlink_message_t* msg)
{
    return _MAV_RETURN_uint16_t(msg,  34);
}

/**
 * @brief Get field CHT from ecu message
 *
 * @return  Cylinder Head Temperature in Celsius degrees.
 */
static inline float mavlink_msg_ecu_get_CHT(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  8);
}

/**
 * @brief Get field fuelPressure from ecu message
 *
 * @return  Fuel pressure in Bar.
 */
static inline float mavlink_msg_ecu_get_fuelPressure(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  12);
}

/**
 * @brief Get field hobbs from ecu message
 *
 * @return  Engine run time in seconds.
 */
static inline uint32_t mavlink_msg_ecu_get_hobbs(const mavlink_message_t* msg)
{
    return _MAV_RETURN_uint32_t(msg,  16);
}

/**
 * @brief Get field cpuLoad from ecu message
 *
 * @return  CPU load in percent.
 */
static inline float mavlink_msg_ecu_get_cpuLoad(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  20);
}

/**
 * @brief Get field chargeTemp from ecu message
 *
 * @return  Charge temperature in Celsius degrees.
 */
static inline float mavlink_msg_ecu_get_chargeTemp(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  24);
}

/**
 * @brief Get field flowRate from ecu message
 *
 * @return  Fuel flow rate in grams per minute.
 */
static inline float mavlink_msg_ecu_get_flowRate(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  28);
}

/**
 * @brief Decode a ecu message into a struct
 *
 * @param msg The message to decode
 * @param ecu C-struct to decode the message contents into
 */
static inline void mavlink_msg_ecu_decode(const mavlink_message_t* msg, mavlink_ecu_t* ecu)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    ecu->throttle = mavlink_msg_ecu_get_throttle(msg);
    ecu->fuelUsed = mavlink_msg_ecu_get_fuelUsed(msg);
    ecu->CHT = mavlink_msg_ecu_get_CHT(msg);
    ecu->fuelPressure = mavlink_msg_ecu_get_fuelPressure(msg);
    ecu->hobbs = mavlink_msg_ecu_get_hobbs(msg);
    ecu->cpuLoad = mavlink_msg_ecu_get_cpuLoad(msg);
    ecu->chargeTemp = mavlink_msg_ecu_get_chargeTemp(msg);
    ecu->flowRate = mavlink_msg_ecu_get_flowRate(msg);
    ecu->rpm = mavlink_msg_ecu_get_rpm(msg);
    ecu->throttlePulse = mavlink_msg_ecu_get_throttlePulse(msg);
#else
        uint8_t len = msg->len < MAVLINK_MSG_ID_ECU_LEN? msg->len : MAVLINK_MSG_ID_ECU_LEN;
        memset(ecu, 0, MAVLINK_MSG_ID_ECU_LEN);
    memcpy(ecu, _MAV_PAYLOAD(msg), len);
#endif
}
