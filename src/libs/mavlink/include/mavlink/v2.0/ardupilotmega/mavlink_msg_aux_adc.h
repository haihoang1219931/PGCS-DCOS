#pragma once
// MESSAGE AUX_ADC PACKING

#define MAVLINK_MSG_ID_AUX_ADC 11025

MAVPACKED(
typedef struct __mavlink_aux_adc_t {
 float Fuel_level; /*<  Fuel level.*/
 float Raw_fuel_level; /*<  Raw fuel level.*/
 float env_temp; /*<  Environment temperature.*/
 float env_RH; /*<  Environment relative humidity.*/
}) mavlink_aux_adc_t;

#define MAVLINK_MSG_ID_AUX_ADC_LEN 16
#define MAVLINK_MSG_ID_AUX_ADC_MIN_LEN 16
#define MAVLINK_MSG_ID_11025_LEN 16
#define MAVLINK_MSG_ID_11025_MIN_LEN 16

#define MAVLINK_MSG_ID_AUX_ADC_CRC 67
#define MAVLINK_MSG_ID_11025_CRC 67



#if MAVLINK_COMMAND_24BIT
#define MAVLINK_MESSAGE_INFO_AUX_ADC { \
    11025, \
    "AUX_ADC", \
    4, \
    {  { "Fuel_level", NULL, MAVLINK_TYPE_FLOAT, 0, 0, offsetof(mavlink_aux_adc_t, Fuel_level) }, \
         { "Raw_fuel_level", NULL, MAVLINK_TYPE_FLOAT, 0, 4, offsetof(mavlink_aux_adc_t, Raw_fuel_level) }, \
         { "env_temp", NULL, MAVLINK_TYPE_FLOAT, 0, 8, offsetof(mavlink_aux_adc_t, env_temp) }, \
         { "env_RH", NULL, MAVLINK_TYPE_FLOAT, 0, 12, offsetof(mavlink_aux_adc_t, env_RH) }, \
         } \
}
#else
#define MAVLINK_MESSAGE_INFO_AUX_ADC { \
    "AUX_ADC", \
    4, \
    {  { "Fuel_level", NULL, MAVLINK_TYPE_FLOAT, 0, 0, offsetof(mavlink_aux_adc_t, Fuel_level) }, \
         { "Raw_fuel_level", NULL, MAVLINK_TYPE_FLOAT, 0, 4, offsetof(mavlink_aux_adc_t, Raw_fuel_level) }, \
         { "env_temp", NULL, MAVLINK_TYPE_FLOAT, 0, 8, offsetof(mavlink_aux_adc_t, env_temp) }, \
         { "env_RH", NULL, MAVLINK_TYPE_FLOAT, 0, 12, offsetof(mavlink_aux_adc_t, env_RH) }, \
         } \
}
#endif

/**
 * @brief Pack a aux_adc message
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param msg The MAVLink message to compress the data into
 *
 * @param Fuel_level  Fuel level.
 * @param Raw_fuel_level  Raw fuel level.
 * @param env_temp  Environment temperature.
 * @param env_RH  Environment relative humidity.
 * @return length of the message in bytes (excluding serial stream start sign)
 */
static inline uint16_t mavlink_msg_aux_adc_pack(uint8_t system_id, uint8_t component_id, mavlink_message_t* msg,
                               float Fuel_level, float Raw_fuel_level, float env_temp, float env_RH)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char buf[MAVLINK_MSG_ID_AUX_ADC_LEN];
    _mav_put_float(buf, 0, Fuel_level);
    _mav_put_float(buf, 4, Raw_fuel_level);
    _mav_put_float(buf, 8, env_temp);
    _mav_put_float(buf, 12, env_RH);

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), buf, MAVLINK_MSG_ID_AUX_ADC_LEN);
#else
    mavlink_aux_adc_t packet;
    packet.Fuel_level = Fuel_level;
    packet.Raw_fuel_level = Raw_fuel_level;
    packet.env_temp = env_temp;
    packet.env_RH = env_RH;

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), &packet, MAVLINK_MSG_ID_AUX_ADC_LEN);
#endif

    msg->msgid = MAVLINK_MSG_ID_AUX_ADC;
    return mavlink_finalize_message(msg, system_id, component_id, MAVLINK_MSG_ID_AUX_ADC_MIN_LEN, MAVLINK_MSG_ID_AUX_ADC_LEN, MAVLINK_MSG_ID_AUX_ADC_CRC);
}

/**
 * @brief Pack a aux_adc message on a channel
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param chan The MAVLink channel this message will be sent over
 * @param msg The MAVLink message to compress the data into
 * @param Fuel_level  Fuel level.
 * @param Raw_fuel_level  Raw fuel level.
 * @param env_temp  Environment temperature.
 * @param env_RH  Environment relative humidity.
 * @return length of the message in bytes (excluding serial stream start sign)
 */
static inline uint16_t mavlink_msg_aux_adc_pack_chan(uint8_t system_id, uint8_t component_id, uint8_t chan,
                               mavlink_message_t* msg,
                                   float Fuel_level,float Raw_fuel_level,float env_temp,float env_RH)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char buf[MAVLINK_MSG_ID_AUX_ADC_LEN];
    _mav_put_float(buf, 0, Fuel_level);
    _mav_put_float(buf, 4, Raw_fuel_level);
    _mav_put_float(buf, 8, env_temp);
    _mav_put_float(buf, 12, env_RH);

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), buf, MAVLINK_MSG_ID_AUX_ADC_LEN);
#else
    mavlink_aux_adc_t packet;
    packet.Fuel_level = Fuel_level;
    packet.Raw_fuel_level = Raw_fuel_level;
    packet.env_temp = env_temp;
    packet.env_RH = env_RH;

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), &packet, MAVLINK_MSG_ID_AUX_ADC_LEN);
#endif

    msg->msgid = MAVLINK_MSG_ID_AUX_ADC;
    return mavlink_finalize_message_chan(msg, system_id, component_id, chan, MAVLINK_MSG_ID_AUX_ADC_MIN_LEN, MAVLINK_MSG_ID_AUX_ADC_LEN, MAVLINK_MSG_ID_AUX_ADC_CRC);
}

/**
 * @brief Encode a aux_adc struct
 *
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param msg The MAVLink message to compress the data into
 * @param aux_adc C-struct to read the message contents from
 */
static inline uint16_t mavlink_msg_aux_adc_encode(uint8_t system_id, uint8_t component_id, mavlink_message_t* msg, const mavlink_aux_adc_t* aux_adc)
{
    return mavlink_msg_aux_adc_pack(system_id, component_id, msg, aux_adc->Fuel_level, aux_adc->Raw_fuel_level, aux_adc->env_temp, aux_adc->env_RH);
}

/**
 * @brief Encode a aux_adc struct on a channel
 *
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param chan The MAVLink channel this message will be sent over
 * @param msg The MAVLink message to compress the data into
 * @param aux_adc C-struct to read the message contents from
 */
static inline uint16_t mavlink_msg_aux_adc_encode_chan(uint8_t system_id, uint8_t component_id, uint8_t chan, mavlink_message_t* msg, const mavlink_aux_adc_t* aux_adc)
{
    return mavlink_msg_aux_adc_pack_chan(system_id, component_id, chan, msg, aux_adc->Fuel_level, aux_adc->Raw_fuel_level, aux_adc->env_temp, aux_adc->env_RH);
}

/**
 * @brief Send a aux_adc message
 * @param chan MAVLink channel to send the message
 *
 * @param Fuel_level  Fuel level.
 * @param Raw_fuel_level  Raw fuel level.
 * @param env_temp  Environment temperature.
 * @param env_RH  Environment relative humidity.
 */
#ifdef MAVLINK_USE_CONVENIENCE_FUNCTIONS

static inline void mavlink_msg_aux_adc_send(mavlink_channel_t chan, float Fuel_level, float Raw_fuel_level, float env_temp, float env_RH)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char buf[MAVLINK_MSG_ID_AUX_ADC_LEN];
    _mav_put_float(buf, 0, Fuel_level);
    _mav_put_float(buf, 4, Raw_fuel_level);
    _mav_put_float(buf, 8, env_temp);
    _mav_put_float(buf, 12, env_RH);

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_AUX_ADC, buf, MAVLINK_MSG_ID_AUX_ADC_MIN_LEN, MAVLINK_MSG_ID_AUX_ADC_LEN, MAVLINK_MSG_ID_AUX_ADC_CRC);
#else
    mavlink_aux_adc_t packet;
    packet.Fuel_level = Fuel_level;
    packet.Raw_fuel_level = Raw_fuel_level;
    packet.env_temp = env_temp;
    packet.env_RH = env_RH;

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_AUX_ADC, (const char *)&packet, MAVLINK_MSG_ID_AUX_ADC_MIN_LEN, MAVLINK_MSG_ID_AUX_ADC_LEN, MAVLINK_MSG_ID_AUX_ADC_CRC);
#endif
}

/**
 * @brief Send a aux_adc message
 * @param chan MAVLink channel to send the message
 * @param struct The MAVLink struct to serialize
 */
static inline void mavlink_msg_aux_adc_send_struct(mavlink_channel_t chan, const mavlink_aux_adc_t* aux_adc)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    mavlink_msg_aux_adc_send(chan, aux_adc->Fuel_level, aux_adc->Raw_fuel_level, aux_adc->env_temp, aux_adc->env_RH);
#else
    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_AUX_ADC, (const char *)aux_adc, MAVLINK_MSG_ID_AUX_ADC_MIN_LEN, MAVLINK_MSG_ID_AUX_ADC_LEN, MAVLINK_MSG_ID_AUX_ADC_CRC);
#endif
}

#if MAVLINK_MSG_ID_AUX_ADC_LEN <= MAVLINK_MAX_PAYLOAD_LEN
/*
  This varient of _send() can be used to save stack space by re-using
  memory from the receive buffer.  The caller provides a
  mavlink_message_t which is the size of a full mavlink message. This
  is usually the receive buffer for the channel, and allows a reply to an
  incoming message with minimum stack space usage.
 */
static inline void mavlink_msg_aux_adc_send_buf(mavlink_message_t *msgbuf, mavlink_channel_t chan,  float Fuel_level, float Raw_fuel_level, float env_temp, float env_RH)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char *buf = (char *)msgbuf;
    _mav_put_float(buf, 0, Fuel_level);
    _mav_put_float(buf, 4, Raw_fuel_level);
    _mav_put_float(buf, 8, env_temp);
    _mav_put_float(buf, 12, env_RH);

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_AUX_ADC, buf, MAVLINK_MSG_ID_AUX_ADC_MIN_LEN, MAVLINK_MSG_ID_AUX_ADC_LEN, MAVLINK_MSG_ID_AUX_ADC_CRC);
#else
    mavlink_aux_adc_t *packet = (mavlink_aux_adc_t *)msgbuf;
    packet->Fuel_level = Fuel_level;
    packet->Raw_fuel_level = Raw_fuel_level;
    packet->env_temp = env_temp;
    packet->env_RH = env_RH;

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_AUX_ADC, (const char *)packet, MAVLINK_MSG_ID_AUX_ADC_MIN_LEN, MAVLINK_MSG_ID_AUX_ADC_LEN, MAVLINK_MSG_ID_AUX_ADC_CRC);
#endif
}
#endif

#endif

// MESSAGE AUX_ADC UNPACKING


/**
 * @brief Get field Fuel_level from aux_adc message
 *
 * @return  Fuel level.
 */
static inline float mavlink_msg_aux_adc_get_Fuel_level(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  0);
}

/**
 * @brief Get field Raw_fuel_level from aux_adc message
 *
 * @return  Raw fuel level.
 */
static inline float mavlink_msg_aux_adc_get_Raw_fuel_level(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  4);
}

/**
 * @brief Get field env_temp from aux_adc message
 *
 * @return  Environment temperature.
 */
static inline float mavlink_msg_aux_adc_get_env_temp(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  8);
}

/**
 * @brief Get field env_RH from aux_adc message
 *
 * @return  Environment relative humidity.
 */
static inline float mavlink_msg_aux_adc_get_env_RH(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  12);
}

/**
 * @brief Decode a aux_adc message into a struct
 *
 * @param msg The message to decode
 * @param aux_adc C-struct to decode the message contents into
 */
static inline void mavlink_msg_aux_adc_decode(const mavlink_message_t* msg, mavlink_aux_adc_t* aux_adc)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    aux_adc->Fuel_level = mavlink_msg_aux_adc_get_Fuel_level(msg);
    aux_adc->Raw_fuel_level = mavlink_msg_aux_adc_get_Raw_fuel_level(msg);
    aux_adc->env_temp = mavlink_msg_aux_adc_get_env_temp(msg);
    aux_adc->env_RH = mavlink_msg_aux_adc_get_env_RH(msg);
#else
        uint8_t len = msg->len < MAVLINK_MSG_ID_AUX_ADC_LEN? msg->len : MAVLINK_MSG_ID_AUX_ADC_LEN;
        memset(aux_adc, 0, MAVLINK_MSG_ID_AUX_ADC_LEN);
    memcpy(aux_adc, _MAV_PAYLOAD(msg), len);
#endif
}
