#pragma once
// MESSAGE PMU PACKING

#define MAVLINK_MSG_ID_PMU 225

MAVPACKED(
typedef struct __mavlink_pmu_t {
 float VbattA; /*<  Voltage battery A.*/
 float IbattA; /*<  Current battery A.*/
 float VbattB; /*<  Voltage battery B.*/
 float IbattB; /*<  Current battery B.*/
 float Vbatt12S; /*<  Voltage battery 12S.*/
 float Fuel_level; /*<  Fuel level.*/
 float Raw_fuel_level; /*<  Raw fuel level.*/
 float env_temp; /*<  Environment temperature.*/
 float env_RH; /*<  Environment relative humidity.*/
 uint16_t PMU_RPM; /*<  Main engine speed.*/
 uint16_t PMU_data_status; /*<  PMU data status.*/
 uint8_t PMU_temp; /*<  PMU temperature.*/
 int8_t PMU_frame_ok; /*<  PMU frame status.*/
 int8_t PMU_com; /*<  PMU communication.*/
}) mavlink_pmu_t;

#define MAVLINK_MSG_ID_PMU_LEN 43
#define MAVLINK_MSG_ID_PMU_MIN_LEN 43
#define MAVLINK_MSG_ID_225_LEN 43
#define MAVLINK_MSG_ID_225_MIN_LEN 43

#define MAVLINK_MSG_ID_PMU_CRC 128
#define MAVLINK_MSG_ID_225_CRC 128



#if MAVLINK_COMMAND_24BIT
#define MAVLINK_MESSAGE_INFO_PMU { \
    225, \
    "PMU", \
    14, \
    {  { "VbattA", NULL, MAVLINK_TYPE_FLOAT, 0, 0, offsetof(mavlink_pmu_t, VbattA) }, \
         { "IbattA", NULL, MAVLINK_TYPE_FLOAT, 0, 4, offsetof(mavlink_pmu_t, IbattA) }, \
         { "VbattB", NULL, MAVLINK_TYPE_FLOAT, 0, 8, offsetof(mavlink_pmu_t, VbattB) }, \
         { "IbattB", NULL, MAVLINK_TYPE_FLOAT, 0, 12, offsetof(mavlink_pmu_t, IbattB) }, \
         { "Vbatt12S", NULL, MAVLINK_TYPE_FLOAT, 0, 16, offsetof(mavlink_pmu_t, Vbatt12S) }, \
         { "PMU_RPM", NULL, MAVLINK_TYPE_UINT16_T, 0, 36, offsetof(mavlink_pmu_t, PMU_RPM) }, \
         { "PMU_temp", NULL, MAVLINK_TYPE_UINT8_T, 0, 40, offsetof(mavlink_pmu_t, PMU_temp) }, \
         { "PMU_frame_ok", NULL, MAVLINK_TYPE_INT8_T, 0, 41, offsetof(mavlink_pmu_t, PMU_frame_ok) }, \
         { "PMU_com", NULL, MAVLINK_TYPE_INT8_T, 0, 42, offsetof(mavlink_pmu_t, PMU_com) }, \
         { "Fuel_level", NULL, MAVLINK_TYPE_FLOAT, 0, 20, offsetof(mavlink_pmu_t, Fuel_level) }, \
         { "Raw_fuel_level", NULL, MAVLINK_TYPE_FLOAT, 0, 24, offsetof(mavlink_pmu_t, Raw_fuel_level) }, \
         { "env_temp", NULL, MAVLINK_TYPE_FLOAT, 0, 28, offsetof(mavlink_pmu_t, env_temp) }, \
         { "env_RH", NULL, MAVLINK_TYPE_FLOAT, 0, 32, offsetof(mavlink_pmu_t, env_RH) }, \
         { "PMU_data_status", NULL, MAVLINK_TYPE_UINT16_T, 0, 38, offsetof(mavlink_pmu_t, PMU_data_status) }, \
         } \
}
#else
#define MAVLINK_MESSAGE_INFO_PMU { \
    "PMU", \
    14, \
    {  { "VbattA", NULL, MAVLINK_TYPE_FLOAT, 0, 0, offsetof(mavlink_pmu_t, VbattA) }, \
         { "IbattA", NULL, MAVLINK_TYPE_FLOAT, 0, 4, offsetof(mavlink_pmu_t, IbattA) }, \
         { "VbattB", NULL, MAVLINK_TYPE_FLOAT, 0, 8, offsetof(mavlink_pmu_t, VbattB) }, \
         { "IbattB", NULL, MAVLINK_TYPE_FLOAT, 0, 12, offsetof(mavlink_pmu_t, IbattB) }, \
         { "Vbatt12S", NULL, MAVLINK_TYPE_FLOAT, 0, 16, offsetof(mavlink_pmu_t, Vbatt12S) }, \
         { "PMU_RPM", NULL, MAVLINK_TYPE_UINT16_T, 0, 36, offsetof(mavlink_pmu_t, PMU_RPM) }, \
         { "PMU_temp", NULL, MAVLINK_TYPE_UINT8_T, 0, 40, offsetof(mavlink_pmu_t, PMU_temp) }, \
         { "PMU_frame_ok", NULL, MAVLINK_TYPE_INT8_T, 0, 41, offsetof(mavlink_pmu_t, PMU_frame_ok) }, \
         { "PMU_com", NULL, MAVLINK_TYPE_INT8_T, 0, 42, offsetof(mavlink_pmu_t, PMU_com) }, \
         { "Fuel_level", NULL, MAVLINK_TYPE_FLOAT, 0, 20, offsetof(mavlink_pmu_t, Fuel_level) }, \
         { "Raw_fuel_level", NULL, MAVLINK_TYPE_FLOAT, 0, 24, offsetof(mavlink_pmu_t, Raw_fuel_level) }, \
         { "env_temp", NULL, MAVLINK_TYPE_FLOAT, 0, 28, offsetof(mavlink_pmu_t, env_temp) }, \
         { "env_RH", NULL, MAVLINK_TYPE_FLOAT, 0, 32, offsetof(mavlink_pmu_t, env_RH) }, \
         { "PMU_data_status", NULL, MAVLINK_TYPE_UINT16_T, 0, 38, offsetof(mavlink_pmu_t, PMU_data_status) }, \
         } \
}
#endif

/**
 * @brief Pack a pmu message
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param msg The MAVLink message to compress the data into
 *
 * @param VbattA  Voltage battery A.
 * @param IbattA  Current battery A.
 * @param VbattB  Voltage battery B.
 * @param IbattB  Current battery B.
 * @param Vbatt12S  Voltage battery 12S.
 * @param PMU_RPM  Main engine speed.
 * @param PMU_temp  PMU temperature.
 * @param PMU_frame_ok  PMU frame status.
 * @param PMU_com  PMU communication.
 * @param Fuel_level  Fuel level.
 * @param Raw_fuel_level  Raw fuel level.
 * @param env_temp  Environment temperature.
 * @param env_RH  Environment relative humidity.
 * @param PMU_data_status  PMU data status.
 * @return length of the message in bytes (excluding serial stream start sign)
 */
static inline uint16_t mavlink_msg_pmu_pack(uint8_t system_id, uint8_t component_id, mavlink_message_t* msg,
                               float VbattA, float IbattA, float VbattB, float IbattB, float Vbatt12S, uint16_t PMU_RPM, uint8_t PMU_temp, int8_t PMU_frame_ok, int8_t PMU_com, float Fuel_level, float Raw_fuel_level, float env_temp, float env_RH, uint16_t PMU_data_status)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char buf[MAVLINK_MSG_ID_PMU_LEN];
    _mav_put_float(buf, 0, VbattA);
    _mav_put_float(buf, 4, IbattA);
    _mav_put_float(buf, 8, VbattB);
    _mav_put_float(buf, 12, IbattB);
    _mav_put_float(buf, 16, Vbatt12S);
    _mav_put_float(buf, 20, Fuel_level);
    _mav_put_float(buf, 24, Raw_fuel_level);
    _mav_put_float(buf, 28, env_temp);
    _mav_put_float(buf, 32, env_RH);
    _mav_put_uint16_t(buf, 36, PMU_RPM);
    _mav_put_uint16_t(buf, 38, PMU_data_status);
    _mav_put_uint8_t(buf, 40, PMU_temp);
    _mav_put_int8_t(buf, 41, PMU_frame_ok);
    _mav_put_int8_t(buf, 42, PMU_com);

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), buf, MAVLINK_MSG_ID_PMU_LEN);
#else
    mavlink_pmu_t packet;
    packet.VbattA = VbattA;
    packet.IbattA = IbattA;
    packet.VbattB = VbattB;
    packet.IbattB = IbattB;
    packet.Vbatt12S = Vbatt12S;
    packet.Fuel_level = Fuel_level;
    packet.Raw_fuel_level = Raw_fuel_level;
    packet.env_temp = env_temp;
    packet.env_RH = env_RH;
    packet.PMU_RPM = PMU_RPM;
    packet.PMU_data_status = PMU_data_status;
    packet.PMU_temp = PMU_temp;
    packet.PMU_frame_ok = PMU_frame_ok;
    packet.PMU_com = PMU_com;

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), &packet, MAVLINK_MSG_ID_PMU_LEN);
#endif

    msg->msgid = MAVLINK_MSG_ID_PMU;
    return mavlink_finalize_message(msg, system_id, component_id, MAVLINK_MSG_ID_PMU_MIN_LEN, MAVLINK_MSG_ID_PMU_LEN, MAVLINK_MSG_ID_PMU_CRC);
}

/**
 * @brief Pack a pmu message on a channel
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param chan The MAVLink channel this message will be sent over
 * @param msg The MAVLink message to compress the data into
 * @param VbattA  Voltage battery A.
 * @param IbattA  Current battery A.
 * @param VbattB  Voltage battery B.
 * @param IbattB  Current battery B.
 * @param Vbatt12S  Voltage battery 12S.
 * @param PMU_RPM  Main engine speed.
 * @param PMU_temp  PMU temperature.
 * @param PMU_frame_ok  PMU frame status.
 * @param PMU_com  PMU communication.
 * @param Fuel_level  Fuel level.
 * @param Raw_fuel_level  Raw fuel level.
 * @param env_temp  Environment temperature.
 * @param env_RH  Environment relative humidity.
 * @param PMU_data_status  PMU data status.
 * @return length of the message in bytes (excluding serial stream start sign)
 */
static inline uint16_t mavlink_msg_pmu_pack_chan(uint8_t system_id, uint8_t component_id, uint8_t chan,
                               mavlink_message_t* msg,
                                   float VbattA,float IbattA,float VbattB,float IbattB,float Vbatt12S,uint16_t PMU_RPM,uint8_t PMU_temp,int8_t PMU_frame_ok,int8_t PMU_com,float Fuel_level,float Raw_fuel_level,float env_temp,float env_RH,uint16_t PMU_data_status)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char buf[MAVLINK_MSG_ID_PMU_LEN];
    _mav_put_float(buf, 0, VbattA);
    _mav_put_float(buf, 4, IbattA);
    _mav_put_float(buf, 8, VbattB);
    _mav_put_float(buf, 12, IbattB);
    _mav_put_float(buf, 16, Vbatt12S);
    _mav_put_float(buf, 20, Fuel_level);
    _mav_put_float(buf, 24, Raw_fuel_level);
    _mav_put_float(buf, 28, env_temp);
    _mav_put_float(buf, 32, env_RH);
    _mav_put_uint16_t(buf, 36, PMU_RPM);
    _mav_put_uint16_t(buf, 38, PMU_data_status);
    _mav_put_uint8_t(buf, 40, PMU_temp);
    _mav_put_int8_t(buf, 41, PMU_frame_ok);
    _mav_put_int8_t(buf, 42, PMU_com);

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), buf, MAVLINK_MSG_ID_PMU_LEN);
#else
    mavlink_pmu_t packet;
    packet.VbattA = VbattA;
    packet.IbattA = IbattA;
    packet.VbattB = VbattB;
    packet.IbattB = IbattB;
    packet.Vbatt12S = Vbatt12S;
    packet.Fuel_level = Fuel_level;
    packet.Raw_fuel_level = Raw_fuel_level;
    packet.env_temp = env_temp;
    packet.env_RH = env_RH;
    packet.PMU_RPM = PMU_RPM;
    packet.PMU_data_status = PMU_data_status;
    packet.PMU_temp = PMU_temp;
    packet.PMU_frame_ok = PMU_frame_ok;
    packet.PMU_com = PMU_com;

        memcpy(_MAV_PAYLOAD_NON_CONST(msg), &packet, MAVLINK_MSG_ID_PMU_LEN);
#endif

    msg->msgid = MAVLINK_MSG_ID_PMU;
    return mavlink_finalize_message_chan(msg, system_id, component_id, chan, MAVLINK_MSG_ID_PMU_MIN_LEN, MAVLINK_MSG_ID_PMU_LEN, MAVLINK_MSG_ID_PMU_CRC);
}

/**
 * @brief Encode a pmu struct
 *
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param msg The MAVLink message to compress the data into
 * @param pmu C-struct to read the message contents from
 */
static inline uint16_t mavlink_msg_pmu_encode(uint8_t system_id, uint8_t component_id, mavlink_message_t* msg, const mavlink_pmu_t* pmu)
{
    return mavlink_msg_pmu_pack(system_id, component_id, msg, pmu->VbattA, pmu->IbattA, pmu->VbattB, pmu->IbattB, pmu->Vbatt12S, pmu->PMU_RPM, pmu->PMU_temp, pmu->PMU_frame_ok, pmu->PMU_com, pmu->Fuel_level, pmu->Raw_fuel_level, pmu->env_temp, pmu->env_RH, pmu->PMU_data_status);
}

/**
 * @brief Encode a pmu struct on a channel
 *
 * @param system_id ID of this system
 * @param component_id ID of this component (e.g. 200 for IMU)
 * @param chan The MAVLink channel this message will be sent over
 * @param msg The MAVLink message to compress the data into
 * @param pmu C-struct to read the message contents from
 */
static inline uint16_t mavlink_msg_pmu_encode_chan(uint8_t system_id, uint8_t component_id, uint8_t chan, mavlink_message_t* msg, const mavlink_pmu_t* pmu)
{
    return mavlink_msg_pmu_pack_chan(system_id, component_id, chan, msg, pmu->VbattA, pmu->IbattA, pmu->VbattB, pmu->IbattB, pmu->Vbatt12S, pmu->PMU_RPM, pmu->PMU_temp, pmu->PMU_frame_ok, pmu->PMU_com, pmu->Fuel_level, pmu->Raw_fuel_level, pmu->env_temp, pmu->env_RH, pmu->PMU_data_status);
}

/**
 * @brief Send a pmu message
 * @param chan MAVLink channel to send the message
 *
 * @param VbattA  Voltage battery A.
 * @param IbattA  Current battery A.
 * @param VbattB  Voltage battery B.
 * @param IbattB  Current battery B.
 * @param Vbatt12S  Voltage battery 12S.
 * @param PMU_RPM  Main engine speed.
 * @param PMU_temp  PMU temperature.
 * @param PMU_frame_ok  PMU frame status.
 * @param PMU_com  PMU communication.
 * @param Fuel_level  Fuel level.
 * @param Raw_fuel_level  Raw fuel level.
 * @param env_temp  Environment temperature.
 * @param env_RH  Environment relative humidity.
 * @param PMU_data_status  PMU data status.
 */
#ifdef MAVLINK_USE_CONVENIENCE_FUNCTIONS

static inline void mavlink_msg_pmu_send(mavlink_channel_t chan, float VbattA, float IbattA, float VbattB, float IbattB, float Vbatt12S, uint16_t PMU_RPM, uint8_t PMU_temp, int8_t PMU_frame_ok, int8_t PMU_com, float Fuel_level, float Raw_fuel_level, float env_temp, float env_RH, uint16_t PMU_data_status)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char buf[MAVLINK_MSG_ID_PMU_LEN];
    _mav_put_float(buf, 0, VbattA);
    _mav_put_float(buf, 4, IbattA);
    _mav_put_float(buf, 8, VbattB);
    _mav_put_float(buf, 12, IbattB);
    _mav_put_float(buf, 16, Vbatt12S);
    _mav_put_float(buf, 20, Fuel_level);
    _mav_put_float(buf, 24, Raw_fuel_level);
    _mav_put_float(buf, 28, env_temp);
    _mav_put_float(buf, 32, env_RH);
    _mav_put_uint16_t(buf, 36, PMU_RPM);
    _mav_put_uint16_t(buf, 38, PMU_data_status);
    _mav_put_uint8_t(buf, 40, PMU_temp);
    _mav_put_int8_t(buf, 41, PMU_frame_ok);
    _mav_put_int8_t(buf, 42, PMU_com);

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_PMU, buf, MAVLINK_MSG_ID_PMU_MIN_LEN, MAVLINK_MSG_ID_PMU_LEN, MAVLINK_MSG_ID_PMU_CRC);
#else
    mavlink_pmu_t packet;
    packet.VbattA = VbattA;
    packet.IbattA = IbattA;
    packet.VbattB = VbattB;
    packet.IbattB = IbattB;
    packet.Vbatt12S = Vbatt12S;
    packet.Fuel_level = Fuel_level;
    packet.Raw_fuel_level = Raw_fuel_level;
    packet.env_temp = env_temp;
    packet.env_RH = env_RH;
    packet.PMU_RPM = PMU_RPM;
    packet.PMU_data_status = PMU_data_status;
    packet.PMU_temp = PMU_temp;
    packet.PMU_frame_ok = PMU_frame_ok;
    packet.PMU_com = PMU_com;

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_PMU, (const char *)&packet, MAVLINK_MSG_ID_PMU_MIN_LEN, MAVLINK_MSG_ID_PMU_LEN, MAVLINK_MSG_ID_PMU_CRC);
#endif
}

/**
 * @brief Send a pmu message
 * @param chan MAVLink channel to send the message
 * @param struct The MAVLink struct to serialize
 */
static inline void mavlink_msg_pmu_send_struct(mavlink_channel_t chan, const mavlink_pmu_t* pmu)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    mavlink_msg_pmu_send(chan, pmu->VbattA, pmu->IbattA, pmu->VbattB, pmu->IbattB, pmu->Vbatt12S, pmu->PMU_RPM, pmu->PMU_temp, pmu->PMU_frame_ok, pmu->PMU_com, pmu->Fuel_level, pmu->Raw_fuel_level, pmu->env_temp, pmu->env_RH, pmu->PMU_data_status);
#else
    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_PMU, (const char *)pmu, MAVLINK_MSG_ID_PMU_MIN_LEN, MAVLINK_MSG_ID_PMU_LEN, MAVLINK_MSG_ID_PMU_CRC);
#endif
}

#if MAVLINK_MSG_ID_PMU_LEN <= MAVLINK_MAX_PAYLOAD_LEN
/*
  This varient of _send() can be used to save stack space by re-using
  memory from the receive buffer.  The caller provides a
  mavlink_message_t which is the size of a full mavlink message. This
  is usually the receive buffer for the channel, and allows a reply to an
  incoming message with minimum stack space usage.
 */
static inline void mavlink_msg_pmu_send_buf(mavlink_message_t *msgbuf, mavlink_channel_t chan,  float VbattA, float IbattA, float VbattB, float IbattB, float Vbatt12S, uint16_t PMU_RPM, uint8_t PMU_temp, int8_t PMU_frame_ok, int8_t PMU_com, float Fuel_level, float Raw_fuel_level, float env_temp, float env_RH, uint16_t PMU_data_status)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    char *buf = (char *)msgbuf;
    _mav_put_float(buf, 0, VbattA);
    _mav_put_float(buf, 4, IbattA);
    _mav_put_float(buf, 8, VbattB);
    _mav_put_float(buf, 12, IbattB);
    _mav_put_float(buf, 16, Vbatt12S);
    _mav_put_float(buf, 20, Fuel_level);
    _mav_put_float(buf, 24, Raw_fuel_level);
    _mav_put_float(buf, 28, env_temp);
    _mav_put_float(buf, 32, env_RH);
    _mav_put_uint16_t(buf, 36, PMU_RPM);
    _mav_put_uint16_t(buf, 38, PMU_data_status);
    _mav_put_uint8_t(buf, 40, PMU_temp);
    _mav_put_int8_t(buf, 41, PMU_frame_ok);
    _mav_put_int8_t(buf, 42, PMU_com);

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_PMU, buf, MAVLINK_MSG_ID_PMU_MIN_LEN, MAVLINK_MSG_ID_PMU_LEN, MAVLINK_MSG_ID_PMU_CRC);
#else
    mavlink_pmu_t *packet = (mavlink_pmu_t *)msgbuf;
    packet->VbattA = VbattA;
    packet->IbattA = IbattA;
    packet->VbattB = VbattB;
    packet->IbattB = IbattB;
    packet->Vbatt12S = Vbatt12S;
    packet->Fuel_level = Fuel_level;
    packet->Raw_fuel_level = Raw_fuel_level;
    packet->env_temp = env_temp;
    packet->env_RH = env_RH;
    packet->PMU_RPM = PMU_RPM;
    packet->PMU_data_status = PMU_data_status;
    packet->PMU_temp = PMU_temp;
    packet->PMU_frame_ok = PMU_frame_ok;
    packet->PMU_com = PMU_com;

    _mav_finalize_message_chan_send(chan, MAVLINK_MSG_ID_PMU, (const char *)packet, MAVLINK_MSG_ID_PMU_MIN_LEN, MAVLINK_MSG_ID_PMU_LEN, MAVLINK_MSG_ID_PMU_CRC);
#endif
}
#endif

#endif

// MESSAGE PMU UNPACKING


/**
 * @brief Get field VbattA from pmu message
 *
 * @return  Voltage battery A.
 */
static inline float mavlink_msg_pmu_get_VbattA(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  0);
}

/**
 * @brief Get field IbattA from pmu message
 *
 * @return  Current battery A.
 */
static inline float mavlink_msg_pmu_get_IbattA(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  4);
}

/**
 * @brief Get field VbattB from pmu message
 *
 * @return  Voltage battery B.
 */
static inline float mavlink_msg_pmu_get_VbattB(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  8);
}

/**
 * @brief Get field IbattB from pmu message
 *
 * @return  Current battery B.
 */
static inline float mavlink_msg_pmu_get_IbattB(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  12);
}

/**
 * @brief Get field Vbatt12S from pmu message
 *
 * @return  Voltage battery 12S.
 */
static inline float mavlink_msg_pmu_get_Vbatt12S(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  16);
}

/**
 * @brief Get field PMU_RPM from pmu message
 *
 * @return  Main engine speed.
 */
static inline uint16_t mavlink_msg_pmu_get_PMU_RPM(const mavlink_message_t* msg)
{
    return _MAV_RETURN_uint16_t(msg,  36);
}

/**
 * @brief Get field PMU_temp from pmu message
 *
 * @return  PMU temperature.
 */
static inline uint8_t mavlink_msg_pmu_get_PMU_temp(const mavlink_message_t* msg)
{
    return _MAV_RETURN_uint8_t(msg,  40);
}

/**
 * @brief Get field PMU_frame_ok from pmu message
 *
 * @return  PMU frame status.
 */
static inline int8_t mavlink_msg_pmu_get_PMU_frame_ok(const mavlink_message_t* msg)
{
    return _MAV_RETURN_int8_t(msg,  41);
}

/**
 * @brief Get field PMU_com from pmu message
 *
 * @return  PMU communication.
 */
static inline int8_t mavlink_msg_pmu_get_PMU_com(const mavlink_message_t* msg)
{
    return _MAV_RETURN_int8_t(msg,  42);
}

/**
 * @brief Get field Fuel_level from pmu message
 *
 * @return  Fuel level.
 */
static inline float mavlink_msg_pmu_get_Fuel_level(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  20);
}

/**
 * @brief Get field Raw_fuel_level from pmu message
 *
 * @return  Raw fuel level.
 */
static inline float mavlink_msg_pmu_get_Raw_fuel_level(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  24);
}

/**
 * @brief Get field env_temp from pmu message
 *
 * @return  Environment temperature.
 */
static inline float mavlink_msg_pmu_get_env_temp(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  28);
}

/**
 * @brief Get field env_RH from pmu message
 *
 * @return  Environment relative humidity.
 */
static inline float mavlink_msg_pmu_get_env_RH(const mavlink_message_t* msg)
{
    return _MAV_RETURN_float(msg,  32);
}

/**
 * @brief Get field PMU_data_status from pmu message
 *
 * @return  PMU data status.
 */
static inline uint16_t mavlink_msg_pmu_get_PMU_data_status(const mavlink_message_t* msg)
{
    return _MAV_RETURN_uint16_t(msg,  38);
}

/**
 * @brief Decode a pmu message into a struct
 *
 * @param msg The message to decode
 * @param pmu C-struct to decode the message contents into
 */
static inline void mavlink_msg_pmu_decode(const mavlink_message_t* msg, mavlink_pmu_t* pmu)
{
#if MAVLINK_NEED_BYTE_SWAP || !MAVLINK_ALIGNED_FIELDS
    pmu->VbattA = mavlink_msg_pmu_get_VbattA(msg);
    pmu->IbattA = mavlink_msg_pmu_get_IbattA(msg);
    pmu->VbattB = mavlink_msg_pmu_get_VbattB(msg);
    pmu->IbattB = mavlink_msg_pmu_get_IbattB(msg);
    pmu->Vbatt12S = mavlink_msg_pmu_get_Vbatt12S(msg);
    pmu->Fuel_level = mavlink_msg_pmu_get_Fuel_level(msg);
    pmu->Raw_fuel_level = mavlink_msg_pmu_get_Raw_fuel_level(msg);
    pmu->env_temp = mavlink_msg_pmu_get_env_temp(msg);
    pmu->env_RH = mavlink_msg_pmu_get_env_RH(msg);
    pmu->PMU_RPM = mavlink_msg_pmu_get_PMU_RPM(msg);
    pmu->PMU_data_status = mavlink_msg_pmu_get_PMU_data_status(msg);
    pmu->PMU_temp = mavlink_msg_pmu_get_PMU_temp(msg);
    pmu->PMU_frame_ok = mavlink_msg_pmu_get_PMU_frame_ok(msg);
    pmu->PMU_com = mavlink_msg_pmu_get_PMU_com(msg);
#else
        uint8_t len = msg->len < MAVLINK_MSG_ID_PMU_LEN? msg->len : MAVLINK_MSG_ID_PMU_LEN;
        memset(pmu, 0, MAVLINK_MSG_ID_PMU_LEN);
    memcpy(pmu, _MAV_PAYLOAD(msg), len);
#endif
}
