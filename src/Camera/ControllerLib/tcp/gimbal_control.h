#ifndef GIMBAL_CONTROL_H
#define GIMBAL_CONTROL_H
#include "clientStuff.h"
#ifdef __cplusplus
extern "C" {
#endif
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <string.h>
#include <math.h>

#define COMM_MAX_BUFF_SIZE  36
#define COMM_START_FRAME    0xA5

int s16_Gimbal_Control_Init(void);
int s16_Gimbal_Control (ClientStuff *client,int s16_roll_spd, int s16_tilt_spd, int s16_pan_spd);
int copter_Send_Cmd(ClientStuff *client,uint8_t* data, uint16_t dataSize);
#ifdef __cplusplus
}
#endif
#endif


