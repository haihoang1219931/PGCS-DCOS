#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <termios.h>
#include "gimbal_control.h"

#include <sstream>
#include <iostream>
#include <QDebug>
#define COMM_TCP
//#define COMM_SERIAL

#define TRACK_IMAGE_WIDTH     ((int)1280)
#define TRACK_IMAGE_HEIGHT    ((int)720)
#ifdef COMM_SERIAL
#define device_name           "/dev/ttyACM0"
#elif defined COMM_TCP
//#define GIMBAL_CONTROL_SERVER_IP  "127.0.0.1"
#define GIMBAL_CONTROL_SERVER_IP  "192.168.0.123"
#define GIMBAL_CONTROL_SERVER_PORT  1234
#else
#error "Gimbal control protocol was not defined"
#endif

#define SBUS_ROLL_CHANNEL           (10-1)
#define SBUS_TILT_CHANNEL           (9-1)
#define SBUS_PAN_CHANNEL            (11-1)
#define SBUS_TILT_SPEED_CHANNEL     (12-1)
#define SBUS_PAN_SPEED_CHANNEL      (13-1)

#define NEGATIVE_DIRECTION          (1024 - 100)
#define POSITIVE_DIRECTION          (1024 + 100)
#define ZERO_CONTROL                (1024)


/*
SYNC (0xA5 0xFF) | LENGTH (2 bytes) | MESSAGE TYPE (2 bytes) | PAYLOAD (Length - 2) | CRC (4 bytes)
*/

#define COMM_MAX_BUFF_SIZE                   512
#define COMM_START_FRAME_BYTE_1              0xA5
#define COMM_START_FRAME_BYTE_2              0xFF

#define COMM_START_FRAME_POS                 0
#define COMM_MESSAGE_LENGTH_POS              2
#define COMM_MESSAGE_TYPE_POS                4
#define COMM_PAYLOAD_POS                     6
#define COMM_CRC_POS(payload_len)            (COMM_MESSAGE_LENGTH_POS+2+payload_len)
#define COMM_META_DATA_LEN                   8

#define COMM_MINIMUM_MESSAGE_LENGTH          4

/* MESSAGE TYPE LIST */
#define COMM_MESSAGE_SYSTEM                  0x1001
#define COMM_MESSAGE_MAVLINK                 0x2001
#define COMM_MESSAGE_CONTROL_COPTER          0x3001
#define COMM_MESSAGE_CONTROL_GIMBAL          0x3002
#define COMM_MESSAGE_GENERAL_STATUS          0x7001

typedef enum
{
   ENM_COMM_FRAME_PROC_START_BYTE = 0,
   ENM_COMM_FRAME_PROC_LENGTH,
   ENM_COMM_FRAME_PROC_MESSAGE_TYPE,
   ENM_COMM_FRAME_PROC_PAYLOAD,
   ENM_COMM_FRAME_PROC_CRC,
   ENM_COMM_FRAME_PROC_INVALID = 255
} ENM_COMM_FRAME_PROC_STATE_T;

#ifdef __cplusplus
extern "C" {
#endif

#ifdef COMM_SERIAL
struct termios options;
int fd = -1;
#elif defined COMM_TCP
#endif
uint16_t Sbus_CH[16] = {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024};
static uint8_t au8_tx_buff[COMM_MAX_BUFF_SIZE];
static uint8_t u8_tx_length = 0;
static bool b_gimbal_init = false;

/*
** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
**                           DEFINE SECTION
** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*/

/* This macro executes one iteration of the CRC-32. */
#define CRC32_ITER(u32_crc, u8_data)   (((u32_crc) >> 8) ^                          \
                                        u32_crc32_table[(uint8_t)((u32_crc & 0xFF) ^\
                                       (u8_data))])


/*
** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
**                           VARIABLE SECTION
** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*/

//*****************************************************************************
//
// The CRC-32 table for the polynomial C(x) = x^32 + x^26 + x^23 + x^22 +
// x^16 + x^12 + x^11 + x^10 + x^8 + x^7 + x^5 + x^4 + x^2 + x + 1 (standard
// CRC32 as used in Ethernet, MPEG-2, PNG, etc.).
//
//*****************************************************************************
const uint32_t u32_crc32_table[] =
{
    0x00000000, 0x77073096, 0xee0e612c, 0x990951ba,
    0x076dc419, 0x706af48f, 0xe963a535, 0x9e6495a3,
    0x0edb8832, 0x79dcb8a4, 0xe0d5e91e, 0x97d2d988,
    0x09b64c2b, 0x7eb17cbd, 0xe7b82d07, 0x90bf1d91,
    0x1db71064, 0x6ab020f2, 0xf3b97148, 0x84be41de,
    0x1adad47d, 0x6ddde4eb, 0xf4d4b551, 0x83d385c7,
    0x136c9856, 0x646ba8c0, 0xfd62f97a, 0x8a65c9ec,
    0x14015c4f, 0x63066cd9, 0xfa0f3d63, 0x8d080df5,
    0x3b6e20c8, 0x4c69105e, 0xd56041e4, 0xa2677172,
    0x3c03e4d1, 0x4b04d447, 0xd20d85fd, 0xa50ab56b,
    0x35b5a8fa, 0x42b2986c, 0xdbbbc9d6, 0xacbcf940,
    0x32d86ce3, 0x45df5c75, 0xdcd60dcf, 0xabd13d59,
    0x26d930ac, 0x51de003a, 0xc8d75180, 0xbfd06116,
    0x21b4f4b5, 0x56b3c423, 0xcfba9599, 0xb8bda50f,
    0x2802b89e, 0x5f058808, 0xc60cd9b2, 0xb10be924,
    0x2f6f7c87, 0x58684c11, 0xc1611dab, 0xb6662d3d,
    0x76dc4190, 0x01db7106, 0x98d220bc, 0xefd5102a,
    0x71b18589, 0x06b6b51f, 0x9fbfe4a5, 0xe8b8d433,
    0x7807c9a2, 0x0f00f934, 0x9609a88e, 0xe10e9818,
    0x7f6a0dbb, 0x086d3d2d, 0x91646c97, 0xe6635c01,
    0x6b6b51f4, 0x1c6c6162, 0x856530d8, 0xf262004e,
    0x6c0695ed, 0x1b01a57b, 0x8208f4c1, 0xf50fc457,
    0x65b0d9c6, 0x12b7e950, 0x8bbeb8ea, 0xfcb9887c,
    0x62dd1ddf, 0x15da2d49, 0x8cd37cf3, 0xfbd44c65,
    0x4db26158, 0x3ab551ce, 0xa3bc0074, 0xd4bb30e2,
    0x4adfa541, 0x3dd895d7, 0xa4d1c46d, 0xd3d6f4fb,
    0x4369e96a, 0x346ed9fc, 0xad678846, 0xda60b8d0,
    0x44042d73, 0x33031de5, 0xaa0a4c5f, 0xdd0d7cc9,
    0x5005713c, 0x270241aa, 0xbe0b1010, 0xc90c2086,
    0x5768b525, 0x206f85b3, 0xb966d409, 0xce61e49f,
    0x5edef90e, 0x29d9c998, 0xb0d09822, 0xc7d7a8b4,
    0x59b33d17, 0x2eb40d81, 0xb7bd5c3b, 0xc0ba6cad,
    0xedb88320, 0x9abfb3b6, 0x03b6e20c, 0x74b1d29a,
    0xead54739, 0x9dd277af, 0x04db2615, 0x73dc1683,
    0xe3630b12, 0x94643b84, 0x0d6d6a3e, 0x7a6a5aa8,
    0xe40ecf0b, 0x9309ff9d, 0x0a00ae27, 0x7d079eb1,
    0xf00f9344, 0x8708a3d2, 0x1e01f268, 0x6906c2fe,
    0xf762575d, 0x806567cb, 0x196c3671, 0x6e6b06e7,
    0xfed41b76, 0x89d32be0, 0x10da7a5a, 0x67dd4acc,
    0xf9b9df6f, 0x8ebeeff9, 0x17b7be43, 0x60b08ed5,
    0xd6d6a3e8, 0xa1d1937e, 0x38d8c2c4, 0x4fdff252,
    0xd1bb67f1, 0xa6bc5767, 0x3fb506dd, 0x48b2364b,
    0xd80d2bda, 0xaf0a1b4c, 0x36034af6, 0x41047a60,
    0xdf60efc3, 0xa867df55, 0x316e8eef, 0x4669be79,
    0xcb61b38c, 0xbc66831a, 0x256fd2a0, 0x5268e236,
    0xcc0c7795, 0xbb0b4703, 0x220216b9, 0x5505262f,
    0xc5ba3bbe, 0xb2bd0b28, 0x2bb45a92, 0x5cb36a04,
    0xc2d7ffa7, 0xb5d0cf31, 0x2cd99e8b, 0x5bdeae1d,
    0x9b64c2b0, 0xec63f226, 0x756aa39c, 0x026d930a,
    0x9c0906a9, 0xeb0e363f, 0x72076785, 0x05005713,
    0x95bf4a82, 0xe2b87a14, 0x7bb12bae, 0x0cb61b38,
    0x92d28e9b, 0xe5d5be0d, 0x7cdcefb7, 0x0bdbdf21,
    0x86d3d2d4, 0xf1d4e242, 0x68ddb3f8, 0x1fda836e,
    0x81be16cd, 0xf6b9265b, 0x6fb077e1, 0x18b74777,
    0x88085ae6, 0xff0f6a70, 0x66063bca, 0x11010b5c,
    0x8f659eff, 0xf862ae69, 0x616bffd3, 0x166ccf45,
    0xa00ae278, 0xd70dd2ee, 0x4e048354, 0x3903b3c2,
    0xa7672661, 0xd06016f7, 0x4969474d, 0x3e6e77db,
    0xaed16a4a, 0xd9d65adc, 0x40df0b66, 0x37d83bf0,
    0xa9bcae53, 0xdebb9ec5, 0x47b2cf7f, 0x30b5ffe9,
    0xbdbdf21c, 0xcabac28a, 0x53b39330, 0x24b4a3a6,
    0xbad03605, 0xcdd70693, 0x54de5729, 0x23d967bf,
    0xb3667a2e, 0xc4614ab8, 0x5d681b02, 0x2a6f2b94,
    0xb40bbe37, 0xc30c8ea1, 0x5a05df1b, 0x2d02ef8d,
};


/*
** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
**                           FUNCTION SECTION
** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
*/

static int16_t s16_Comm_Send_Message(ClientStuff *client, uint16_t u16_msg_type, uint8_t *pu8_payload, uint16_t u16_len);

/*******************************************************************************
//
//! Calculates the CRC-32 of an array of bytes.
//!
//! \param ulCrc is the starting CRC-32 value.
//! \param pu8_data is a pointer to the data buffer.
//! \param u32_count is the number of bytes in the data buffer.
//!
//! This function is used to calculate the CRC-32 of the input buffer.  The
//! CRC-32 is computed in a running fashion, meaning that the entire data block
//! that is to have its CRC-32 computed does not need to be supplied all at
//! once.  If the input buffer contains the entire block of data, then \b ulCrc
//! should be set to 0xFFFFFFFF.  If, however, the entire block of data is not
//! available, then \b ulCrc should be set to 0xFFFFFFFF for the first portion
//! of the data, and then the returned value should be passed back in as \b
//! ulCrc for the next portion of the data.  Once all data has been passed
//! to the function, the final CRC-32 can be obtained by inverting the last
//! returned value.
//!
//! For example, to compute the CRC-32 of a block that has been split into
//! three pieces, use the following:
//!
//! \verbatim
//!     ulCrc = u32_CRC32(0xFFFFFFFF, pucData1, ulLen1);
//!     ulCrc = u32_CRC32(ulCrc, pucData2, ulLen2);
//!     ulCrc = u32_CRC32(ulCrc, pucData3, ulLen3);
//!     ulCrc ^= 0xFFFFFFFF;
//! \endverbatim
//!
//! Computing a CRC-32 in a running fashion is useful in cases where the data
//! is arriving via a serial link (for example) and is therefore not all
//! available at one time.
//!
//! \return The accumulated CRC-32 of the input data.
//
*******************************************************************************/
uint32_t u32_CRC32 (uint32_t u32_crc, const uint8_t *pu8_data, uint32_t u32_count)
{
    uint32_t u32_temp;
#if 0
    //
    // If the data buffer is not short-aligned, then perform a single step of
    // the CRC to make it short-aligned.
    //
    if((uint32_t)pu8_data & 1)
    {
        //
        // Perform the CRC on this input byte.
        //
        u32_crc = CRC32_ITER(u32_crc, *pu8_data);

        //
        // Skip this input byte.
        //
        pu8_data++;
        u32_count--;
    }

    //
    // If the data buffer is not word-aligned and there are at least two bytes
    // of data left, then perform two steps of the CRC to make it word-aligned.
    //
    if(((uint32_t)pu8_data & 2) && (u32_count > 1))
    {
        //
        // Read the next short.
        //
        u32_temp = *(unsigned short *)pu8_data;

        //
        // Perform the CRC on these two bytes.
        //
        u32_crc = CRC32_ITER(u32_crc, u32_temp);
        u32_crc = CRC32_ITER(u32_crc, u32_temp >> 8);

        //
        // Skip these input bytes.
        //
        pu8_data += 2;
        u32_count -= 2;
    }
#endif
    //
    // While there is at least a word remaining in the data buffer, perform
    // four steps of the CRC to consume a word.
    //
    while(u32_count > 3)
    {
        //
        // Read the next word.
        //
        u32_temp = *(uint32_t *)pu8_data;

        //
        // Perform the CRC on these four bytes.
        //
        u32_crc = CRC32_ITER(u32_crc, u32_temp);
        u32_crc = CRC32_ITER(u32_crc, u32_temp >> 8);
        u32_crc = CRC32_ITER(u32_crc, u32_temp >> 16);
        u32_crc = CRC32_ITER(u32_crc, u32_temp >> 24);

        //
        // Skip these input bytes.
        //
        pu8_data += 4;
        u32_count -= 4;
    }

    //
    // If there is a short left in the input buffer, then perform two steps of
    // the CRC.
    //
    if(u32_count > 1)
    {
        //
        // Read the short.
        //
        u32_temp = *(unsigned short *)pu8_data;

        //
        // Perform the CRC on these two bytes.
        //
        u32_crc = CRC32_ITER(u32_crc, u32_temp);
        u32_crc = CRC32_ITER(u32_crc, u32_temp >> 8);

        //
        // Skip these input bytes.
        //
        pu8_data += 2;
        u32_count -= 2;
    }

    //
    // If there is a final byte remaining in the input buffer, then perform a
    // single step of the CRC.
    //
    if(u32_count != 0)
    {
        u32_crc = CRC32_ITER(u32_crc, *pu8_data);
    }

    //
    // Return the resulting CRC-16 value.
    //
    return(u32_crc);
}

int s16_Gimbal_Control_Init(void)
{
#ifdef COMM_SERIAL
  fd = open(device_name, O_RDWR | O_NDELAY | O_NOCTTY);

  if (fd == -1)
  {
    printf("Cannot open serial device");
    return -1;
  }	
  
  fcntl(fd, F_SETFL,0);
  /* Setting port parameters */
  tcgetattr(fd, &options);

  /* control flags */
  cfsetispeed(&options,B115200);  /* 115200 Bds   */
  options.c_cflag &= ~PARENB;     /* No parity  */
  options.c_cflag &= ~CSTOPB;     /*            */
  options.c_cflag &= ~CSIZE;      /* 8bit       */
  options.c_cflag |= CS8;         /*            */
  options.c_cflag &= ~CRTSCTS;    /* No hdw ctl */

  /* local flags */
  options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG); /* raw input */

  /* input flags */
  /*
  iface->options.c_iflag &= ~(INPCK | ISTRIP); // no parity
  iface->options.c_iflag &= ~(IXON | IXOFF | IXANY); // no soft ctl
  */
  /* patch: bpflegin: set to 0 in order to avoid invalid pan/tilt return values */
  options.c_iflag = 0;

  /* output flags */
  options.c_oflag &= ~OPOST; /* raw output */

  tcsetattr(fd, TCSANOW, &options);

  printf("Serial device is set up");
#elif defined COMM_TCP
  //printf("Init gimbal control");
//  while (tcp_gimbal.setup(GIMBAL_CONTROL_SERVER_IP,GIMBAL_CONTROL_SERVER_PORT) == false)
//  {
//      sleep(2);
//  }
//  b_gimbal_init = true;
//  qDebug()<<"connect to gimbal control port connet OK";
#endif
  return 1;
}

uint16_t ModRTU_CRC(char *buf, int len)
{
	uint16_t crc = 0xFFFF;
	int pos, i;
	for (pos = 0; pos < len; pos++)
	{
		crc ^= (uint16_t)buf[pos];          // XOR byte into least sig. byte of crc

		for (i = 8; i != 0; i--)
		{    // Loop over each bit
			if ((crc & 0x0001) != 0)
			{      // If the LSB is set
				crc >>= 1;                    // Shift right and XOR 0xA001
				crc ^= 0xA001;
			}
			else                            // Else LSB is not set
				crc >>= 1;                    // Just shift right
		}
	}
  // Note, this number has low and high bytes swapped, so use it accordingly (or swap bytes)
  return crc;
}

int s16_Send_Control_Command(ClientStuff *tcp_gimbal)
{
#if 0
  int err;
  int bytes;
  int timeout = 0;
  unsigned char data;
  
  au8_tx_buff[0] = COMM_START_FRAME;
  au8_tx_buff[1] = 32;  //16 channels x 2 bytes
  
  memcpy (&au8_tx_buff[2], Sbus_CH, sizeof(Sbus_CH));
  
  *((uint16_t *)(&au8_tx_buff[34])) = 0xFFFF;//ModRTU_CRC(Sbus_CH, sizeof(Sbus_CH));
  
//  printf("\r\nCommunication Frame: ");
//  for (int i = 0; i < 36; i++)
//  {
//    printf(" %02X", au8_tx_buff[i]);
//  }
    
//  printf("\r\n");

#ifdef COMM_SERIAL  
  if (fd == -1)
    return -1;
    
  err = write(fd, au8_tx_buff, 36);
  if (err < 36)
    return -1;
    
  printf("Received: ");
  
  // wait for message
  timeout = 0;
  ioctl(fd, FIONREAD, &(bytes));
  while (bytes==0) {
	  usleep(10);
	  ioctl(fd, FIONREAD, &(bytes));
	  
	  timeout++;
	  if (timeout > 1000)
	    return 0;
  }

  ioctl(fd, FIONREAD, &(bytes));
  while (bytes > 0) {
    // get octets one by one
    read(fd, &data, 1);
    printf(" %02X", data);
    usleep(1000);
    ioctl(fd, FIONREAD, &(bytes));
  }  
  printf("\r\n");
#elif defined COMM_TCP
  //printf("sending\r\n");
  if (tcp_gimbal.Send(au8_tx_buff, 36) == false)
  {
    //printf("TCP sends error\r\n");
      return -1;
  }

  tcp_gimbal.receive();
  //usleep(5000);
  //tcp_gimbal.read();
#endif
#endif

  s16_Comm_Send_Message(tcp_gimbal, COMM_MESSAGE_CONTROL_GIMBAL, (uint8_t *)Sbus_CH, 32);
//  tcp_gimbal.receive();
  return 0;
}

int copter_Send_Cmd(ClientStuff *tcp_gimbal,uint8_t* data, uint16_t dataSize)
{
  s16_Comm_Send_Message(tcp_gimbal, COMM_MESSAGE_CONTROL_COPTER, (uint8_t *)data, dataSize);
  return 0;
}

int s16_Gimbal_Control (ClientStuff *tcp_gimbal,int roll_speed, int tilt_speed, int pan_speed)
{
    if (roll_speed < -1023)
      roll_speed = -1023;
    else if (roll_speed > 1023)
      roll_speed = 1023;

    if (tilt_speed < -1023)
      tilt_speed = -1023;
    else if (tilt_speed > 1023)
      tilt_speed = 1023;

    if (pan_speed < -1023)
      pan_speed = -1023;
    else if (pan_speed > 1023)
      pan_speed = 1023;

    /* Do not need to controll roll axis */
    Sbus_CH[SBUS_ROLL_CHANNEL] = ZERO_CONTROL;

    if (tilt_speed < 0)
    {
      Sbus_CH[SBUS_TILT_SPEED_CHANNEL] = -tilt_speed;
    }
    else if (tilt_speed > 0)
    {
      Sbus_CH[SBUS_TILT_SPEED_CHANNEL] = tilt_speed;
    }
    else
    {
      Sbus_CH[SBUS_TILT_SPEED_CHANNEL] = 1024;
    }
    Sbus_CH[SBUS_TILT_CHANNEL] = 1024 + tilt_speed;

    if (pan_speed < 0)
    {
      Sbus_CH[SBUS_PAN_SPEED_CHANNEL] = -pan_speed;
    }
    else if (pan_speed > 0)
    {
      Sbus_CH[SBUS_PAN_SPEED_CHANNEL] = pan_speed;
    }
    else
    {
      Sbus_CH[SBUS_PAN_SPEED_CHANNEL] = 1024;
    }
    Sbus_CH[SBUS_PAN_CHANNEL] = 1024 + pan_speed;

    printf("Set pan,tiltSpeed = (%d,%d)\r\n",pan_speed,tilt_speed);
    return s16_Comm_Send_Message(tcp_gimbal,COMM_MESSAGE_CONTROL_GIMBAL, (uint8_t *)Sbus_CH, 32);
}

#ifdef __cplusplus
}
#endif

static int16_t s16_Comm_Send_Message(ClientStuff *client, uint16_t u16_msg_type, uint8_t *pu8_payload, uint16_t u16_len)
{
    uint32_t u32_crc;

   memset(au8_tx_buff, 0, sizeof(au8_tx_buff));

   au8_tx_buff[COMM_START_FRAME_POS] = COMM_START_FRAME_BYTE_1;
   au8_tx_buff[COMM_START_FRAME_POS+1] = COMM_START_FRAME_BYTE_2;

   if (u16_len+2 <= COMM_MINIMUM_MESSAGE_LENGTH)
   {
       *((uint16_t*)&au8_tx_buff[COMM_MESSAGE_LENGTH_POS]) = 4;
   }
   else
   {
       *((uint16_t*)&au8_tx_buff[COMM_MESSAGE_LENGTH_POS]) = u16_len + 2;
   }

   *((uint16_t*)&au8_tx_buff[COMM_MESSAGE_TYPE_POS]) = u16_msg_type;

   memcpy(&au8_tx_buff[COMM_PAYLOAD_POS], pu8_payload, u16_len);
   u16_len += 6;

   u32_crc = 0xFFFFFFFF;
   u32_crc = u32_CRC32(u32_crc, au8_tx_buff, u16_len);

   *((uint32_t*)&au8_tx_buff[u16_len]) = u32_crc;

//    printf("Send [%d] byte Big edian\r\n",u16_len+4);
    client->Send((const char *)au8_tx_buff, u16_len + 4);
    for(int i=0; i<16; i++){
        uint8_t byte0 = au8_tx_buff[6+i*2];
        uint8_t byte1 = au8_tx_buff[6+i*2+1];
        au8_tx_buff[6+i*2] = byte1;
        au8_tx_buff[6+i*2+1] = byte0;
    }
//    printf("Send [%d] byte Little edian\r\n",u16_len+4);
    client->Send((const char *)au8_tx_buff, u16_len + 4);

    return 0;
}
