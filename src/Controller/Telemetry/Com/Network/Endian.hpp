#ifndef __NETWORK_ENDIAN_H_
#define __NETWORK_ENDIAN_H_


#include <endian.h>


/*
 * endianess defines (big or little)
 */
// #define __LITTLE_ENDIAN
// #define __BIG_ENDIAN


/*
 * define byte-swap macros
 */
inline uint64_t bswap64( uint64_t value )		{ return __builtin_bswap64(value); }
inline uint32_t bswap32( uint32_t value )		{ return __builtin_bswap32(value); }
inline uint16_t bswap16( uint16_t value )		{ return ((value >> 8) | (value << 8)); }


/*
 * define network swapping macros, based on endianness
 */
#if (__BYTE_ORDER == __LITTLE_ENDIAN)

inline uint64_t netswap64( uint64_t value )		{ return bswap64(value); }
inline uint32_t netswap32( uint32_t value )		{ return bswap32(value); }
inline uint16_t netswap16( uint16_t value )		{ return bswap16(value); }

#elif (__BYTE_ORDER == __BIG_ENDIAN)

inline uint64_t netswap64( uint64_t value )		{ return value; }
inline uint32_t netswap32( uint32_t value )		{ return value; }
inline uint16_t netswap16( uint16_t value )		{ return value; }

#endif
#endif
