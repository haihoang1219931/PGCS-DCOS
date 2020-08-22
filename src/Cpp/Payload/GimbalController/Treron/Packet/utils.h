#ifndef UTILS_H
#define UTILS_H
#include "Common_type.h"

using namespace Eye;
namespace Utils
{
	enum class Endianness
	{
		Little,
		Big
	};

	template<typename T>
	union b_T
	{
		T value;
		byte data[sizeof(T)];
	};

	template<typename T>
	static std::vector<byte> toByte(T input, Endianness endianess = Endianness::Little)
	{
		std::vector<byte> result;
		b_T<T> tmp;
		tmp.value = input;
		if (endianess == Endianness::Big){
			for (unsigned int i = 0; i< sizeof(input); i++){
				result.push_back(tmp.data[i]);
			}
		}
		else if (endianess == Endianness::Little){
			for (unsigned int i = 0; i< sizeof(input); i++){
				result.push_back(tmp.data[sizeof(input)-1 - i]);
			}
		}
		return result;

	}

	template<typename T>
	static void toByte(T input, unsigned char *Array, int startIndex = 0, Endianness endianess = Endianness::Little)
	{
		b_T<T> tmp;
		tmp.value = input;
		if (endianess == Endianness::Big){
			for (unsigned int i = 0; i< sizeof(input); i++){
				Array[startIndex + i] = tmp.data[i];
			}
		}
		else if (endianess == Endianness::Little){
			for (unsigned int i = 0; i< sizeof(input); i++){
				Array[startIndex + i] = tmp.data[sizeof(input)-1 - i];
			}
		}
	}
	template<typename T>
	static T toValue(byte *bytes, int startIndex, Endianness endianess = Endianness::Little)
	{
		b_T<T> tmp;
		//printf("Parsing byte *...\r\n");
		if (endianess == Endianness::Big){
			for (unsigned int i = 0; i< sizeof(tmp); i++){
				tmp.data[i] = bytes[startIndex + i];
				//printf("0x%02x ",tmp.data[i]);
			}

		}

		else if (endianess == Endianness::Little){
			for (unsigned int i = 0; i< sizeof(tmp); i++){
                // Deadth note
				tmp.data[sizeof(tmp)-1 - i] = bytes[startIndex + i];
			}
		}
		//printf("\r\n");
		return tmp.value;
	}


	template<typename T>
	static T toValue(std::vector<byte> bytes, int startIndex, Endianness endianess = Endianness::Little)
	{
		b_T<T> tmp;

		if (endianess == Endianness::Big){
			for (unsigned int i = 0; i< sizeof(tmp); i++){
				tmp.data[i] = bytes[startIndex + i];

			}

		}
		else if (endianess == Endianness::Little){
			for (unsigned int i = 0; i< sizeof(tmp); i++){
				tmp.data[sizeof(tmp)-1 - i] = bytes[startIndex + i];
			}
		}
		return tmp.value;
	}
};

#endif
