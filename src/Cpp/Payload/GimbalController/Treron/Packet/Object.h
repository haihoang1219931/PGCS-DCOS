#ifndef VOBJECT_H
#define VOBJECT_H

#include "Common_type.h"
namespace Eye
{
	class Object
	{
	public:
		Object() {}
		~Object() {}
	public:
		virtual length_type size() = 0;
		virtual std::vector<byte> toByte() = 0;
		virtual Object* parse(std::vector<byte>, index_type) = 0;
		virtual Object* parse(byte*, index_type) = 0;
	};
}
#endif