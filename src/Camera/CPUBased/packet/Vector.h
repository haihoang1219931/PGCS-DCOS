#ifndef VECTOR_H
#define VECTOR_H

#include "Common_type.h"
#include "utils.h"

#include <iostream>
using namespace std;
using namespace Eye;
namespace math
{
	/*
	------- Class Vector declare-------------
	*/
	template<typename T>
	class Vector_
	{
	public:
		//! Default constructor. Creates an empty vector.
		inline Vector_();
		//! Creates a vector of \c n default instances of \c T.
		inline explicit Vector_(V_UINT_16 _v_size);
		//! Creates a vector of \c n copies of \c a.
		inline Vector_(V_UINT_16 _v_size, const T& _v_data);

		//! Creates a vector from an array of \c n instances of \c T.
		inline Vector_(V_UINT_16 _v_size, const T* _v_data);
		//! creates a vector from vector of data
		inline Vector_(const std::vector<T> &v);
		//! Copy constructor. Performs a deep copy.
		inline Vector_(const Vector_& v);

		V_UINT_16 size() const;

		T* data();

		~Vector_();

	public://Operation

		//! Returns the \c i'th element.
		inline T& operator()(V_UINT_16 _index);
		inline const T& operator()(V_UINT_16 _index) const;
		inline void resize(V_UINT_16 n_);
		inline void swap(const Vector_<T>& v);
		void operator = (const Vector_<T> &_v);
		inline Vector_ operator + (const Vector_<T> &_v);
		inline Vector_ operator - (const Vector_<T> &_v);
		inline Vector_ operator * (const T t);
		inline bool operator ==(const Vector_<T> &_v);
		inline bool operator != (const Vector_<T> &_v);

		inline const T dotProduct(const Vector_<T> &_v)const;
		T norm() const;
		inline V_UINT_16 length()const;
		inline Vector_ normalize();
		Vector_ crossProduct(const Vector_<T> &v) const;

		friend inline std::ostream& operator <<(std::ostream &os, Vector_<T> &v)
		{
			if (v._size == 0) return os;
			os << "[";
			for (V_UINT_16 i = 0; i < v._size; i++)
			{
				os << v(i);
				if (i != v._size - 1) os << ",";
			}

			os << "]" << endl;
			return os;
		}

		friend inline V_UINT_16 b_size(Vector_<T> &_vec)
		{
			return _vec.size()*sizeof(T);
		}

		inline V_UINT_16 b_size()
		{
			return _size*sizeof(T);
		}

	public:

                inline std::vector<Eye::byte> toByte();
                inline Vector_<T>* parse(V_UINT_16 _size,std::vector<Eye::byte> _data, index_type _index=0);
                inline Vector_<T>* parse(V_UINT_16 _size, Eye::byte*_data, index_type _index=0);
	protected:
		std::vector<T> _vimpl;
		V_UINT_16 _size;
		T* _data;
	};

	typedef Vector_<data_type> Vector;

	/*
	------- Class vector implementation-------------
	*/

	template<typename T>
	inline Vector_<T>::Vector_() :_data(0), _size(0), _vimpl(0){}

	template<typename T>
	inline Vector_<T>::Vector_(V_UINT_16 _v_size) : _vimpl(_v_size), _size(_v_size), _data(_v_size != 0 ? &_vimpl[0] : 0){}

	template<typename T>
	inline Vector_<T>::Vector_(V_UINT_16 _v_size, const T& _v_data) : _vimpl(_v_size, _v_data), _size(_v_size), _data((_v_size != 0) ? &_vimpl[0] : 0){}

	template<typename T>
	inline Vector_<T>::Vector_(V_UINT_16 _v_size, const T* _v_data) : _vimpl(_v_data, _v_data + _v_size), _size(_v_size), _data((_v_size != 0) ? &_vimpl[0] : 0){}
	template<typename T>
	inline Vector_<T>::Vector_(const std::vector<T> &v) : _vimpl(v), _size(v.size()), _data(v.size() != 0 ? &_vimpl[0] : 0){}

	template<typename T>
	inline Vector_<T>::Vector_(const Vector_& _v) : _vimpl(_v._vimpl), _size(_v._size), _data(_v._data){}

	template<typename T>
	Vector_<T>::~Vector_(){}

	template<typename T>
	V_UINT_16 Vector_<T>::size() const
	{
		return _size;
	}

	template<typename T>
	void Vector_<T>::operator = (const Vector_<T> &_v)
	{

		_data = _v._data;
		_size = _v._size;
		_vimpl = _v._vimpl;
	}

	template<typename T>
	Vector_<T> Vector_<T>:: operator+ (const Vector_<T> &_v)
	{
		assert(_v._size == _size);
		Vector_<T> result(_size);
		for (V_UINT_16 index = 0; index < _size; index++)
		{
			result._vimpl[index] = _vimpl[index] + _v._vimpl[index];
		}
		return result;
	}

	template<typename T>
	Vector_<T> Vector_<T>:: operator - (const Vector_<T> &_v)
	{
		assert(_v._size == _size);

		Vector_<T> result(_size);
		for (V_UINT_16 index = 0; index < _size; index++)
		{
			result._vimpl[index] = _vimpl[index] - _v._vimpl[index];
		}
		return result;
	}

	template<typename T>
	inline Vector_<T> Vector_<T>::operator * (const T _t)
	{
		Vector_<T> _result(_size);
		for (V_UINT_16 i = 0; i < _size; i++)
		{
			_result._vimpl[i] = _vimpl[i] *_t;
		}
		return _result;
	}

	template<typename T>
	inline bool Vector_<T>::operator ==(const Vector_<T> &_v)
	{
		if (_v.size() != size) return false;
		T dif = 0;
		for (V_INT_16 i = 0; i < size; i++)
		{
			dif = (_vimpl[i] - _v._vimpl[i])*(_vimpl[i] - _v._vimpl[i]);
		}
		if (dif == 0) return true;
		return false;
	}

	template<typename T>
	inline bool Vector_<T>::operator != (const Vector_<T> &_v)
	{
		if (_v.size() != size) return true;
		T dif = 0;
		for (V_INT_16 i = 0; i < size; i++)
		{
			dif = (_vimpl[i] - _v._vimpl[i])*(_vimpl[i] - _v._vimpl[i]);
		}
		if (dif == 0) return false;
		return true;
	}


	template<typename T>
	inline void Vector_<T>::swap(const Vector_<T>& v)
	{
		_data = v._data;
		_size = v._size;
		_vimpl = v._vimpl;
	}

	template<typename T>
	inline T& Vector_<T>::operator()(V_UINT_16 _index)
	{
		assert(_index >= 0 && _index < _size);
		return _vimpl[_index];
	}

	template<typename T>
	inline const T& Vector_<T>::operator()(V_UINT_16 _index) const
	{
		assert(_index >= 0 && _index < _size);
		return _vimpl[_index];
	}

	template<typename T>
	inline const T Vector_<T>::dotProduct(const Vector_<T> &_v) const
	{
		assert(_v._size == _size);
		T _dot = 0;
		for (V_UINT_16 i = 0; i < _size; i++)
		{
			_dot += _vimpl[i] * _v._vimpl[i];
		}
		return _dot;
	}

	template<typename T>
	inline T Vector_<T>::norm() const
	{
		if (_size == 0) return (T)0.0;
		T _norm = 0;
		for (V_UINT_16 i = 0; i < _size; i++)
		{
			//_norm += (*this)(i)*(*this)(i);
			_norm += _vimpl[i] * _vimpl[i];
		}
		return sqrt(_norm);
	}
	template<typename T>
	inline V_UINT_16 Vector_<T>::length()const
	{
		return _size;
	}

	template<typename T>
	inline Vector_<T> Vector_<T>::normalize()
	{
		T _norm = norm();
		if (_norm == 0) return Vector_<T>();

		Vector_<T> _result(_size);
		for (V_UINT_16 i = 0; i < _size; i++)
		{
			_result._vimpl[i] = _vimpl[i] / _norm;
		}

		return _result;
	}

	template<typename T>
	inline Vector_<T> Vector_<T>::crossProduct(const Vector_<T> &v) const
	{
		assert(_size == 3 && v._size == 3);
		Vector_<T> _result(_size);

		_result._vimpl[0] = _vimpl[1] * v._vimpl[2] - _vimpl[2] * v._vimpl[1];
		_result._vimpl[1] = _vimpl[2] * v._vimpl[0] - _vimpl[0] * v._vimpl[2];
		_result._vimpl[2] = _vimpl[0] * v._vimpl[1] - _vimpl[1] * v._vimpl[0];
		return _result;
	}

	template<typename T>
	void Vector_<T>::resize(V_UINT_16 n_)
	{
		if (_size == n_) {
			return;
		}
		_vimpl.resize(n_);
		_data = (n_ != 0) ? &_vimpl[0] : 0;
		_size = n_;
	}

	template<typename T>
	T* Vector_<T>::data()
	{
		//return _data;
		return &_vimpl[0];
	}

	template<typename T>
        inline std::vector<Eye::byte> Vector_<T>::toByte()
	{
                std::vector<Eye::byte> _result(0);
                std::vector<Eye::byte> b_vector;
		for (index_type i = 0; i < size(); i++)
		{
                        b_vector = Utils::toByte<T>(this->_vimpl[i]);
			_result.insert(_result.end(), b_vector.begin(), b_vector.end());
		}
		return _result;
	}
	template<typename T>
        inline Vector_<T>* Vector_<T>::parse(V_UINT_16 _size, std::vector<Eye::byte> _data, index_type _index)
	{
		Vector_<T> tmp(_size);
		this->swap(tmp);
                Eye::byte* data = _data.data();
		for (int i = 0; i < size(); i++)
		{
			this->_vimpl[i] = Utils::toValue<T>(data, _index+ i*sizeof(T));
		}

		return this;
	}

	template<typename T>
        inline Vector_<T>* Vector_<T>::parse(V_UINT_16 _size, Eye::byte* _data, index_type _index)
	{
		Vector_<T> tmp(_size);
		this->swap(tmp);
		for (int i = 0; i < size(); i++)
		{
			this->_vimpl[i] = Utils::toValue<T>(_data, _index + i*sizeof(T));
		}

		return this;
	}

}
#endif
