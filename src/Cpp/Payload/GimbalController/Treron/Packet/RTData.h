#ifndef RT_DATA_H
#define RT_DATA_H
#include "Common_type.h"
#include "Object.h"
#include "utils.h"
#include "Matrix.h"
namespace Eye
{
	class RTData :public Object
	{
	private:
		index_type m_id;
		data_type m_rot, m_drot;//R
		data_type m_scale;
		pixel_type m_tx, m_ty;//velocity of pixel, tx vs ty; diff tx, ty

	public:
		RTData()
		{
			m_id = 0;
			m_rot = 0;
			m_drot = 0;
			m_scale = 1;
			m_tx = 0;
			m_ty = 0;
		}

		inline RTData(const index_type _id, const data_type _rot, const data_type _scale, const data_type _drot, const pixel_type _tx, const pixel_type _ty)
		{
			m_id = _id;  m_rot = _rot; m_drot = _drot; m_tx = _tx; m_ty = _ty;
			m_scale = _scale;
		}

		~RTData(){};

		inline index_type getId()const
		{
			return m_id;
		}

		inline void setId(const index_type &_id)
		{
			m_id = _id;
		}

		
		inline data_type getRot()const
		{
			return m_rot;
		}

		inline void setRot(const data_type &_rot)
		{
			m_rot = _rot;
		}


		inline data_type getDrot()const
		{
			return m_drot;
		}

		inline void setDrot(const data_type &_drot)
		{
			m_drot = _drot;
		}

		inline void setScale(const data_type &_scale)
		{
			m_scale = _scale;
		}

		inline data_type getScale()const
		{
			return m_scale;
		}

		inline pixel_type getTx()const
		{
			return m_tx;
		}

		inline void setTx(const pixel_type &_tx)
		{
			m_tx = _tx;
		}

		inline pixel_type getTy()const
		{
			return m_ty;
		}

		inline void setTy(const pixel_type &_ty)
		{
			m_ty = _ty;
		}


		inline RTData& operator =(const RTData &_rt)
		{
			m_id = _rt.m_id;
			m_rot = _rt.m_rot; m_drot = _rt.m_drot;
			m_tx = _rt.m_tx; m_ty = _rt.m_ty;
			m_scale = _rt.m_scale;
			return *this;
		}

		inline RTData& add(const RTData &_rt)
		{
			m_drot += _rt.m_drot;
			m_tx += _rt.m_tx; m_ty += _rt.m_ty;
			m_scale *= _rt.m_scale;
			return *this;
		}
		inline RTData& sub(const RTData &_rt)
		{
			m_drot -= _rt.m_drot;
			m_tx -= _rt.m_tx; m_ty -= _rt.m_ty;
			m_scale /= _rt.m_scale;
			return *this;
		}

		inline RTData& operator *=(const data_type _scale)
		{
			m_scale *= _scale;
			m_tx = static_cast<pixel_type>(m_tx*_scale); 
			m_ty = static_cast<pixel_type>(m_ty*_scale);
			return *this;
		}

		inline bool operator == (const RTData &_rt)
		{
			return (m_rot == _rt.m_rot&& m_drot == _rt.m_drot&&
				m_tx == _rt.m_tx&& m_ty == _rt.m_ty&&m_scale==_rt.m_scale);
		}

		inline void restart()
		{
			m_drot = 0;
			m_tx = 0;
			m_ty = 0;
			m_scale = 1;
		}

		inline math::Matrix toRT() const
		{
			data_type _cos = cos(m_drot);
			data_type _sin = sin(m_drot);
			math::Matrix _rt(3, 3);
			_rt(0, 0) = m_scale*_cos; _rt(0, 1) = m_scale*_sin; _rt(0, 2) = m_tx;
			_rt(1, 0) = m_scale*-_sin; _rt(1, 1) = m_scale*_cos; _rt(1, 2) = m_ty;
			_rt(2, 0) = 0; _rt(2, 1) = 0; _rt(2, 2) = 1;
			return _rt;
		}

		inline void toRT(math::Matrix &_rt) const
		{
			if (_rt.cols != 3 || _rt.rows != 3)
			{
				_rt.resize(3, 3);
			}

			data_type _cos = cos(m_drot); 
			data_type _sin = sin(m_drot);
			_rt(0, 0) = m_scale*_cos; _rt(0, 1) = m_scale*_sin; _rt(0, 2) = m_tx;
			_rt(1, 0) = -m_scale*_sin; _rt(1, 1) = m_scale*_cos; _rt(1, 2) = m_ty;
			_rt(2, 0) = 0; _rt(2, 1) = 0; _rt(2, 2) = 1;
		}

		inline math::Matrix toRot() const
		{
			data_type _cos = cos(m_rot);
			data_type _sin = sin(m_rot);

			math::Matrix _rot(3, 3);
			_rot(0, 0) = _cos; _rot(0, 1) = _sin; _rot(0, 2) = 0;
			_rot(1, 0) = -_sin; _rot(1, 1) = _cos; _rot(1, 2) = 0;
			_rot(2, 0) = 0; _rot(2, 1) = 0; _rot(2, 2) = 1;
			return _rot;
		}

		inline math::Matrix toRot(math::Matrix &_rot)const
		{
			if (_rot.cols != 3 || _rot.rows != 3)
			{
				_rot.resize(3, 3);
			}
			data_type _cos = cos(m_rot);
			data_type _sin = sin(m_rot);
			_rot(0, 0) = _cos; _rot(0, 1) = _sin; _rot(0, 2) = 0;
			_rot(1, 0) = -_sin; _rot(1, 1) = _cos; _rot(1, 2) = 0;
			_rot(2, 0) = 0; _rot(2, 1) = 0; _rot(2, 2) = 1;
		}

		inline RTData inv()
		{
			return RTData(m_id,m_rot, -m_drot,1/m_scale, -m_tx, -m_ty);
		}


		length_type size()
		{
			return sizeof(index_type) + sizeof(data_type)* 3 + 2 * sizeof(pixel_type);
		}

		std::vector<byte> toByte()
		{
			std::vector<byte> _result;

			std::vector<byte> b_id, b_rot, b_drot, b_scale, b_tx, b_ty;
			b_id = Utils::toByte<index_type>(m_id);
			b_rot = Utils::toByte<data_type>(m_rot);
			b_drot = Utils::toByte<data_type>(m_drot);
			b_scale = Utils::toByte<data_type>(m_scale);
			b_tx = Utils::toByte<pixel_type>(m_tx);
			b_ty = Utils::toByte<pixel_type>(m_ty);

			_result = b_id;
			_result.insert(_result.end(), b_rot.begin(), b_rot.end());
			_result.insert(_result.end(), b_drot.begin(), b_drot.end());
			_result.insert(_result.end(), b_scale.begin(), b_scale.end());
			_result.insert(_result.end(), b_tx.begin(), b_tx.end());
			_result.insert(_result.end(), b_ty.begin(), b_ty.end());

			return _result;
		}

		RTData* parse(std::vector<byte> _data, index_type _index = 0)
		{
			byte * data = _data.data();
			m_id = Utils::toValue<index_type>(data, _index);
			m_rot = Utils::toValue<data_type>(data, _index + sizeof(index_type));
			m_drot = Utils::toValue<data_type>(data, _index + sizeof(index_type)+sizeof(data_type));
			m_scale = Utils::toValue<data_type>(data, _index + sizeof(index_type)+2*sizeof(data_type));
			m_tx = Utils::toValue<pixel_type>(data, _index + sizeof(index_type)+3 * sizeof(data_type));
			m_ty = Utils::toValue<pixel_type>(data, _index + sizeof(index_type)+3 * sizeof(data_type)+sizeof(pixel_type));
			return this;
		}

		RTData* parse(byte* _data, index_type _index = 0)
		{
			m_id = Utils::toValue<index_type>(_data, _index);
			m_rot = Utils::toValue<data_type>(_data, _index + sizeof(index_type));
			m_drot = Utils::toValue<data_type>(_data, _index + sizeof(index_type)+sizeof(data_type));
			m_scale = Utils::toValue<data_type>(_data, _index + sizeof(index_type)+2 * sizeof(data_type));
			m_tx = Utils::toValue<pixel_type>(_data, _index + sizeof(index_type)+3 * sizeof(data_type));
			m_ty = Utils::toValue<pixel_type>(_data, _index + sizeof(index_type)+3 * sizeof(data_type)+sizeof(pixel_type));
			return this;
		}
	};
}

#endif