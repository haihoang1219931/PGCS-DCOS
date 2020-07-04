#ifndef M_DATA_H
#define M_DATA_H

#include "Common_type.h"
#include "utils.h"
#include "Object.h"
#include "Matrix.h"

#ifndef PI
    #define PI 3.14159265
#endif

namespace Eye
{
    class MData : public Object
    {
        private:
            data_type m_rot;// radian
            data_type m_scale;
            data_type m_tx, m_ty;//velocity of pixel, tx vs ty; diff tx, ty

        public:
            MData()
            {
                m_rot = 0;
                m_scale = 1;
                m_tx = 0;
                m_ty = 0;
            }

            inline MData(const data_type _scale, const data_type _rot, const data_type _tx, const data_type _ty)
            {
                m_scale = _scale;
                m_rot = _rot;
                m_tx = _tx;
                m_ty = _ty;
            }

            ~MData() {}

            inline void reset()
            {
                m_rot = 0;
                m_scale = 1;
                m_tx = 0;
                m_ty = 0;
            }

            inline data_type getRot()const
            {
                return m_rot;
            }

            inline void setRot(const data_type& _rot)
            {
                m_rot = _rot;
            }

            inline void setScale(const data_type& _scale)
            {
                m_scale = _scale;
            }

            inline data_type getScale()const
            {
                return m_scale;
            }

            inline data_type getTx()const
            {
                return m_tx;
            }

            inline void setTx(const data_type& _tx)
            {
                m_tx = _tx;
            }

            inline data_type getTy()const
            {
                return m_ty;
            }

            inline void setTy(const data_type& _ty)
            {
                m_ty = _ty;
            }


            inline MData& operator =(const MData& _rt)
            {
                m_rot = _rt.m_rot;
                m_tx = _rt.m_tx;
                m_ty = _rt.m_ty;
                m_scale = _rt.m_scale;
                return *this;
            }

            inline MData& operator +=(const MData& _rt)
            {
                data_type _sin = sin(m_rot);
                data_type _cos = cos(m_rot);
                data_type _tx = m_tx;
                data_type _ty = m_ty;
                m_tx += _rt.m_tx * m_scale * _cos + _rt.m_ty * m_scale * _sin;
                m_ty += -_rt.m_tx * m_scale * _sin + _rt.m_ty * m_scale * _cos;
                m_rot += _rt.m_rot;
                m_scale *= _rt.m_scale;
                return *this;
            }

            inline MData operator +(const MData& _rt)
            {
                data_type _sin = sin(m_rot);
                data_type _cos = cos(m_rot);
                data_type _tx =  m_tx + _rt.m_tx * m_scale * _cos + _rt.m_ty * m_scale * _sin;
                data_type _ty = m_ty + -_rt.m_tx * m_scale * _sin + _rt.m_ty * m_scale * _cos;
                data_type _rot = m_rot + _rt.m_rot;
                data_type _scale = m_scale * _rt.m_scale;
                return MData(_scale, _rot, _tx, _ty);
            }

            static inline void addWeight(const MData& _src1, const data_type alpha, const MData& _src2, const data_type beta, MData& _dst)
            {
                data_type _s1 = _src1.getScale(), _s2 = _src2.getScale();
                data_type _tx1 = _src1.getTx(), _tx2 = _src2.getTx();
                data_type _ty1 = _src1.getTy(), _ty2 = _src2.getTy();
                data_type _sin1 = sin(_src1.getRot()), _sin2 = sin(_src2.getRot());
                data_type _cos1 = cos(_src1.getRot()), _cos2 = cos(_src2.getRot());
                data_type _a = alpha * _s1 * _cos1 + beta * _s2 * _cos2;
                data_type _b = alpha * _s1 * _sin1 + beta * _s2 * _sin2;
                data_type _s = sqrt(_a * _a + _b * _b);
                data_type _rot = atan(_b / _a);
                data_type _tx = alpha * _tx1 + beta * _tx2;
                data_type _ty = alpha * _ty1 + beta * _ty2;
                _dst.setRot(_rot);
                _dst.setScale(_s);
                _dst.setTx(_tx);
                _dst.setTy(_ty);
            }

            inline bool operator == (const MData& _rt)
            {
                return (m_rot == _rt.m_rot &&
                        m_tx == _rt.m_tx && m_ty == _rt.m_ty && m_scale == _rt.m_scale);
            }

            length_type size()
            {
                return sizeof(data_type) * 4;
            }

            std::vector<byte> toByte()
            {
                std::vector<byte> _result;
                std::vector<byte> b_rot, b_scale, b_tx, b_ty;
                b_scale = Utils::toByte<data_type>(m_scale);
                b_rot = Utils::toByte<data_type>(m_rot);
                b_tx = Utils::toByte<data_type>(m_tx);
                b_ty = Utils::toByte<data_type>(m_ty);
                _result = b_scale;
                _result.insert(_result.end(), b_rot.begin(), b_rot.end());
                _result.insert(_result.end(), b_tx.begin(), b_tx.end());
                _result.insert(_result.end(), b_ty.begin(), b_ty.end());
                return _result;
            }

            MData* parse(std::vector<byte> _data, index_type _index = 0)
            {
                byte* data = _data.data();
                m_scale = Utils::toValue<data_type>(data, _index);
                m_rot = Utils::toValue<data_type>(data, _index + sizeof(data_type));
                m_tx = Utils::toValue<data_type>(data, _index + 2 * sizeof(data_type));
                m_ty = Utils::toValue<data_type>(data, _index + 3 * sizeof(data_type));
                return this;
            }

            MData* parse(byte* _data, index_type _index = 0)
            {
                m_scale = Utils::toValue<data_type>(_data, _index);
                m_rot = Utils::toValue<data_type>(_data, _index + sizeof(data_type));
                m_tx = Utils::toValue<data_type>(_data, _index + 2 * sizeof(data_type));
                m_ty = Utils::toValue<data_type>(_data, _index + 3 * sizeof(data_type));
                return this;
            }

            MData& mul(const MData& _md)
            {
                m_tx = m_scale * cos(m_rot) * _md.m_tx + m_scale * sin(m_rot) * _md.m_ty + m_tx;
                m_ty = -1 * m_scale * sin(m_rot) * _md.m_tx + m_scale * cos(m_rot) * _md.m_ty + m_ty;
                m_scale *= _md.m_scale;
                m_rot += _md.m_rot;
                return *this;
            }

            inline math::Matrix toMatrix() const
            {
                data_type _cos = cos(m_rot);
                data_type _sin = sin(m_rot);
                math::Matrix res(3, 3);
                res(0, 0) = m_scale * _cos;
                res(0, 1) = m_scale * _sin;
                res(0, 2) = m_tx;
                res(1, 0) = -m_scale * _sin;
                res(1, 1) = m_scale * _cos;
                res(1, 2) = m_ty;
                res(2, 0) = 0;
                res(2, 1) = 0;
                res(2, 2) = 1;
                return res;
            }

            inline math::Matrix toMatrix(math::Matrix& _md) const
            {
                if (_md.cols != 3 || _md.rows != 3)
                {
                    _md.resize(3, 3);
                }

                data_type _cos = cos(m_rot);
                data_type _sin = sin(m_rot);
                _md(0, 0) = m_scale * _cos;
                _md(0, 1) = m_scale * _sin;
                _md(0, 2) = m_tx;
                _md(1, 0) = -m_scale * _sin;
                _md(1, 1) = m_scale * _cos;
                _md(1, 2) = m_ty;
                _md(2, 0) = 0;
                _md(2, 1) = 0;
                _md(2, 2) = 1;
            }

            inline MData inv()
            {
                data_type _cos = cos(m_rot);
                data_type _sin = sin(m_rot);
                return MData(1.0 / m_scale, -m_rot, (m_ty * _sin - m_tx * _cos) / m_scale, -(m_tx * _sin + m_ty * _cos) / m_scale);
            }
    };
}

#endif
#pragma once
