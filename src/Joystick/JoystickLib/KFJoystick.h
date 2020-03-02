#ifndef KFJOYSTICK_H
#define KFJOYSTICK_H

#include "../ControllerLib/Packet/Matrix.h"
#include "../ControllerLib/Packet/Vector.h"

class KFJoystick{
public:
    KFJoystick(){}
    ~KFJoystick(){}

    void init(int _nSize, int _mSize,
              math::Matrix _A,
              math::Matrix _H0, math::Matrix _R0,
              math::Matrix _Q0, math::Matrix _P0,
              math::Vector _X0){
        m_mSize = _mSize;
        m_nSize = _nSize;
        m_prevP = math::Matrix(m_nSize, m_nSize);
        m_postP = math::Matrix(m_nSize, m_nSize);
        m_Q = math::Matrix(m_nSize, m_nSize);
        m_R = math::Matrix(m_mSize, m_mSize);
        m_A = math::Matrix(m_nSize, m_nSize);
        m_H = math::Matrix(m_mSize, m_nSize);
        m_Z = math::Vector(m_mSize);
        m_prevX = math::Vector(m_nSize);
        m_postX = math::Vector(m_nSize);

        m_A = _A;
        m_H = _H0;
        m_R = _R0;
        m_Q = _Q0;
        m_postP = _P0;
        m_postX = _X0;
    }

    void predict(){
        m_prevX = m_A * m_postX;
        m_prevP = (m_A * m_postP *(m_A.trans())) + m_Q;
    }

    int update(math::Vector _Z){
        m_Z = _Z;
        math::Matrix m_tmp = m_H * m_prevP * (m_H.trans()) + m_R;
        if(math::determinant(m_tmp) == 0){
            return -1;
        }
        m_K = m_prevP * (m_H.trans()) * (m_tmp.inv());
        m_postX = m_prevX + m_K * (m_Z - m_H*m_prevX);
        m_postP = m_prevP - m_K * m_H * m_prevP;
    }

    math::Vector getPostState(){
        return m_postX;
    }

    math::Vector getPredState(){
        return m_prevX;
    }

private:
    int m_nSize; // state size (X)
    int m_mSize; // measurement size (Z)
    data_type m_t;
    math::Matrix m_prevP, m_postP, m_Q, m_R, m_H, m_K, m_A;
    math::Vector m_prevX, m_postX, m_Z;
};

#endif // KFJOYSTICK_H
