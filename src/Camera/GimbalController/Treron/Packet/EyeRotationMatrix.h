#ifndef EYEROTATIONMATRIX_H
#define EYEROTATIONMATRIX_H

#include "Common_type.h"
#include <math.h>
#include "Matrix.h"


using namespace math;
namespace Eye {
    class EyeRotationMatrix{

    public:
        static Matrix rB2V(data_type _roll, data_type _pitch, data_type _yaw)
        {
            Matrix m_rB2V(3,3);
            data_type sin_roll, cos_roll, sin_pitch, cos_pitch, sin_yaw, cos_yaw;
            sin_roll = sin(_roll); cos_roll = cos(_roll);
            sin_pitch = sin(_pitch); cos_pitch = cos(_pitch);
            sin_yaw = sin(_yaw); cos_yaw = cos(_yaw);
            m_rB2V(0, 0) = cos_pitch*cos_yaw; m_rB2V(0, 1) = cos_pitch*sin_yaw; m_rB2V(0, 2) = -sin_pitch;
            m_rB2V(1, 0) = sin_roll*sin_pitch*cos_yaw - cos_roll*sin_yaw; m_rB2V(1, 1) = sin_roll*sin_pitch*sin_yaw + cos_roll*cos_yaw; m_rB2V(1, 2) = sin_roll*cos_pitch;
            m_rB2V(2, 0) = cos_roll*sin_pitch*cos_yaw + sin_roll*sin_yaw; m_rB2V(2, 1) = cos_roll*sin_pitch*sin_yaw - sin_roll*cos_yaw; m_rB2V(2, 2) = cos_roll*cos_pitch;
            return m_rB2V;
        }

        static Matrix rV2B(data_type _roll, data_type _pitch, data_type _yaw)
        {
            Matrix m_rV2B(3,3);
            data_type sin_roll, cos_roll, sin_pitch, cos_pitch, sin_yaw, cos_yaw;
            sin_roll = sin(_roll); cos_roll = cos(_roll);
            sin_pitch = sin(_pitch); cos_pitch = cos(_pitch);
            sin_yaw = sin(_yaw); cos_yaw = cos(_yaw);
            m_rV2B(0, 0) = cos_pitch*cos_yaw; m_rV2B(0, 1) = sin_roll*sin_pitch*cos_yaw - cos_roll*sin_yaw; m_rV2B(0, 2) = cos_roll*sin_pitch*cos_yaw + sin_roll*sin_yaw;
            m_rV2B(1, 0) = cos_pitch*sin_yaw; m_rV2B(1, 1) = sin_roll*sin_pitch*sin_yaw + cos_roll*cos_yaw; m_rV2B(1, 2) = cos_roll*sin_pitch*sin_yaw - sin_roll*cos_yaw;
            m_rV2B(2, 0) = -sin_pitch; m_rV2B(2, 1) = sin_roll*cos_pitch; m_rV2B(2, 2) = cos_roll*cos_pitch;
            return m_rV2B;
        }

        /*tuna matrix*/
        static math::Matrix rOz(data_type _alpha){
            math::Matrix res(3,3);
            data_type _cos = cos(_alpha);
            data_type _sin = sin(_alpha);
            res(0,0) = _cos; res(0,1) = _sin; res(0,2) = 0;
            res(1,0) = -_sin; res(1,1) = _cos; res(1,2) = 0;
            res(2,0) = 0; res(2,1) = 0; res(2,2) = 1;
            return res;
        }

        static math::Matrix rOy(data_type _alpha){
            math::Matrix res(3,3);
            data_type _cos = cos(_alpha);
            data_type _sin = sin(_alpha);
            res(0,0) = _cos; res(0,1) = 0; res(0,2) = -_sin;
            res(1,0) = 0; res(1,1) = 1; res(1,2) = 0;
            res(2,0) = _sin; res(2,1) = 0; res(2,2) = _cos;
            return res;
        }

        static math::Matrix rOx(data_type _alpha){
            math::Matrix res(3,3);
            data_type _cos = cos(_alpha);
            data_type _sin = sin(_alpha);
            res(0,0) = 1; res(0,1) = 0; res(0,2) = 0;
            res(1,0) = 0; res(1,1) = _cos; res(1,2) = _sin;
            res(2,0) = 0; res(2,1) = -_sin; res(2,2) = _cos;
            return res;
        }

        static math::Matrix rG2C(){
            math::Matrix res(3,3);
            res(0,0) = 0; res(0,1) = 1; res(0,2) = 0;
            res(1,0) = 0; res(1,1) = 0; res(1,2) = 1;
            res(2,0) = 1; res(2,1) = 0; res(2,2) = 0;
            return res;
        }

        static math::Matrix rC2G(){
            return rG2C().trans();
        }

        static math::Matrix rC2B(data_type _az, data_type _el){
            math::Matrix res(3,3);
            data_type cosaz = cos(_az); data_type sinaz = sin(_az);
            data_type cosel = cos(_el); data_type sinel = sin(_el);
            res(0,0) = -sinaz; res(0,1) = cosaz*sinel; res(0,2) = cosaz*cosel;
            res(1,0) = cosaz; res(1,1) = sinaz*sinel; res(1,2) = sinaz*cosel;
            res(2,0) = 0; res(2,1) = cosel; res(2,2) = -sinel;
            return res;
        }

        static math::Matrix rB2C(data_type _az, data_type _el){
            math::Matrix res(3,3);
            data_type cosaz = cos(_az); data_type sinaz = sin(_az);
            data_type cosel = cos(_el); data_type sinel = sin(_el);
            res(0,0) = -sinaz; res(0,1) = cosaz; res(0,2) = 0;
            res(1,0) = cosaz*sinel; res(1,1) = sinaz*sinel; res(1,2) = cosel;
            res(2,0) = cosaz*cosel; res(2,1) = sinaz*cosel; res(2,2) = -sinel;
            return res;
        }

        static Matrix rI2B(data_type _roll, data_type _pitch, data_type _yaw)
        {
            Matrix m_rI2B(3,3);
            data_type sin_roll, cos_roll, sin_pitch, cos_pitch, sin_yaw, cos_yaw;
            sin_roll = sin(_roll); cos_roll = cos(_roll);
            sin_pitch = sin(_pitch); cos_pitch = cos(_pitch);
            sin_yaw = sin(_yaw); cos_yaw = cos(_yaw);
            m_rI2B(0, 0) = cos_pitch*cos_yaw; m_rI2B(0, 1) = cos_pitch*sin_yaw; m_rI2B(0, 2) = -sin_pitch;
            m_rI2B(1, 0) = sin_roll*sin_pitch*cos_yaw - cos_roll*sin_yaw; m_rI2B(1, 1) = sin_roll*sin_pitch*sin_yaw + cos_roll*cos_yaw; m_rI2B(1, 2) = sin_roll*cos_pitch;
            m_rI2B(2, 0) = cos_roll*sin_pitch*cos_yaw + sin_roll*sin_yaw; m_rI2B(2, 1) = cos_roll*sin_pitch*sin_yaw - sin_roll*cos_yaw; m_rI2B(2, 2) = cos_roll*cos_pitch;
            return m_rI2B;
        }

        static Matrix rB2I(data_type _roll, data_type _pitch, data_type _yaw)
        {
            Matrix m_rB2I(3,3);
            data_type sin_roll, cos_roll, sin_pitch, cos_pitch, sin_yaw, cos_yaw;
            sin_roll = sin(_roll); cos_roll = cos(_roll);
            sin_pitch = sin(_pitch); cos_pitch = cos(_pitch);
            sin_yaw = sin(_yaw); cos_yaw = cos(_yaw);
            m_rB2I(0, 0) = cos_pitch*cos_yaw; m_rB2I(0, 1) = sin_roll*sin_pitch*cos_yaw - cos_roll*sin_yaw; m_rB2I(0, 2) = cos_roll*sin_pitch*cos_yaw + sin_roll*sin_yaw;
            m_rB2I(1, 0) = cos_pitch*sin_yaw; m_rB2I(1, 1) = sin_roll*sin_pitch*sin_yaw + cos_roll*cos_yaw; m_rB2I(1, 2) = cos_roll*sin_pitch*sin_yaw - sin_roll*cos_yaw;
            m_rB2I(2, 0) = -sin_pitch; m_rB2I(2, 1) = sin_roll*cos_pitch; m_rB2I(2, 2) = cos_roll*cos_pitch;
            return m_rB2I;
        }

        /*thapnk matrix below:*/
//        static Matrix rC2B(data_type el, data_type az)
//        {
//            Matrix m_rC2B(3,3);
//            data_type sin_el, cos_el, sin_az, cos_az;
//            sin_az = sin(az); cos_az = cos(az);
//            sin_el = sin(el); cos_el = cos(el);

//            m_rC2B(0, 0) = -sin_az; m_rC2B(0, 1) = cos_az; m_rC2B(0, 2) = 0;
//            m_rC2B(1, 0) = sin_el*cos_az; m_rC2B(1, 1) = sin_el*sin_az; m_rC2B(1, 2) = cos_el;
//            m_rC2B(2, 0) = cos_el*cos_az; m_rC2B(2, 1) = cos_el*sin_az; m_rC2B(2, 2) = -sin_el;
//            return m_rC2B;
//        }
//        static Matrix rB2C(data_type el, data_type az)
//        {
//            Matrix m_rB2C(3,3);
//            data_type sin_el, cos_el, sin_az, cos_az;
//            sin_az = sin(az); cos_az = cos(az);
//            sin_el = sin(el); cos_el = cos(el);

//            m_rB2C(0, 0) = -sin_az; m_rB2C(0, 1) = cos_az*sin_el; m_rB2C(0, 2) = cos_az*cos_el;
//            m_rB2C(1, 0) = cos_az; m_rB2C(1, 1) = sin_az*sin_el; m_rB2C(1, 2) = sin_az*cos_el;
//            m_rB2C(2, 0) = 0; m_rB2C(2, 1) = cos_el; m_rB2C(2, 2) = -sin_el;
//            return m_rB2C;
//        }

//-------------------------------------------------
        static Matrix rB2G(data_type el, data_type az)
        {
            Matrix m_rB2G(3,3);
            data_type sin_el, cos_el, sin_az, cos_az;
            sin_az = sin(az); cos_az = cos(az);
            sin_el = sin(el); cos_el = cos(el);

            m_rB2G(0, 0) = cos_az*cos_el; m_rB2G(0, 1) = -sin_az; m_rB2G(0, 2) = cos_az*sin_el;
            m_rB2G(1, 0) = sin_az*cos_el; m_rB2G(1, 1) = cos_az; m_rB2G(1, 2) = sin_az*sin_el;
            m_rB2G(2, 0) = -sin_el; m_rB2G(2, 1) = 0; m_rB2G(2, 2) = cos_el;
            return m_rB2G;
        }
        static Matrix rG2B(data_type el, data_type az)
        {
            Matrix m_rG2B(3,3);
            data_type sin_el, cos_el, sin_az, cos_az;
            sin_az = sin(az); cos_az = cos(az);
            sin_el = sin(el); cos_el = cos(el);

            m_rG2B(0, 0) = cos_el*cos_az; m_rG2B(0, 1) = cos_el*sin_az; m_rG2B(0, 2) = -sin_el;
            m_rG2B(1, 0) = -sin_az; m_rG2B(1, 1) = cos_az; m_rG2B(1, 2) = 0;
            m_rG2B(2, 0) = sin_el*cos_az; m_rG2B(2, 1) = sin_el*sin_az; m_rG2B(2, 2) = cos_el;
            return m_rG2B;
        }

    };
}

#endif // EYEROTATIONMATRIX_H
