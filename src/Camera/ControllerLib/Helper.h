#ifndef HEPLER_H
#define HEPLER_H

#include <QObject>
#include <QVariantMap>
#include <QVariant>
#include "Packet/Matrix.h"
#include "Packet/Vector.h"
#include "Packet/EyeRotationMatrix.h"
#include "Controller/JoystickLib/KFJoystick.h"

class Helper : public QObject {
    Q_OBJECT
public:
    Helper(QObject *_parent = 0)
        : QObject(_parent)
    {
        math::Matrix A(4, 4);
        A(0, 0) = 1.0; A(0, 1) = 0.0; A(0, 2) = 0.1; A(0, 3) = 0.0;
        A(1, 0) = 0.0; A(1, 1) = 1.0; A(1, 2) = 0.0; A(1, 3) = 0.1;
        A(2, 0) = 0.0; A(2, 1) = 0.0; A(2, 2) = 1.0; A(2, 3) = 0.0;
        A(3, 0) = 0.0; A(3, 1) = 0.0; A(3, 2) = 0.0; A(3, 3) = 1.0;

        math::Matrix H(2, 4);
        H(0, 0) = 1.0; H(0, 1) = 0.0; H(0, 2) = 0.0; H(0, 3) = 0.0;
        H(1, 0) = 0.0; H(1, 1) = 1.0; H(1, 2) = 0.0; H(1, 3) = 0.0;

        math::Matrix Q(4, 4);
        Q(0, 0) = 1.0; Q(0, 1) = 0.0; Q(0, 2) = 0.0; Q(0, 3) = 0.0;
        Q(1, 0) = 0.0; Q(1, 1) = 1.0; Q(1, 2) = 0.0; Q(1, 3) = 0.0;
        Q(2, 0) = 0.0; Q(2, 1) = 0.0; Q(2, 2) = 0.01; Q(2, 3) = 0.0;
        Q(3, 0) = 0.0; Q(3, 1) = 0.0; Q(3, 2) = 0.0; Q(3, 3) = 0.01;

        math::Matrix R(2, 2);
        R(0, 0) = 100; R(0, 1) = -20;
        R(1, 0) = -20; R(1, 1) = 100;

        math::Matrix P(4, 4);
        P(0, 0) = 1.0; P(0, 1) = 0.0; P(0, 2) = 0.0; P(0, 3) = 0.0;
        P(1, 0) = 0.0; P(1, 1) = 1.0; P(1, 2) = 0.0; P(1, 3) = 0.0;
        P(2, 0) = 0.0; P(2, 1) = 0.0; P(2, 2) = 1.0; P(2, 3) = 0.0;
        P(3, 0) = 0.0; P(3, 1) = 0.0; P(3, 2) = 0.0; P(3, 3) = 1.0;

        math::Vector X0(4);
        X0(0) = 0; X0(1) = 0, X0(2) = 0; X0(3) = 0;
        m_kf.init(4, 2, A, H, R, Q, P, X0);
    }

    ~Helper(){

    }

    Q_INVOKABLE void init(){
        math::Matrix A(4, 4);
        A(0, 0) = 1.0; A(0, 1) = 0.0; A(0, 2) = 0.1; A(0, 3) = 0.0;
        A(1, 0) = 0.0; A(1, 1) = 1.0; A(1, 2) = 0.0; A(1, 3) = 0.1;
        A(2, 0) = 0.0; A(2, 1) = 0.0; A(2, 2) = 1.0; A(2, 3) = 0.0;
        A(3, 0) = 0.0; A(3, 1) = 0.0; A(3, 2) = 0.0; A(3, 3) = 1.0;

        math::Matrix H(2, 4);
        H(0, 0) = 1.0; H(0, 1) = 0.0; H(0, 2) = 0.0; H(0, 3) = 0.0;
        H(1, 0) = 0.0; H(1, 1) = 1.0; H(1, 2) = 0.0; H(1, 3) = 0.0;

        math::Matrix Q(4, 4);
        Q(0, 0) = 1.0; Q(0, 1) = 0.0; Q(0, 2) = 0.0; Q(0, 3) = 0.0;
        Q(1, 0) = 0.0; Q(1, 1) = 1.0; Q(1, 2) = 0.0; Q(1, 3) = 0.0;
        Q(2, 0) = 0.0; Q(2, 1) = 0.0; Q(2, 2) = 0.01; Q(2, 3) = 0.0;
        Q(3, 0) = 0.0; Q(3, 1) = 0.0; Q(3, 2) = 0.0; Q(3, 3) = 0.01;

        math::Matrix R(2, 2);
        R(0, 0) = 100; R(0, 1) = -20;
        R(1, 0) = -20; R(1, 1) = 100;

        math::Matrix P(4, 4);
        P(0, 0) = 1.0; P(0, 1) = 0.0; P(0, 2) = 0.0; P(0, 3) = 0.0;
        P(1, 0) = 0.0; P(1, 1) = 1.0; P(1, 2) = 0.0; P(1, 3) = 0.0;
        P(2, 0) = 0.0; P(2, 1) = 0.0; P(2, 2) = 1.0; P(2, 3) = 0.0;
        P(3, 0) = 0.0; P(3, 1) = 0.0; P(3, 2) = 0.0; P(3, 3) = 1.0;

        math::Vector X0(4);
        X0(0) = 0; X0(1) = 0, X0(2) = 0; X0(3) = 0;
        m_kf.init(4, 2, A, H, R, Q, P, X0);
    }

    Q_INVOKABLE QVariant computeLdOnB(float _x, float _y, float _w, float _h,
                                      float _hfov, float _az, float _el){
        QVariant res;
        float px = _x - _w / 2.0;
        float py = _y - _h / 2.0;
        float fc = _w / (2.0 * tan(_hfov/2.0));

        float F = sqrt(px * px + py * py + fc * fc);

        math::Vector ldOnC(3), ldOnB(3);
        ldOnC(0) = px / F;
        ldOnC(1) = py / F;
        ldOnC(2) = fc / F;

        math::Matrix rc2b = EyeRotationMatrix::rC2B(-_az, _el);
        ldOnB = rc2b * ldOnC;

        QVariantList list;
        list.append((float)ldOnB(0));
        list.append((float)ldOnB(1));
        list.append((float)ldOnB(2));

        res = QVariant(list);
        return res;
    }

    Q_INVOKABLE QVariant updateCenter(float _ldOnB0, float _ldOnB1, float _ldOnB2,
                                      float _hfov, float _w, float _h,
                                      float _az, float _el){
        QVariant res;
        QVariantMap map;

        float fc = _w / (2.0 * tan(_hfov/2.0));

        math::Vector ldOnC(3), ldOnB(3);
        ldOnB(0) = _ldOnB0;
        ldOnB(1) = _ldOnB1;
        ldOnB(2) = _ldOnB2;

        math::Matrix rb2c = EyeRotationMatrix::rB2C(-_az, _el);
        ldOnC = rb2c * ldOnB;

        float px = fc * ldOnC(0) / ldOnC(2) + _w / 2.0;
        float py = fc * ldOnC(1) / ldOnC(2) + _h / 2.0;

        map.insert("px", px);
        map.insert("py", py);
        map.insert("width", _w);
        map.insert("height", _h);


        res = QVariant(map);
        return res;
    }

    Q_INVOKABLE int changePos(int _x, int _y){
        m_kf.predict();
        math::Vector Z(2);
        Z(0) = _x; Z(1) = _y;

        return m_kf.update(Z);
    }

    Q_INVOKABLE QVariant getPostState(){
        QVariant res;

        math::Vector xPost = m_kf.getPostState();

        QVariantList list;
        list.append((float)xPost(0));
        list.append((float)xPost(1));
        list.append((float)xPost(2));
        list.append((float)xPost(3));

        res = QVariant(list);
        return res;
    }

    Q_INVOKABLE QVariant getPredState(){
        QVariant res;

        math::Vector xPred = m_kf.getPostState();

        QVariantList list;
        list.append((float)xPred(0));
        list.append((float)xPred(1));
        list.append((float)xPred(2));
        list.append((float)xPred(3));

        res = QVariant(list);
        return res;
    }

private:
    KFJoystick m_kf;
};
#endif // HEPLER_H
