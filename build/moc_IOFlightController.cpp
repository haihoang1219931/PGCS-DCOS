/****************************************************************************
** Meta object code from reading C++ file 'IOFlightController.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.11.3)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../src/Controller/Com/IOFlightController.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'IOFlightController.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.11.3. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_IOFlightController_t {
    QByteArrayData data[26];
    char stringdata0[292];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_IOFlightController_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_IOFlightController_t qt_meta_stringdata_IOFlightController = {
    {
QT_MOC_LITERAL(0, 0, 18), // "IOFlightController"
QT_MOC_LITERAL(1, 19, 13), // "receivePacket"
QT_MOC_LITERAL(2, 33, 0), // ""
QT_MOC_LITERAL(3, 34, 3), // "msg"
QT_MOC_LITERAL(4, 38, 15), // "messageReceived"
QT_MOC_LITERAL(5, 54, 17), // "mavlink_message_t"
QT_MOC_LITERAL(6, 72, 7), // "message"
QT_MOC_LITERAL(7, 80, 20), // "mavlinkMessageStatus"
QT_MOC_LITERAL(8, 101, 5), // "uasId"
QT_MOC_LITERAL(9, 107, 8), // "uint64_t"
QT_MOC_LITERAL(10, 116, 9), // "totalSent"
QT_MOC_LITERAL(11, 126, 13), // "totalReceived"
QT_MOC_LITERAL(12, 140, 9), // "totalLoss"
QT_MOC_LITERAL(13, 150, 11), // "lossPercent"
QT_MOC_LITERAL(14, 162, 10), // "loadConfig"
QT_MOC_LITERAL(15, 173, 7), // "Config*"
QT_MOC_LITERAL(16, 181, 10), // "linkConfig"
QT_MOC_LITERAL(17, 192, 11), // "connectLink"
QT_MOC_LITERAL(18, 204, 14), // "disConnectLink"
QT_MOC_LITERAL(19, 219, 5), // "pause"
QT_MOC_LITERAL(20, 225, 6), // "_pause"
QT_MOC_LITERAL(21, 232, 12), // "handlePacket"
QT_MOC_LITERAL(22, 245, 6), // "packet"
QT_MOC_LITERAL(23, 252, 11), // "isConnected"
QT_MOC_LITERAL(24, 264, 12), // "getInterface"
QT_MOC_LITERAL(25, 277, 14) // "LinkInterface*"

    },
    "IOFlightController\0receivePacket\0\0msg\0"
    "messageReceived\0mavlink_message_t\0"
    "message\0mavlinkMessageStatus\0uasId\0"
    "uint64_t\0totalSent\0totalReceived\0"
    "totalLoss\0lossPercent\0loadConfig\0"
    "Config*\0linkConfig\0connectLink\0"
    "disConnectLink\0pause\0_pause\0handlePacket\0"
    "packet\0isConnected\0getInterface\0"
    "LinkInterface*"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_IOFlightController[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
      10,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       3,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   64,    2, 0x06 /* Public */,
       4,    1,   67,    2, 0x06 /* Public */,
       7,    5,   70,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
      14,    1,   81,    2, 0x0a /* Public */,
      17,    0,   84,    2, 0x0a /* Public */,
      18,    0,   85,    2, 0x0a /* Public */,
      19,    1,   86,    2, 0x0a /* Public */,
      21,    1,   89,    2, 0x0a /* Public */,
      23,    0,   92,    2, 0x0a /* Public */,
      24,    0,   93,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void, QMetaType::QByteArray,    3,
    QMetaType::Void, 0x80000000 | 5,    6,
    QMetaType::Void, QMetaType::Int, 0x80000000 | 9, 0x80000000 | 9, 0x80000000 | 9, QMetaType::Float,    8,   10,   11,   12,   13,

 // slots: parameters
    QMetaType::Void, 0x80000000 | 15,   16,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Bool,   20,
    QMetaType::Void, QMetaType::QByteArray,   22,
    QMetaType::Bool,
    0x80000000 | 25,

       0        // eod
};

void IOFlightController::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        IOFlightController *_t = static_cast<IOFlightController *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->receivePacket((*reinterpret_cast< QByteArray(*)>(_a[1]))); break;
        case 1: _t->messageReceived((*reinterpret_cast< mavlink_message_t(*)>(_a[1]))); break;
        case 2: _t->mavlinkMessageStatus((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< uint64_t(*)>(_a[2])),(*reinterpret_cast< uint64_t(*)>(_a[3])),(*reinterpret_cast< uint64_t(*)>(_a[4])),(*reinterpret_cast< float(*)>(_a[5]))); break;
        case 3: _t->loadConfig((*reinterpret_cast< Config*(*)>(_a[1]))); break;
        case 4: _t->connectLink(); break;
        case 5: _t->disConnectLink(); break;
        case 6: _t->pause((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 7: _t->handlePacket((*reinterpret_cast< QByteArray(*)>(_a[1]))); break;
        case 8: { bool _r = _t->isConnected();
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = std::move(_r); }  break;
        case 9: { LinkInterface* _r = _t->getInterface();
            if (_a[0]) *reinterpret_cast< LinkInterface**>(_a[0]) = std::move(_r); }  break;
        default: ;
        }
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        switch (_id) {
        default: *reinterpret_cast<int*>(_a[0]) = -1; break;
        case 1:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< mavlink_message_t >(); break;
            }
            break;
        case 3:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< Config* >(); break;
            }
            break;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (IOFlightController::*)(QByteArray );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&IOFlightController::receivePacket)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (IOFlightController::*)(mavlink_message_t );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&IOFlightController::messageReceived)) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (IOFlightController::*)(int , uint64_t , uint64_t , uint64_t , float );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&IOFlightController::mavlinkMessageStatus)) {
                *result = 2;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject IOFlightController::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_IOFlightController.data,
      qt_meta_data_IOFlightController,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *IOFlightController::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *IOFlightController::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_IOFlightController.stringdata0))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int IOFlightController::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 10)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 10;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 10)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 10;
    }
    return _id;
}

// SIGNAL 0
void IOFlightController::receivePacket(QByteArray _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void IOFlightController::messageReceived(mavlink_message_t _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void IOFlightController::mavlinkMessageStatus(int _t1, uint64_t _t2, uint64_t _t3, uint64_t _t4, float _t5)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)), const_cast<void*>(reinterpret_cast<const void*>(&_t3)), const_cast<void*>(reinterpret_cast<const void*>(&_t4)), const_cast<void*>(reinterpret_cast<const void*>(&_t5)) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
