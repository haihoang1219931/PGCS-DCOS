/****************************************************************************
** Meta object code from reading C++ file 'PlanController.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.11.3)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../src/Controller/Mission/PlanController.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'PlanController.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.11.3. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_PlanController_t {
    QByteArrayData data[32];
    char stringdata0[471];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_PlanController_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_PlanController_t qt_meta_stringdata_PlanController = {
    {
QT_MOC_LITERAL(0, 0, 14), // "PlanController"
QT_MOC_LITERAL(1, 15, 14), // "readyToRequest"
QT_MOC_LITERAL(2, 30, 0), // ""
QT_MOC_LITERAL(3, 31, 19), // "missionItemsChanged"
QT_MOC_LITERAL(4, 51, 24), // "writeMissionItemsChanged"
QT_MOC_LITERAL(5, 76, 18), // "requestMissionDone"
QT_MOC_LITERAL(6, 95, 5), // "valid"
QT_MOC_LITERAL(7, 101, 17), // "uploadMissionDone"
QT_MOC_LITERAL(8, 119, 17), // "inProgressChanged"
QT_MOC_LITERAL(9, 137, 10), // "inProgress"
QT_MOC_LITERAL(10, 148, 11), // "progressPct"
QT_MOC_LITERAL(11, 160, 18), // "progressPercentPct"
QT_MOC_LITERAL(12, 179, 5), // "error"
QT_MOC_LITERAL(13, 185, 9), // "errorCode"
QT_MOC_LITERAL(14, 195, 8), // "errorMsg"
QT_MOC_LITERAL(15, 204, 17), // "removeAllComplete"
QT_MOC_LITERAL(16, 222, 12), // "sendComplete"
QT_MOC_LITERAL(17, 235, 23), // "_mavlinkMessageReceived"
QT_MOC_LITERAL(18, 259, 17), // "mavlink_message_t"
QT_MOC_LITERAL(19, 277, 7), // "message"
QT_MOC_LITERAL(20, 285, 19), // "_ackDownloadTimeout"
QT_MOC_LITERAL(21, 305, 17), // "_ackUploadTimeout"
QT_MOC_LITERAL(22, 323, 15), // "loadFromVehicle"
QT_MOC_LITERAL(23, 339, 13), // "sendToVehicle"
QT_MOC_LITERAL(24, 353, 16), // "readWaypointFile"
QT_MOC_LITERAL(25, 370, 4), // "file"
QT_MOC_LITERAL(26, 375, 17), // "writeWaypointFile"
QT_MOC_LITERAL(27, 393, 7), // "vehicle"
QT_MOC_LITERAL(28, 401, 8), // "Vehicle*"
QT_MOC_LITERAL(29, 410, 12), // "missionItems"
QT_MOC_LITERAL(30, 423, 29), // "QQmlListProperty<MissionItem>"
QT_MOC_LITERAL(31, 453, 17) // "writeMissionItems"

    },
    "PlanController\0readyToRequest\0\0"
    "missionItemsChanged\0writeMissionItemsChanged\0"
    "requestMissionDone\0valid\0uploadMissionDone\0"
    "inProgressChanged\0inProgress\0progressPct\0"
    "progressPercentPct\0error\0errorCode\0"
    "errorMsg\0removeAllComplete\0sendComplete\0"
    "_mavlinkMessageReceived\0mavlink_message_t\0"
    "message\0_ackDownloadTimeout\0"
    "_ackUploadTimeout\0loadFromVehicle\0"
    "sendToVehicle\0readWaypointFile\0file\0"
    "writeWaypointFile\0vehicle\0Vehicle*\0"
    "missionItems\0QQmlListProperty<MissionItem>\0"
    "writeMissionItems"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_PlanController[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
      17,   14, // methods
       3,  138, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
      10,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    0,   99,    2, 0x06 /* Public */,
       3,    0,  100,    2, 0x06 /* Public */,
       4,    0,  101,    2, 0x06 /* Public */,
       5,    1,  102,    2, 0x06 /* Public */,
       7,    1,  105,    2, 0x06 /* Public */,
       8,    1,  108,    2, 0x06 /* Public */,
      10,    1,  111,    2, 0x06 /* Public */,
      12,    2,  114,    2, 0x06 /* Public */,
      15,    1,  119,    2, 0x06 /* Public */,
      16,    1,  122,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
      17,    1,  125,    2, 0x0a /* Public */,
      20,    0,  128,    2, 0x0a /* Public */,
      21,    0,  129,    2, 0x0a /* Public */,
      22,    0,  130,    2, 0x0a /* Public */,
      23,    0,  131,    2, 0x0a /* Public */,
      24,    1,  132,    2, 0x0a /* Public */,
      26,    1,  135,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int,    6,
    QMetaType::Void, QMetaType::Int,    6,
    QMetaType::Void, QMetaType::Bool,    9,
    QMetaType::Void, QMetaType::Float,   11,
    QMetaType::Void, QMetaType::Int, QMetaType::QString,   13,   14,
    QMetaType::Void, QMetaType::Bool,   12,
    QMetaType::Void, QMetaType::Bool,   12,

 // slots: parameters
    QMetaType::Void, 0x80000000 | 18,   19,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::QString,   25,
    QMetaType::Void, QMetaType::QString,   25,

 // properties: name, type, flags
      27, 0x80000000 | 28, 0x0009510b,
      29, 0x80000000 | 30, 0x00495009,
      31, 0x80000000 | 30, 0x00495009,

 // properties: notify_signal_id
       0,
       1,
       2,

       0        // eod
};

void PlanController::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        PlanController *_t = static_cast<PlanController *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->readyToRequest(); break;
        case 1: _t->missionItemsChanged(); break;
        case 2: _t->writeMissionItemsChanged(); break;
        case 3: _t->requestMissionDone((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 4: _t->uploadMissionDone((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 5: _t->inProgressChanged((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 6: _t->progressPct((*reinterpret_cast< float(*)>(_a[1]))); break;
        case 7: _t->error((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< const QString(*)>(_a[2]))); break;
        case 8: _t->removeAllComplete((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 9: _t->sendComplete((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 10: _t->_mavlinkMessageReceived((*reinterpret_cast< mavlink_message_t(*)>(_a[1]))); break;
        case 11: _t->_ackDownloadTimeout(); break;
        case 12: _t->_ackUploadTimeout(); break;
        case 13: _t->loadFromVehicle(); break;
        case 14: _t->sendToVehicle(); break;
        case 15: _t->readWaypointFile((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 16: _t->writeWaypointFile((*reinterpret_cast< QString(*)>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        switch (_id) {
        default: *reinterpret_cast<int*>(_a[0]) = -1; break;
        case 10:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< mavlink_message_t >(); break;
            }
            break;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (PlanController::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&PlanController::readyToRequest)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (PlanController::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&PlanController::missionItemsChanged)) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (PlanController::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&PlanController::writeMissionItemsChanged)) {
                *result = 2;
                return;
            }
        }
        {
            using _t = void (PlanController::*)(int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&PlanController::requestMissionDone)) {
                *result = 3;
                return;
            }
        }
        {
            using _t = void (PlanController::*)(int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&PlanController::uploadMissionDone)) {
                *result = 4;
                return;
            }
        }
        {
            using _t = void (PlanController::*)(bool );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&PlanController::inProgressChanged)) {
                *result = 5;
                return;
            }
        }
        {
            using _t = void (PlanController::*)(float );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&PlanController::progressPct)) {
                *result = 6;
                return;
            }
        }
        {
            using _t = void (PlanController::*)(int , const QString & );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&PlanController::error)) {
                *result = 7;
                return;
            }
        }
        {
            using _t = void (PlanController::*)(bool );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&PlanController::removeAllComplete)) {
                *result = 8;
                return;
            }
        }
        {
            using _t = void (PlanController::*)(bool );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&PlanController::sendComplete)) {
                *result = 9;
                return;
            }
        }
    } else if (_c == QMetaObject::RegisterPropertyMetaType) {
        switch (_id) {
        default: *reinterpret_cast<int*>(_a[0]) = -1; break;
        case 0:
            *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< Vehicle* >(); break;
        }
    }

#ifndef QT_NO_PROPERTIES
    else if (_c == QMetaObject::ReadProperty) {
        PlanController *_t = static_cast<PlanController *>(_o);
        Q_UNUSED(_t)
        void *_v = _a[0];
        switch (_id) {
        case 0: *reinterpret_cast< Vehicle**>(_v) = _t->vehicle(); break;
        case 1: *reinterpret_cast< QQmlListProperty<MissionItem>*>(_v) = _t->missionItems(); break;
        case 2: *reinterpret_cast< QQmlListProperty<MissionItem>*>(_v) = _t->writeMissionItems(); break;
        default: break;
        }
    } else if (_c == QMetaObject::WriteProperty) {
        PlanController *_t = static_cast<PlanController *>(_o);
        Q_UNUSED(_t)
        void *_v = _a[0];
        switch (_id) {
        case 0: _t->setVehicle(*reinterpret_cast< Vehicle**>(_v)); break;
        default: break;
        }
    } else if (_c == QMetaObject::ResetProperty) {
    }
#endif // QT_NO_PROPERTIES
}

QT_INIT_METAOBJECT const QMetaObject PlanController::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_PlanController.data,
      qt_meta_data_PlanController,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *PlanController::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *PlanController::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_PlanController.stringdata0))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int PlanController::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 17)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 17;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 17)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 17;
    }
#ifndef QT_NO_PROPERTIES
   else if (_c == QMetaObject::ReadProperty || _c == QMetaObject::WriteProperty
            || _c == QMetaObject::ResetProperty || _c == QMetaObject::RegisterPropertyMetaType) {
        qt_static_metacall(this, _c, _id, _a);
        _id -= 3;
    } else if (_c == QMetaObject::QueryPropertyDesignable) {
        _id -= 3;
    } else if (_c == QMetaObject::QueryPropertyScriptable) {
        _id -= 3;
    } else if (_c == QMetaObject::QueryPropertyStored) {
        _id -= 3;
    } else if (_c == QMetaObject::QueryPropertyEditable) {
        _id -= 3;
    } else if (_c == QMetaObject::QueryPropertyUser) {
        _id -= 3;
    }
#endif // QT_NO_PROPERTIES
    return _id;
}

// SIGNAL 0
void PlanController::readyToRequest()
{
    QMetaObject::activate(this, &staticMetaObject, 0, nullptr);
}

// SIGNAL 1
void PlanController::missionItemsChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 1, nullptr);
}

// SIGNAL 2
void PlanController::writeMissionItemsChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 2, nullptr);
}

// SIGNAL 3
void PlanController::requestMissionDone(int _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 3, _a);
}

// SIGNAL 4
void PlanController::uploadMissionDone(int _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 4, _a);
}

// SIGNAL 5
void PlanController::inProgressChanged(bool _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 5, _a);
}

// SIGNAL 6
void PlanController::progressPct(float _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 6, _a);
}

// SIGNAL 7
void PlanController::error(int _t1, const QString & _t2)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 7, _a);
}

// SIGNAL 8
void PlanController::removeAllComplete(bool _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 8, _a);
}

// SIGNAL 9
void PlanController::sendComplete(bool _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 9, _a);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
