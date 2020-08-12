/****************************************************************************
** Meta object code from reading C++ file 'ParamsController.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.11.3)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../src/Controller/Params/ParamsController.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'ParamsController.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.11.3. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_ParamsController_t {
    QByteArrayData data[20];
    char stringdata0[325];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_ParamsController_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_ParamsController_t qt_meta_stringdata_ParamsController = {
    {
QT_MOC_LITERAL(0, 0, 16), // "ParamsController"
QT_MOC_LITERAL(1, 17, 22), // "parametersReadyChanged"
QT_MOC_LITERAL(2, 40, 0), // ""
QT_MOC_LITERAL(3, 41, 15), // "parametersReady"
QT_MOC_LITERAL(4, 57, 24), // "missingParametersChanged"
QT_MOC_LITERAL(5, 82, 17), // "missingParameters"
QT_MOC_LITERAL(6, 100, 19), // "loadProgressChanged"
QT_MOC_LITERAL(7, 120, 5), // "value"
QT_MOC_LITERAL(8, 126, 12), // "paramChanged"
QT_MOC_LITERAL(9, 139, 9), // "paramName"
QT_MOC_LITERAL(10, 149, 22), // "_handleMessageReceived"
QT_MOC_LITERAL(11, 172, 17), // "mavlink_message_t"
QT_MOC_LITERAL(12, 190, 3), // "msg"
QT_MOC_LITERAL(13, 194, 20), // "_waitingParamTimeout"
QT_MOC_LITERAL(14, 215, 19), // "_updateParamTimeout"
QT_MOC_LITERAL(15, 235, 28), // "_initialRequestMissingParams"
QT_MOC_LITERAL(16, 264, 17), // "_readParameterRaw"
QT_MOC_LITERAL(17, 282, 10), // "paramIndex"
QT_MOC_LITERAL(18, 293, 18), // "_writeParameterRaw"
QT_MOC_LITERAL(19, 312, 12) // "loadProgress"

    },
    "ParamsController\0parametersReadyChanged\0"
    "\0parametersReady\0missingParametersChanged\0"
    "missingParameters\0loadProgressChanged\0"
    "value\0paramChanged\0paramName\0"
    "_handleMessageReceived\0mavlink_message_t\0"
    "msg\0_waitingParamTimeout\0_updateParamTimeout\0"
    "_initialRequestMissingParams\0"
    "_readParameterRaw\0paramIndex\0"
    "_writeParameterRaw\0loadProgress"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_ParamsController[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
      10,   14, // methods
       3,   92, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       4,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   64,    2, 0x06 /* Public */,
       4,    1,   67,    2, 0x06 /* Public */,
       6,    1,   70,    2, 0x06 /* Public */,
       8,    1,   73,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
      10,    1,   76,    2, 0x0a /* Public */,
      13,    0,   79,    2, 0x0a /* Public */,
      14,    0,   80,    2, 0x0a /* Public */,
      15,    0,   81,    2, 0x0a /* Public */,
      16,    2,   82,    2, 0x0a /* Public */,
      18,    2,   87,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void, QMetaType::Bool,    3,
    QMetaType::Void, QMetaType::Bool,    5,
    QMetaType::Void, QMetaType::Float,    7,
    QMetaType::Void, QMetaType::QString,    9,

 // slots: parameters
    QMetaType::Void, 0x80000000 | 11,   12,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::QString, QMetaType::Int,    9,   17,
    QMetaType::Void, QMetaType::QString, QMetaType::QVariant,    9,    7,

 // properties: name, type, flags
       3, QMetaType::Bool, 0x00495001,
       5, QMetaType::Bool, 0x00495001,
      19, QMetaType::Double, 0x00495001,

 // properties: notify_signal_id
       0,
       1,
       2,

       0        // eod
};

void ParamsController::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        ParamsController *_t = static_cast<ParamsController *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->parametersReadyChanged((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 1: _t->missingParametersChanged((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 2: _t->loadProgressChanged((*reinterpret_cast< float(*)>(_a[1]))); break;
        case 3: _t->paramChanged((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 4: _t->_handleMessageReceived((*reinterpret_cast< mavlink_message_t(*)>(_a[1]))); break;
        case 5: _t->_waitingParamTimeout(); break;
        case 6: _t->_updateParamTimeout(); break;
        case 7: _t->_initialRequestMissingParams(); break;
        case 8: _t->_readParameterRaw((*reinterpret_cast< const QString(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 9: _t->_writeParameterRaw((*reinterpret_cast< const QString(*)>(_a[1])),(*reinterpret_cast< const QVariant(*)>(_a[2]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        switch (_id) {
        default: *reinterpret_cast<int*>(_a[0]) = -1; break;
        case 4:
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
            using _t = void (ParamsController::*)(bool );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&ParamsController::parametersReadyChanged)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (ParamsController::*)(bool );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&ParamsController::missingParametersChanged)) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (ParamsController::*)(float );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&ParamsController::loadProgressChanged)) {
                *result = 2;
                return;
            }
        }
        {
            using _t = void (ParamsController::*)(QString );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&ParamsController::paramChanged)) {
                *result = 3;
                return;
            }
        }
    }
#ifndef QT_NO_PROPERTIES
    else if (_c == QMetaObject::ReadProperty) {
        ParamsController *_t = static_cast<ParamsController *>(_o);
        Q_UNUSED(_t)
        void *_v = _a[0];
        switch (_id) {
        case 0: *reinterpret_cast< bool*>(_v) = _t->parametersReady(); break;
        case 1: *reinterpret_cast< bool*>(_v) = _t->missingParameters(); break;
        case 2: *reinterpret_cast< double*>(_v) = _t->loadProgress(); break;
        default: break;
        }
    } else if (_c == QMetaObject::WriteProperty) {
    } else if (_c == QMetaObject::ResetProperty) {
    }
#endif // QT_NO_PROPERTIES
}

QT_INIT_METAOBJECT const QMetaObject ParamsController::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_ParamsController.data,
      qt_meta_data_ParamsController,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *ParamsController::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *ParamsController::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_ParamsController.stringdata0))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int ParamsController::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
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
void ParamsController::parametersReadyChanged(bool _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void ParamsController::missingParametersChanged(bool _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void ParamsController::loadProgressChanged(float _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}

// SIGNAL 3
void ParamsController::paramChanged(QString _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 3, _a);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
