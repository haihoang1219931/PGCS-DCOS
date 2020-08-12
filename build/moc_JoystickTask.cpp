/****************************************************************************
** Meta object code from reading C++ file 'JoystickTask.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.11.3)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../src/Joystick/JoystickLib/JoystickTask.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'JoystickTask.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.11.3. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_JoystickTask_t {
    QByteArrayData data[21];
    char stringdata0[187];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_JoystickTask_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_JoystickTask_t qt_meta_stringdata_JoystickTask = {
    {
QT_MOC_LITERAL(0, 0, 12), // "JoystickTask"
QT_MOC_LITERAL(1, 13, 10), // "btnClicked"
QT_MOC_LITERAL(2, 24, 0), // ""
QT_MOC_LITERAL(3, 25, 5), // "btnID"
QT_MOC_LITERAL(4, 31, 7), // "clicked"
QT_MOC_LITERAL(5, 39, 17), // "joystickConnected"
QT_MOC_LITERAL(6, 57, 5), // "state"
QT_MOC_LITERAL(7, 63, 12), // "joyIDChanged"
QT_MOC_LITERAL(8, 76, 16), // "axisStateChanged"
QT_MOC_LITERAL(9, 93, 6), // "axisID"
QT_MOC_LITERAL(10, 100, 5), // "value"
QT_MOC_LITERAL(11, 106, 5), // "pause"
QT_MOC_LITERAL(12, 112, 6), // "_pause"
QT_MOC_LITERAL(13, 119, 6), // "doWork"
QT_MOC_LITERAL(14, 126, 15), // "getListJoystick"
QT_MOC_LITERAL(15, 142, 15), // "getJoystickInfo"
QT_MOC_LITERAL(16, 158, 6), // "jsFile"
QT_MOC_LITERAL(17, 165, 8), // "setJoyID"
QT_MOC_LITERAL(18, 174, 1), // "a"
QT_MOC_LITERAL(19, 176, 4), // "stop"
QT_MOC_LITERAL(20, 181, 5) // "joyID"

    },
    "JoystickTask\0btnClicked\0\0btnID\0clicked\0"
    "joystickConnected\0state\0joyIDChanged\0"
    "axisStateChanged\0axisID\0value\0pause\0"
    "_pause\0doWork\0getListJoystick\0"
    "getJoystickInfo\0jsFile\0setJoyID\0a\0"
    "stop\0joyID"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_JoystickTask[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       9,   14, // methods
       2,   84, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       4,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    2,   59,    2, 0x06 /* Public */,
       5,    1,   64,    2, 0x06 /* Public */,
       7,    0,   67,    2, 0x06 /* Public */,
       8,    2,   68,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
      11,    1,   73,    2, 0x0a /* Public */,
      13,    0,   76,    2, 0x0a /* Public */,
      14,    0,   77,    2, 0x0a /* Public */,
      15,    1,   78,    2, 0x0a /* Public */,

 // methods: name, argc, parameters, tag, flags
      17,    1,   81,    2, 0x02 /* Public */,

 // signals: parameters
    QMetaType::Void, QMetaType::Int, QMetaType::Bool,    3,    4,
    QMetaType::Void, QMetaType::Bool,    6,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int, QMetaType::Float,    9,   10,

 // slots: parameters
    QMetaType::Void, QMetaType::Bool,   12,
    QMetaType::Void,
    QMetaType::QStringList,
    QMetaType::QVariant, QMetaType::QString,   16,

 // methods: parameters
    QMetaType::Void, QMetaType::QString,   18,

 // properties: name, type, flags
      19, QMetaType::Bool, 0x00095103,
      20, QMetaType::QString, 0x00495103,

 // properties: notify_signal_id
       0,
       2,

       0        // eod
};

void JoystickTask::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        JoystickTask *_t = static_cast<JoystickTask *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->btnClicked((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< bool(*)>(_a[2]))); break;
        case 1: _t->joystickConnected((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 2: _t->joyIDChanged(); break;
        case 3: _t->axisStateChanged((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< float(*)>(_a[2]))); break;
        case 4: _t->pause((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 5: _t->doWork(); break;
        case 6: { QStringList _r = _t->getListJoystick();
            if (_a[0]) *reinterpret_cast< QStringList*>(_a[0]) = std::move(_r); }  break;
        case 7: { QVariant _r = _t->getJoystickInfo((*reinterpret_cast< QString(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< QVariant*>(_a[0]) = std::move(_r); }  break;
        case 8: _t->setJoyID((*reinterpret_cast< QString(*)>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (JoystickTask::*)(int , bool );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&JoystickTask::btnClicked)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (JoystickTask::*)(bool );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&JoystickTask::joystickConnected)) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (JoystickTask::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&JoystickTask::joyIDChanged)) {
                *result = 2;
                return;
            }
        }
        {
            using _t = void (JoystickTask::*)(int , float );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&JoystickTask::axisStateChanged)) {
                *result = 3;
                return;
            }
        }
    }
#ifndef QT_NO_PROPERTIES
    else if (_c == QMetaObject::ReadProperty) {
        JoystickTask *_t = static_cast<JoystickTask *>(_o);
        Q_UNUSED(_t)
        void *_v = _a[0];
        switch (_id) {
        case 0: *reinterpret_cast< bool*>(_v) = _t->stop(); break;
        case 1: *reinterpret_cast< QString*>(_v) = _t->joyID(); break;
        default: break;
        }
    } else if (_c == QMetaObject::WriteProperty) {
        JoystickTask *_t = static_cast<JoystickTask *>(_o);
        Q_UNUSED(_t)
        void *_v = _a[0];
        switch (_id) {
        case 0: _t->setStop(*reinterpret_cast< bool*>(_v)); break;
        case 1: _t->setJoyID(*reinterpret_cast< QString*>(_v)); break;
        default: break;
        }
    } else if (_c == QMetaObject::ResetProperty) {
    }
#endif // QT_NO_PROPERTIES
}

QT_INIT_METAOBJECT const QMetaObject JoystickTask::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_JoystickTask.data,
      qt_meta_data_JoystickTask,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *JoystickTask::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *JoystickTask::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_JoystickTask.stringdata0))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int JoystickTask::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 9)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 9;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 9)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 9;
    }
#ifndef QT_NO_PROPERTIES
   else if (_c == QMetaObject::ReadProperty || _c == QMetaObject::WriteProperty
            || _c == QMetaObject::ResetProperty || _c == QMetaObject::RegisterPropertyMetaType) {
        qt_static_metacall(this, _c, _id, _a);
        _id -= 2;
    } else if (_c == QMetaObject::QueryPropertyDesignable) {
        _id -= 2;
    } else if (_c == QMetaObject::QueryPropertyScriptable) {
        _id -= 2;
    } else if (_c == QMetaObject::QueryPropertyStored) {
        _id -= 2;
    } else if (_c == QMetaObject::QueryPropertyEditable) {
        _id -= 2;
    } else if (_c == QMetaObject::QueryPropertyUser) {
        _id -= 2;
    }
#endif // QT_NO_PROPERTIES
    return _id;
}

// SIGNAL 0
void JoystickTask::btnClicked(int _t1, bool _t2)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void JoystickTask::joystickConnected(bool _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void JoystickTask::joyIDChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 2, nullptr);
}

// SIGNAL 3
void JoystickTask::axisStateChanged(int _t1, float _t2)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 3, _a);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
