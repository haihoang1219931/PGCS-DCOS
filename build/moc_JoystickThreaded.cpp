/****************************************************************************
** Meta object code from reading C++ file 'JoystickThreaded.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.11.3)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../src/Joystick/JoystickLib/JoystickThreaded.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'JoystickThreaded.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.11.3. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_JSAxis_t {
    QByteArrayData data[10];
    char stringdata0[88];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_JSAxis_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_JSAxis_t qt_meta_stringdata_JSAxis = {
    {
QT_MOC_LITERAL(0, 0, 6), // "JSAxis"
QT_MOC_LITERAL(1, 7, 15), // "invertedChanged"
QT_MOC_LITERAL(2, 23, 0), // ""
QT_MOC_LITERAL(3, 24, 9), // "idChanged"
QT_MOC_LITERAL(4, 34, 12), // "valueChanged"
QT_MOC_LITERAL(5, 47, 14), // "mapFuncChanged"
QT_MOC_LITERAL(6, 62, 2), // "id"
QT_MOC_LITERAL(7, 65, 8), // "inverted"
QT_MOC_LITERAL(8, 74, 5), // "value"
QT_MOC_LITERAL(9, 80, 7) // "mapFunc"

    },
    "JSAxis\0invertedChanged\0\0idChanged\0"
    "valueChanged\0mapFuncChanged\0id\0inverted\0"
    "value\0mapFunc"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_JSAxis[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       4,   14, // methods
       4,   38, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       4,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    0,   34,    2, 0x06 /* Public */,
       3,    0,   35,    2, 0x06 /* Public */,
       4,    0,   36,    2, 0x06 /* Public */,
       5,    0,   37,    2, 0x06 /* Public */,

 // signals: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,

 // properties: name, type, flags
       6, QMetaType::Int, 0x00495103,
       7, QMetaType::Bool, 0x00495103,
       8, QMetaType::Float, 0x00495103,
       9, QMetaType::QString, 0x00495103,

 // properties: notify_signal_id
       1,
       0,
       2,
       3,

       0        // eod
};

void JSAxis::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        JSAxis *_t = static_cast<JSAxis *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->invertedChanged(); break;
        case 1: _t->idChanged(); break;
        case 2: _t->valueChanged(); break;
        case 3: _t->mapFuncChanged(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (JSAxis::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&JSAxis::invertedChanged)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (JSAxis::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&JSAxis::idChanged)) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (JSAxis::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&JSAxis::valueChanged)) {
                *result = 2;
                return;
            }
        }
        {
            using _t = void (JSAxis::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&JSAxis::mapFuncChanged)) {
                *result = 3;
                return;
            }
        }
    }
#ifndef QT_NO_PROPERTIES
    else if (_c == QMetaObject::ReadProperty) {
        JSAxis *_t = static_cast<JSAxis *>(_o);
        Q_UNUSED(_t)
        void *_v = _a[0];
        switch (_id) {
        case 0: *reinterpret_cast< int*>(_v) = _t->id(); break;
        case 1: *reinterpret_cast< bool*>(_v) = _t->inverted(); break;
        case 2: *reinterpret_cast< float*>(_v) = _t->value(); break;
        case 3: *reinterpret_cast< QString*>(_v) = _t->mapFunc(); break;
        default: break;
        }
    } else if (_c == QMetaObject::WriteProperty) {
        JSAxis *_t = static_cast<JSAxis *>(_o);
        Q_UNUSED(_t)
        void *_v = _a[0];
        switch (_id) {
        case 0: _t->setId(*reinterpret_cast< int*>(_v)); break;
        case 1: _t->setInverted(*reinterpret_cast< bool*>(_v)); break;
        case 2: _t->setValue(*reinterpret_cast< float*>(_v)); break;
        case 3: _t->setMapFunc(*reinterpret_cast< QString*>(_v)); break;
        default: break;
        }
    } else if (_c == QMetaObject::ResetProperty) {
    }
#endif // QT_NO_PROPERTIES
    Q_UNUSED(_a);
}

QT_INIT_METAOBJECT const QMetaObject JSAxis::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_JSAxis.data,
      qt_meta_data_JSAxis,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *JSAxis::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *JSAxis::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_JSAxis.stringdata0))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int JSAxis::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 4)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 4;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 4)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 4;
    }
#ifndef QT_NO_PROPERTIES
   else if (_c == QMetaObject::ReadProperty || _c == QMetaObject::WriteProperty
            || _c == QMetaObject::ResetProperty || _c == QMetaObject::RegisterPropertyMetaType) {
        qt_static_metacall(this, _c, _id, _a);
        _id -= 4;
    } else if (_c == QMetaObject::QueryPropertyDesignable) {
        _id -= 4;
    } else if (_c == QMetaObject::QueryPropertyScriptable) {
        _id -= 4;
    } else if (_c == QMetaObject::QueryPropertyStored) {
        _id -= 4;
    } else if (_c == QMetaObject::QueryPropertyEditable) {
        _id -= 4;
    } else if (_c == QMetaObject::QueryPropertyUser) {
        _id -= 4;
    }
#endif // QT_NO_PROPERTIES
    return _id;
}

// SIGNAL 0
void JSAxis::invertedChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 0, nullptr);
}

// SIGNAL 1
void JSAxis::idChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 1, nullptr);
}

// SIGNAL 2
void JSAxis::valueChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 2, nullptr);
}

// SIGNAL 3
void JSAxis::mapFuncChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 3, nullptr);
}
struct qt_meta_stringdata_JSButton_t {
    QByteArrayData data[8];
    char stringdata0[69];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_JSButton_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_JSButton_t qt_meta_stringdata_JSButton = {
    {
QT_MOC_LITERAL(0, 0, 8), // "JSButton"
QT_MOC_LITERAL(1, 9, 9), // "idChanged"
QT_MOC_LITERAL(2, 19, 0), // ""
QT_MOC_LITERAL(3, 20, 14), // "pressedChanged"
QT_MOC_LITERAL(4, 35, 14), // "mapFuncChanged"
QT_MOC_LITERAL(5, 50, 2), // "id"
QT_MOC_LITERAL(6, 53, 7), // "pressed"
QT_MOC_LITERAL(7, 61, 7) // "mapFunc"

    },
    "JSButton\0idChanged\0\0pressedChanged\0"
    "mapFuncChanged\0id\0pressed\0mapFunc"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_JSButton[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       3,   14, // methods
       3,   32, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       3,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    0,   29,    2, 0x06 /* Public */,
       3,    0,   30,    2, 0x06 /* Public */,
       4,    0,   31,    2, 0x06 /* Public */,

 // signals: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,

 // properties: name, type, flags
       5, QMetaType::Int, 0x00495103,
       6, QMetaType::Bool, 0x00495103,
       7, QMetaType::QString, 0x00495103,

 // properties: notify_signal_id
       0,
       1,
       2,

       0        // eod
};

void JSButton::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        JSButton *_t = static_cast<JSButton *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->idChanged(); break;
        case 1: _t->pressedChanged(); break;
        case 2: _t->mapFuncChanged(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (JSButton::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&JSButton::idChanged)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (JSButton::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&JSButton::pressedChanged)) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (JSButton::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&JSButton::mapFuncChanged)) {
                *result = 2;
                return;
            }
        }
    }
#ifndef QT_NO_PROPERTIES
    else if (_c == QMetaObject::ReadProperty) {
        JSButton *_t = static_cast<JSButton *>(_o);
        Q_UNUSED(_t)
        void *_v = _a[0];
        switch (_id) {
        case 0: *reinterpret_cast< int*>(_v) = _t->id(); break;
        case 1: *reinterpret_cast< bool*>(_v) = _t->pressed(); break;
        case 2: *reinterpret_cast< QString*>(_v) = _t->mapFunc(); break;
        default: break;
        }
    } else if (_c == QMetaObject::WriteProperty) {
        JSButton *_t = static_cast<JSButton *>(_o);
        Q_UNUSED(_t)
        void *_v = _a[0];
        switch (_id) {
        case 0: _t->setId(*reinterpret_cast< int*>(_v)); break;
        case 1: _t->setPressed(*reinterpret_cast< bool*>(_v)); break;
        case 2: _t->setMapFunc(*reinterpret_cast< QString*>(_v)); break;
        default: break;
        }
    } else if (_c == QMetaObject::ResetProperty) {
    }
#endif // QT_NO_PROPERTIES
    Q_UNUSED(_a);
}

QT_INIT_METAOBJECT const QMetaObject JSButton::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_JSButton.data,
      qt_meta_data_JSButton,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *JSButton::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *JSButton::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_JSButton.stringdata0))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int JSButton::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 3)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 3;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 3)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 3;
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
void JSButton::idChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 0, nullptr);
}

// SIGNAL 1
void JSButton::pressedChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 1, nullptr);
}

// SIGNAL 2
void JSButton::mapFuncChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 2, nullptr);
}
struct qt_meta_stringdata_JoystickThreaded_t {
    QByteArrayData data[57];
    char stringdata0[684];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_JoystickThreaded_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_JoystickThreaded_t qt_meta_stringdata_JoystickThreaded = {
    {
QT_MOC_LITERAL(0, 0, 16), // "JoystickThreaded"
QT_MOC_LITERAL(1, 17, 18), // "useJoystickChanged"
QT_MOC_LITERAL(2, 36, 0), // ""
QT_MOC_LITERAL(3, 37, 11), // "useJoystick"
QT_MOC_LITERAL(4, 49, 10), // "picChanged"
QT_MOC_LITERAL(5, 60, 16), // "buttonAxisLoaded"
QT_MOC_LITERAL(6, 77, 17), // "joystickConnected"
QT_MOC_LITERAL(7, 95, 5), // "state"
QT_MOC_LITERAL(8, 101, 11), // "axesChanged"
QT_MOC_LITERAL(9, 113, 14), // "buttonsChanged"
QT_MOC_LITERAL(10, 128, 17), // "axesConfigChanged"
QT_MOC_LITERAL(11, 146, 20), // "buttonsConfigChanged"
QT_MOC_LITERAL(12, 167, 16), // "axisValueChanged"
QT_MOC_LITERAL(13, 184, 6), // "axisID"
QT_MOC_LITERAL(14, 191, 5), // "value"
QT_MOC_LITERAL(15, 197, 18), // "buttonStateChanged"
QT_MOC_LITERAL(16, 216, 8), // "buttonID"
QT_MOC_LITERAL(17, 225, 7), // "pressed"
QT_MOC_LITERAL(18, 233, 15), // "axisRollChanged"
QT_MOC_LITERAL(19, 249, 16), // "axisPitchChanged"
QT_MOC_LITERAL(20, 266, 14), // "axisYawChanged"
QT_MOC_LITERAL(21, 281, 19), // "axisThrottleChanged"
QT_MOC_LITERAL(22, 301, 16), // "updateButtonAxis"
QT_MOC_LITERAL(23, 318, 9), // "connected"
QT_MOC_LITERAL(24, 328, 17), // "changeButtonState"
QT_MOC_LITERAL(25, 346, 5), // "btnID"
QT_MOC_LITERAL(26, 352, 7), // "clicked"
QT_MOC_LITERAL(27, 360, 15), // "changeAxisValue"
QT_MOC_LITERAL(28, 376, 5), // "start"
QT_MOC_LITERAL(29, 382, 5), // "pause"
QT_MOC_LITERAL(30, 388, 4), // "stop"
QT_MOC_LITERAL(31, 393, 8), // "setJoyID"
QT_MOC_LITERAL(32, 402, 5), // "joyID"
QT_MOC_LITERAL(33, 408, 10), // "saveConfig"
QT_MOC_LITERAL(34, 419, 10), // "loadConfig"
QT_MOC_LITERAL(35, 430, 11), // "resetConfig"
QT_MOC_LITERAL(36, 442, 13), // "mapAxisConfig"
QT_MOC_LITERAL(37, 456, 7), // "mapFunc"
QT_MOC_LITERAL(38, 464, 6), // "invert"
QT_MOC_LITERAL(39, 471, 15), // "mapButtonConfig"
QT_MOC_LITERAL(40, 487, 9), // "setInvert"
QT_MOC_LITERAL(41, 497, 7), // "camFunc"
QT_MOC_LITERAL(42, 505, 14), // "setUseJoystick"
QT_MOC_LITERAL(43, 520, 6), // "enable"
QT_MOC_LITERAL(44, 527, 7), // "mapFile"
QT_MOC_LITERAL(45, 535, 4), // "task"
QT_MOC_LITERAL(46, 540, 13), // "JoystickTask*"
QT_MOC_LITERAL(47, 554, 4), // "axes"
QT_MOC_LITERAL(48, 559, 24), // "QQmlListProperty<JSAxis>"
QT_MOC_LITERAL(49, 584, 7), // "buttons"
QT_MOC_LITERAL(50, 592, 26), // "QQmlListProperty<JSButton>"
QT_MOC_LITERAL(51, 619, 10), // "axesConfig"
QT_MOC_LITERAL(52, 630, 13), // "buttonsConfig"
QT_MOC_LITERAL(53, 644, 8), // "axisRoll"
QT_MOC_LITERAL(54, 653, 9), // "axisPitch"
QT_MOC_LITERAL(55, 663, 7), // "axisYaw"
QT_MOC_LITERAL(56, 671, 12) // "axisThrottle"

    },
    "JoystickThreaded\0useJoystickChanged\0"
    "\0useJoystick\0picChanged\0buttonAxisLoaded\0"
    "joystickConnected\0state\0axesChanged\0"
    "buttonsChanged\0axesConfigChanged\0"
    "buttonsConfigChanged\0axisValueChanged\0"
    "axisID\0value\0buttonStateChanged\0"
    "buttonID\0pressed\0axisRollChanged\0"
    "axisPitchChanged\0axisYawChanged\0"
    "axisThrottleChanged\0updateButtonAxis\0"
    "connected\0changeButtonState\0btnID\0"
    "clicked\0changeAxisValue\0start\0pause\0"
    "stop\0setJoyID\0joyID\0saveConfig\0"
    "loadConfig\0resetConfig\0mapAxisConfig\0"
    "mapFunc\0invert\0mapButtonConfig\0setInvert\0"
    "camFunc\0setUseJoystick\0enable\0mapFile\0"
    "task\0JoystickTask*\0axes\0"
    "QQmlListProperty<JSAxis>\0buttons\0"
    "QQmlListProperty<JSButton>\0axesConfig\0"
    "buttonsConfig\0axisRoll\0axisPitch\0"
    "axisYaw\0axisThrottle"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_JoystickThreaded[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
      28,   14, // methods
      10,  224, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
      14,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,  154,    2, 0x06 /* Public */,
       4,    0,  157,    2, 0x06 /* Public */,
       5,    0,  158,    2, 0x06 /* Public */,
       6,    1,  159,    2, 0x06 /* Public */,
       8,    0,  162,    2, 0x06 /* Public */,
       9,    0,  163,    2, 0x06 /* Public */,
      10,    0,  164,    2, 0x06 /* Public */,
      11,    0,  165,    2, 0x06 /* Public */,
      12,    2,  166,    2, 0x06 /* Public */,
      15,    2,  171,    2, 0x06 /* Public */,
      18,    0,  176,    2, 0x06 /* Public */,
      19,    0,  177,    2, 0x06 /* Public */,
      20,    0,  178,    2, 0x06 /* Public */,
      21,    0,  179,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
      22,    1,  180,    2, 0x0a /* Public */,
      24,    2,  183,    2, 0x0a /* Public */,
      27,    2,  188,    2, 0x0a /* Public */,

 // methods: name, argc, parameters, tag, flags
      28,    0,  193,    2, 0x02 /* Public */,
      29,    1,  194,    2, 0x02 /* Public */,
      30,    0,  197,    2, 0x02 /* Public */,
      31,    1,  198,    2, 0x02 /* Public */,
      33,    0,  201,    2, 0x02 /* Public */,
      34,    0,  202,    2, 0x02 /* Public */,
      35,    0,  203,    2, 0x02 /* Public */,
      36,    3,  204,    2, 0x02 /* Public */,
      39,    2,  211,    2, 0x02 /* Public */,
      40,    2,  216,    2, 0x02 /* Public */,
      42,    1,  221,    2, 0x02 /* Public */,

 // signals: parameters
    QMetaType::Void, QMetaType::Bool,    3,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Bool,    7,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int, QMetaType::Float,   13,   14,
    QMetaType::Void, QMetaType::Int, QMetaType::Bool,   16,   17,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,

 // slots: parameters
    QMetaType::Void, QMetaType::Bool,   23,
    QMetaType::Void, QMetaType::Int, QMetaType::Bool,   25,   26,
    QMetaType::Void, QMetaType::Int, QMetaType::Float,   13,   14,

 // methods: parameters
    QMetaType::Void,
    QMetaType::Void, QMetaType::Bool,   29,
    QMetaType::Void,
    QMetaType::Void, QMetaType::QString,   32,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int, QMetaType::QString, QMetaType::Bool,   13,   37,   38,
    QMetaType::Void, QMetaType::Int, QMetaType::QString,   16,   37,
    QMetaType::Void, QMetaType::QString, QMetaType::Bool,   41,   38,
    QMetaType::Void, QMetaType::Bool,   43,

 // properties: name, type, flags
      44, QMetaType::QString, 0x00095103,
      45, 0x80000000 | 46, 0x00095009,
      47, 0x80000000 | 48, 0x00495009,
      49, 0x80000000 | 50, 0x00495009,
      51, 0x80000000 | 48, 0x00495009,
      52, 0x80000000 | 50, 0x00495009,
      53, QMetaType::Int, 0x00495103,
      54, QMetaType::Int, 0x00495103,
      55, QMetaType::Int, 0x00495103,
      56, QMetaType::Int, 0x00495103,

 // properties: notify_signal_id
       0,
       0,
       4,
       5,
       6,
       7,
      10,
      11,
      12,
      13,

       0        // eod
};

void JoystickThreaded::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        JoystickThreaded *_t = static_cast<JoystickThreaded *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->useJoystickChanged((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 1: _t->picChanged(); break;
        case 2: _t->buttonAxisLoaded(); break;
        case 3: _t->joystickConnected((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 4: _t->axesChanged(); break;
        case 5: _t->buttonsChanged(); break;
        case 6: _t->axesConfigChanged(); break;
        case 7: _t->buttonsConfigChanged(); break;
        case 8: _t->axisValueChanged((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< float(*)>(_a[2]))); break;
        case 9: _t->buttonStateChanged((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< bool(*)>(_a[2]))); break;
        case 10: _t->axisRollChanged(); break;
        case 11: _t->axisPitchChanged(); break;
        case 12: _t->axisYawChanged(); break;
        case 13: _t->axisThrottleChanged(); break;
        case 14: _t->updateButtonAxis((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 15: _t->changeButtonState((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< bool(*)>(_a[2]))); break;
        case 16: _t->changeAxisValue((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< float(*)>(_a[2]))); break;
        case 17: _t->start(); break;
        case 18: _t->pause((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 19: _t->stop(); break;
        case 20: _t->setJoyID((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 21: _t->saveConfig(); break;
        case 22: _t->loadConfig(); break;
        case 23: _t->resetConfig(); break;
        case 24: _t->mapAxisConfig((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< QString(*)>(_a[2])),(*reinterpret_cast< bool(*)>(_a[3]))); break;
        case 25: _t->mapButtonConfig((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< QString(*)>(_a[2]))); break;
        case 26: _t->setInvert((*reinterpret_cast< QString(*)>(_a[1])),(*reinterpret_cast< bool(*)>(_a[2]))); break;
        case 27: _t->setUseJoystick((*reinterpret_cast< bool(*)>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (JoystickThreaded::*)(bool );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&JoystickThreaded::useJoystickChanged)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (JoystickThreaded::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&JoystickThreaded::picChanged)) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (JoystickThreaded::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&JoystickThreaded::buttonAxisLoaded)) {
                *result = 2;
                return;
            }
        }
        {
            using _t = void (JoystickThreaded::*)(bool );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&JoystickThreaded::joystickConnected)) {
                *result = 3;
                return;
            }
        }
        {
            using _t = void (JoystickThreaded::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&JoystickThreaded::axesChanged)) {
                *result = 4;
                return;
            }
        }
        {
            using _t = void (JoystickThreaded::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&JoystickThreaded::buttonsChanged)) {
                *result = 5;
                return;
            }
        }
        {
            using _t = void (JoystickThreaded::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&JoystickThreaded::axesConfigChanged)) {
                *result = 6;
                return;
            }
        }
        {
            using _t = void (JoystickThreaded::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&JoystickThreaded::buttonsConfigChanged)) {
                *result = 7;
                return;
            }
        }
        {
            using _t = void (JoystickThreaded::*)(int , float );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&JoystickThreaded::axisValueChanged)) {
                *result = 8;
                return;
            }
        }
        {
            using _t = void (JoystickThreaded::*)(int , bool );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&JoystickThreaded::buttonStateChanged)) {
                *result = 9;
                return;
            }
        }
        {
            using _t = void (JoystickThreaded::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&JoystickThreaded::axisRollChanged)) {
                *result = 10;
                return;
            }
        }
        {
            using _t = void (JoystickThreaded::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&JoystickThreaded::axisPitchChanged)) {
                *result = 11;
                return;
            }
        }
        {
            using _t = void (JoystickThreaded::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&JoystickThreaded::axisYawChanged)) {
                *result = 12;
                return;
            }
        }
        {
            using _t = void (JoystickThreaded::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&JoystickThreaded::axisThrottleChanged)) {
                *result = 13;
                return;
            }
        }
    } else if (_c == QMetaObject::RegisterPropertyMetaType) {
        switch (_id) {
        default: *reinterpret_cast<int*>(_a[0]) = -1; break;
        case 1:
            *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< JoystickTask* >(); break;
        }
    }

#ifndef QT_NO_PROPERTIES
    else if (_c == QMetaObject::ReadProperty) {
        JoystickThreaded *_t = static_cast<JoystickThreaded *>(_o);
        Q_UNUSED(_t)
        void *_v = _a[0];
        switch (_id) {
        case 0: *reinterpret_cast< QString*>(_v) = _t->mapFile(); break;
        case 1: *reinterpret_cast< JoystickTask**>(_v) = _t->task(); break;
        case 2: *reinterpret_cast< QQmlListProperty<JSAxis>*>(_v) = _t->axes(); break;
        case 3: *reinterpret_cast< QQmlListProperty<JSButton>*>(_v) = _t->buttons(); break;
        case 4: *reinterpret_cast< QQmlListProperty<JSAxis>*>(_v) = _t->axesConfig(); break;
        case 5: *reinterpret_cast< QQmlListProperty<JSButton>*>(_v) = _t->buttonsConfig(); break;
        case 6: *reinterpret_cast< int*>(_v) = _t->axisRoll(); break;
        case 7: *reinterpret_cast< int*>(_v) = _t->axisPitch(); break;
        case 8: *reinterpret_cast< int*>(_v) = _t->axisYaw(); break;
        case 9: *reinterpret_cast< int*>(_v) = _t->axisThrottle(); break;
        default: break;
        }
    } else if (_c == QMetaObject::WriteProperty) {
        JoystickThreaded *_t = static_cast<JoystickThreaded *>(_o);
        Q_UNUSED(_t)
        void *_v = _a[0];
        switch (_id) {
        case 0: _t->setMapFile(*reinterpret_cast< QString*>(_v)); break;
        case 6: _t->setAxisRoll(*reinterpret_cast< int*>(_v)); break;
        case 7: _t->setAxisPitch(*reinterpret_cast< int*>(_v)); break;
        case 8: _t->setAxisYaw(*reinterpret_cast< int*>(_v)); break;
        case 9: _t->setAxisThrottle(*reinterpret_cast< int*>(_v)); break;
        default: break;
        }
    } else if (_c == QMetaObject::ResetProperty) {
    }
#endif // QT_NO_PROPERTIES
}

QT_INIT_METAOBJECT const QMetaObject JoystickThreaded::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_JoystickThreaded.data,
      qt_meta_data_JoystickThreaded,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *JoystickThreaded::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *JoystickThreaded::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_JoystickThreaded.stringdata0))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int JoystickThreaded::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 28)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 28;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 28)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 28;
    }
#ifndef QT_NO_PROPERTIES
   else if (_c == QMetaObject::ReadProperty || _c == QMetaObject::WriteProperty
            || _c == QMetaObject::ResetProperty || _c == QMetaObject::RegisterPropertyMetaType) {
        qt_static_metacall(this, _c, _id, _a);
        _id -= 10;
    } else if (_c == QMetaObject::QueryPropertyDesignable) {
        _id -= 10;
    } else if (_c == QMetaObject::QueryPropertyScriptable) {
        _id -= 10;
    } else if (_c == QMetaObject::QueryPropertyStored) {
        _id -= 10;
    } else if (_c == QMetaObject::QueryPropertyEditable) {
        _id -= 10;
    } else if (_c == QMetaObject::QueryPropertyUser) {
        _id -= 10;
    }
#endif // QT_NO_PROPERTIES
    return _id;
}

// SIGNAL 0
void JoystickThreaded::useJoystickChanged(bool _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void JoystickThreaded::picChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 1, nullptr);
}

// SIGNAL 2
void JoystickThreaded::buttonAxisLoaded()
{
    QMetaObject::activate(this, &staticMetaObject, 2, nullptr);
}

// SIGNAL 3
void JoystickThreaded::joystickConnected(bool _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 3, _a);
}

// SIGNAL 4
void JoystickThreaded::axesChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 4, nullptr);
}

// SIGNAL 5
void JoystickThreaded::buttonsChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 5, nullptr);
}

// SIGNAL 6
void JoystickThreaded::axesConfigChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 6, nullptr);
}

// SIGNAL 7
void JoystickThreaded::buttonsConfigChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 7, nullptr);
}

// SIGNAL 8
void JoystickThreaded::axisValueChanged(int _t1, float _t2)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 8, _a);
}

// SIGNAL 9
void JoystickThreaded::buttonStateChanged(int _t1, bool _t2)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 9, _a);
}

// SIGNAL 10
void JoystickThreaded::axisRollChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 10, nullptr);
}

// SIGNAL 11
void JoystickThreaded::axisPitchChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 11, nullptr);
}

// SIGNAL 12
void JoystickThreaded::axisYawChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 12, nullptr);
}

// SIGNAL 13
void JoystickThreaded::axisThrottleChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 13, nullptr);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
