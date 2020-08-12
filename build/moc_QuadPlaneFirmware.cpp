/****************************************************************************
** Meta object code from reading C++ file 'QuadPlaneFirmware.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.11.3)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../src/Controller/Firmware/APM/QuadPlaneFirmware.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'QuadPlaneFirmware.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.11.3. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_QuadPlaneFirmware_t {
    QByteArrayData data[9];
    char stringdata0[99];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_QuadPlaneFirmware_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_QuadPlaneFirmware_t qt_meta_stringdata_QuadPlaneFirmware = {
    {
QT_MOC_LITERAL(0, 0, 17), // "QuadPlaneFirmware"
QT_MOC_LITERAL(1, 18, 16), // "sendJoystickData"
QT_MOC_LITERAL(2, 35, 0), // ""
QT_MOC_LITERAL(3, 36, 11), // "sendClearRC"
QT_MOC_LITERAL(4, 48, 14), // "handleJSButton"
QT_MOC_LITERAL(5, 63, 2), // "id"
QT_MOC_LITERAL(6, 66, 7), // "clicked"
QT_MOC_LITERAL(7, 74, 17), // "handleUseJoystick"
QT_MOC_LITERAL(8, 92, 6) // "enable"

    },
    "QuadPlaneFirmware\0sendJoystickData\0\0"
    "sendClearRC\0handleJSButton\0id\0clicked\0"
    "handleUseJoystick\0enable"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_QuadPlaneFirmware[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       4,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    0,   34,    2, 0x0a /* Public */,
       3,    0,   35,    2, 0x0a /* Public */,
       4,    2,   36,    2, 0x0a /* Public */,
       7,    1,   41,    2, 0x0a /* Public */,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int, QMetaType::Bool,    5,    6,
    QMetaType::Void, QMetaType::Bool,    8,

       0        // eod
};

void QuadPlaneFirmware::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        QuadPlaneFirmware *_t = static_cast<QuadPlaneFirmware *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->sendJoystickData(); break;
        case 1: _t->sendClearRC(); break;
        case 2: _t->handleJSButton((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< bool(*)>(_a[2]))); break;
        case 3: _t->handleUseJoystick((*reinterpret_cast< bool(*)>(_a[1]))); break;
        default: ;
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject QuadPlaneFirmware::staticMetaObject = {
    { &FirmwarePlugin::staticMetaObject, qt_meta_stringdata_QuadPlaneFirmware.data,
      qt_meta_data_QuadPlaneFirmware,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *QuadPlaneFirmware::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *QuadPlaneFirmware::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_QuadPlaneFirmware.stringdata0))
        return static_cast<void*>(this);
    return FirmwarePlugin::qt_metacast(_clname);
}

int QuadPlaneFirmware::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = FirmwarePlugin::qt_metacall(_c, _id, _a);
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
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
