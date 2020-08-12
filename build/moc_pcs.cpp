/****************************************************************************
** Meta object code from reading C++ file 'pcs.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.11.3)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../src/Setting/pcs.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'pcs.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.11.3. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_PCSConfig_t {
    QByteArrayData data[8];
    char stringdata0[55];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_PCSConfig_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_PCSConfig_t qt_meta_stringdata_PCSConfig = {
    {
QT_MOC_LITERAL(0, 0, 9), // "PCSConfig"
QT_MOC_LITERAL(1, 10, 10), // "readConfig"
QT_MOC_LITERAL(2, 21, 0), // ""
QT_MOC_LITERAL(3, 22, 4), // "file"
QT_MOC_LITERAL(4, 27, 10), // "changeData"
QT_MOC_LITERAL(5, 38, 4), // "data"
QT_MOC_LITERAL(6, 43, 5), // "value"
QT_MOC_LITERAL(7, 49, 5) // "print"

    },
    "PCSConfig\0readConfig\0\0file\0changeData\0"
    "data\0value\0print"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_PCSConfig[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       3,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // methods: name, argc, parameters, tag, flags
       1,    1,   29,    2, 0x02 /* Public */,
       4,    2,   32,    2, 0x02 /* Public */,
       7,    0,   37,    2, 0x02 /* Public */,

 // methods: parameters
    QMetaType::Int, QMetaType::QString,    3,
    QMetaType::Int, QMetaType::QString, QMetaType::QString,    5,    6,
    QMetaType::Void,

       0        // eod
};

void PCSConfig::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        PCSConfig *_t = static_cast<PCSConfig *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: { int _r = _t->readConfig((*reinterpret_cast< QString(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = std::move(_r); }  break;
        case 1: { int _r = _t->changeData((*reinterpret_cast< QString(*)>(_a[1])),(*reinterpret_cast< QString(*)>(_a[2])));
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = std::move(_r); }  break;
        case 2: _t->print(); break;
        default: ;
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject PCSConfig::staticMetaObject = {
    { &Config::staticMetaObject, qt_meta_stringdata_PCSConfig.data,
      qt_meta_data_PCSConfig,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *PCSConfig::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *PCSConfig::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_PCSConfig.stringdata0))
        return static_cast<void*>(this);
    return Config::qt_metacast(_clname);
}

int PCSConfig::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = Config::qt_metacall(_c, _id, _a);
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
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
