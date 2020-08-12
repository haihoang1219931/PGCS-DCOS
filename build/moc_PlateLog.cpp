/****************************************************************************
** Meta object code from reading C++ file 'PlateLog.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.11.3)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../src/Files/PlateLog.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'PlateLog.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.11.3. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_PlateLog_t {
    QByteArrayData data[9];
    char stringdata0[72];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_PlateLog_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_PlateLog_t qt_meta_stringdata_PlateLog = {
    {
QT_MOC_LITERAL(0, 0, 8), // "PlateLog"
QT_MOC_LITERAL(1, 9, 11), // "plateReaded"
QT_MOC_LITERAL(2, 21, 0), // ""
QT_MOC_LITERAL(3, 22, 7), // "logLine"
QT_MOC_LITERAL(4, 30, 13), // "appendLogFile"
QT_MOC_LITERAL(5, 44, 4), // "file"
QT_MOC_LITERAL(6, 49, 4), // "line"
QT_MOC_LITERAL(7, 54, 11), // "readLogFile"
QT_MOC_LITERAL(8, 66, 5) // "pause"

    },
    "PlateLog\0plateReaded\0\0logLine\0"
    "appendLogFile\0file\0line\0readLogFile\0"
    "pause"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_PlateLog[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       4,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   34,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       4,    2,   37,    2, 0x0a /* Public */,
       7,    1,   42,    2, 0x0a /* Public */,
       8,    1,   45,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void, QMetaType::QString,    3,

 // slots: parameters
    QMetaType::Void, QMetaType::QString, QMetaType::QString,    5,    6,
    QMetaType::Void, QMetaType::QString,    5,
    QMetaType::Void, QMetaType::Bool,    8,

       0        // eod
};

void PlateLog::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        PlateLog *_t = static_cast<PlateLog *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->plateReaded((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 1: _t->appendLogFile((*reinterpret_cast< QString(*)>(_a[1])),(*reinterpret_cast< QString(*)>(_a[2]))); break;
        case 2: _t->readLogFile((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 3: _t->pause((*reinterpret_cast< bool(*)>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (PlateLog::*)(QString );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&PlateLog::plateReaded)) {
                *result = 0;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject PlateLog::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_PlateLog.data,
      qt_meta_data_PlateLog,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *PlateLog::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *PlateLog::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_PlateLog.stringdata0))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int PlateLog::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
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
    return _id;
}

// SIGNAL 0
void PlateLog::plateReaded(QString _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
