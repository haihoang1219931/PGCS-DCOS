/****************************************************************************
** Meta object code from reading C++ file 'LinkInterface.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.11.3)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../src/Controller/Com/LinkInterface.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'LinkInterface.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.11.3. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_LinkInterface_t {
    QByteArrayData data[11];
    char stringdata0[125];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_LinkInterface_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_LinkInterface_t qt_meta_stringdata_LinkInterface = {
    {
QT_MOC_LITERAL(0, 0, 13), // "LinkInterface"
QT_MOC_LITERAL(1, 14, 13), // "statusChanged"
QT_MOC_LITERAL(2, 28, 0), // ""
QT_MOC_LITERAL(3, 29, 11), // "hasReadSome"
QT_MOC_LITERAL(4, 41, 3), // "msg"
QT_MOC_LITERAL(5, 45, 15), // "closeConnection"
QT_MOC_LITERAL(6, 61, 12), // "connect2host"
QT_MOC_LITERAL(7, 74, 9), // "readyRead"
QT_MOC_LITERAL(8, 84, 9), // "connected"
QT_MOC_LITERAL(9, 94, 12), // "disconnected"
QT_MOC_LITERAL(10, 107, 17) // "connectionTimeout"

    },
    "LinkInterface\0statusChanged\0\0hasReadSome\0"
    "msg\0closeConnection\0connect2host\0"
    "readyRead\0connected\0disconnected\0"
    "connectionTimeout"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_LinkInterface[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       8,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       2,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   54,    2, 0x06 /* Public */,
       3,    1,   57,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       5,    0,   60,    2, 0x0a /* Public */,
       6,    0,   61,    2, 0x0a /* Public */,
       7,    0,   62,    2, 0x0a /* Public */,
       8,    0,   63,    2, 0x0a /* Public */,
       9,    0,   64,    2, 0x0a /* Public */,
      10,    0,   65,    2, 0x0a /* Public */,

 // signals: parameters
    QMetaType::Void, QMetaType::Bool,    2,
    QMetaType::Void, QMetaType::QByteArray,    4,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void LinkInterface::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        LinkInterface *_t = static_cast<LinkInterface *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->statusChanged((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 1: _t->hasReadSome((*reinterpret_cast< QByteArray(*)>(_a[1]))); break;
        case 2: _t->closeConnection(); break;
        case 3: _t->connect2host(); break;
        case 4: _t->readyRead(); break;
        case 5: _t->connected(); break;
        case 6: _t->disconnected(); break;
        case 7: _t->connectionTimeout(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (LinkInterface::*)(bool );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&LinkInterface::statusChanged)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (LinkInterface::*)(QByteArray );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&LinkInterface::hasReadSome)) {
                *result = 1;
                return;
            }
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject LinkInterface::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_LinkInterface.data,
      qt_meta_data_LinkInterface,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *LinkInterface::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *LinkInterface::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_LinkInterface.stringdata0))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int LinkInterface::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 8)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 8;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 8)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 8;
    }
    return _id;
}

// SIGNAL 0
void LinkInterface::statusChanged(bool _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void LinkInterface::hasReadSome(QByteArray _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
