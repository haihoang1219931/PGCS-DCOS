/****************************************************************************
** Meta object code from reading C++ file 'MissionItem.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.11.3)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../src/Controller/Mission/MissionItem.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'MissionItem.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.11.3. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_MissionItem_t {
    QByteArrayData data[16];
    char stringdata0[142];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_MissionItem_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_MissionItem_t qt_meta_stringdata_MissionItem = {
    {
QT_MOC_LITERAL(0, 0, 11), // "MissionItem"
QT_MOC_LITERAL(1, 12, 7), // "command"
QT_MOC_LITERAL(2, 20, 5), // "frame"
QT_MOC_LITERAL(3, 26, 6), // "option"
QT_MOC_LITERAL(4, 33, 6), // "param1"
QT_MOC_LITERAL(5, 40, 6), // "param2"
QT_MOC_LITERAL(6, 47, 6), // "param3"
QT_MOC_LITERAL(7, 54, 6), // "param4"
QT_MOC_LITERAL(8, 61, 6), // "param5"
QT_MOC_LITERAL(9, 68, 6), // "param6"
QT_MOC_LITERAL(10, 75, 6), // "param7"
QT_MOC_LITERAL(11, 82, 12), // "autoContinue"
QT_MOC_LITERAL(12, 95, 13), // "isCurrentItem"
QT_MOC_LITERAL(13, 109, 8), // "sequence"
QT_MOC_LITERAL(14, 118, 8), // "position"
QT_MOC_LITERAL(15, 127, 14) // "QGeoCoordinate"

    },
    "MissionItem\0command\0frame\0option\0"
    "param1\0param2\0param3\0param4\0param5\0"
    "param6\0param7\0autoContinue\0isCurrentItem\0"
    "sequence\0position\0QGeoCoordinate"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_MissionItem[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
      14,   14, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // properties: name, type, flags
       1, QMetaType::Int, 0x00095103,
       2, QMetaType::Int, 0x00095103,
       3, QMetaType::Int, 0x00095103,
       4, QMetaType::Float, 0x00095103,
       5, QMetaType::Float, 0x00095103,
       6, QMetaType::Float, 0x00095103,
       7, QMetaType::Float, 0x00095103,
       8, QMetaType::Float, 0x00095103,
       9, QMetaType::Float, 0x00095103,
      10, QMetaType::Float, 0x00095103,
      11, QMetaType::Bool, 0x00095103,
      12, QMetaType::Bool, 0x00095103,
      13, QMetaType::Int, 0x00095103,
      14, 0x80000000 | 15, 0x0009510b,

       0        // eod
};

void MissionItem::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::RegisterPropertyMetaType) {
        switch (_id) {
        default: *reinterpret_cast<int*>(_a[0]) = -1; break;
        case 13:
            *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QGeoCoordinate >(); break;
        }
    }

#ifndef QT_NO_PROPERTIES
    else if (_c == QMetaObject::ReadProperty) {
        MissionItem *_t = static_cast<MissionItem *>(_o);
        Q_UNUSED(_t)
        void *_v = _a[0];
        switch (_id) {
        case 0: *reinterpret_cast< int*>(_v) = _t->command(); break;
        case 1: *reinterpret_cast< int*>(_v) = _t->frame(); break;
        case 2: *reinterpret_cast< int*>(_v) = _t->option(); break;
        case 3: *reinterpret_cast< float*>(_v) = _t->param1(); break;
        case 4: *reinterpret_cast< float*>(_v) = _t->param2(); break;
        case 5: *reinterpret_cast< float*>(_v) = _t->param3(); break;
        case 6: *reinterpret_cast< float*>(_v) = _t->param4(); break;
        case 7: *reinterpret_cast< float*>(_v) = _t->param5(); break;
        case 8: *reinterpret_cast< float*>(_v) = _t->param6(); break;
        case 9: *reinterpret_cast< float*>(_v) = _t->param7(); break;
        case 10: *reinterpret_cast< bool*>(_v) = _t->autoContinue(); break;
        case 11: *reinterpret_cast< bool*>(_v) = _t->isCurrentItem(); break;
        case 12: *reinterpret_cast< int*>(_v) = _t->sequence(); break;
        case 13: *reinterpret_cast< QGeoCoordinate*>(_v) = _t->position(); break;
        default: break;
        }
    } else if (_c == QMetaObject::WriteProperty) {
        MissionItem *_t = static_cast<MissionItem *>(_o);
        Q_UNUSED(_t)
        void *_v = _a[0];
        switch (_id) {
        case 0: _t->setCommand(*reinterpret_cast< int*>(_v)); break;
        case 1: _t->setFrame(*reinterpret_cast< int*>(_v)); break;
        case 2: _t->setOption(*reinterpret_cast< int*>(_v)); break;
        case 3: _t->setParam1(*reinterpret_cast< float*>(_v)); break;
        case 4: _t->setParam2(*reinterpret_cast< float*>(_v)); break;
        case 5: _t->setParam3(*reinterpret_cast< float*>(_v)); break;
        case 6: _t->setParam4(*reinterpret_cast< float*>(_v)); break;
        case 7: _t->setParam5(*reinterpret_cast< float*>(_v)); break;
        case 8: _t->setParam6(*reinterpret_cast< float*>(_v)); break;
        case 9: _t->setParam7(*reinterpret_cast< float*>(_v)); break;
        case 10: _t->setAutoContinue(*reinterpret_cast< bool*>(_v)); break;
        case 11: _t->setIsCurrentItem(*reinterpret_cast< bool*>(_v)); break;
        case 12: _t->setSequence(*reinterpret_cast< int*>(_v)); break;
        case 13: _t->setPosition(*reinterpret_cast< QGeoCoordinate*>(_v)); break;
        default: break;
        }
    } else if (_c == QMetaObject::ResetProperty) {
    }
#endif // QT_NO_PROPERTIES
    Q_UNUSED(_o);
}

QT_INIT_METAOBJECT const QMetaObject MissionItem::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_MissionItem.data,
      qt_meta_data_MissionItem,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *MissionItem::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *MissionItem::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_MissionItem.stringdata0))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int MissionItem::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    
#ifndef QT_NO_PROPERTIES
   if (_c == QMetaObject::ReadProperty || _c == QMetaObject::WriteProperty
            || _c == QMetaObject::ResetProperty || _c == QMetaObject::RegisterPropertyMetaType) {
        qt_static_metacall(this, _c, _id, _a);
        _id -= 14;
    } else if (_c == QMetaObject::QueryPropertyDesignable) {
        _id -= 14;
    } else if (_c == QMetaObject::QueryPropertyScriptable) {
        _id -= 14;
    } else if (_c == QMetaObject::QueryPropertyStored) {
        _id -= 14;
    } else if (_c == QMetaObject::QueryPropertyEditable) {
        _id -= 14;
    } else if (_c == QMetaObject::QueryPropertyUser) {
        _id -= 14;
    }
#endif // QT_NO_PROPERTIES
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
