/****************************************************************************
** Meta object code from reading C++ file 'MarkerList.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.11.3)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../src/Maplib/Marker/MarkerList.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'MarkerList.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.11.3. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_MarkerList_t {
    QByteArrayData data[16];
    char stringdata0[141];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_MarkerList_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_MarkerList_t qt_meta_stringdata_MarkerList = {
    {
QT_MOC_LITERAL(0, 0, 10), // "MarkerList"
QT_MOC_LITERAL(1, 11, 11), // "cleanMarker"
QT_MOC_LITERAL(2, 23, 0), // ""
QT_MOC_LITERAL(3, 24, 12), // "insertMarker"
QT_MOC_LITERAL(4, 37, 7), // "Marker*"
QT_MOC_LITERAL(5, 45, 7), // "_marker"
QT_MOC_LITERAL(6, 53, 3), // "lat"
QT_MOC_LITERAL(7, 57, 3), // "lon"
QT_MOC_LITERAL(8, 61, 4), // "type"
QT_MOC_LITERAL(9, 66, 11), // "description"
QT_MOC_LITERAL(10, 78, 11), // "saveMarkers"
QT_MOC_LITERAL(11, 90, 8), // "fileName"
QT_MOC_LITERAL(12, 99, 11), // "loadMarkers"
QT_MOC_LITERAL(13, 111, 9), // "numMarker"
QT_MOC_LITERAL(14, 121, 9), // "getMarker"
QT_MOC_LITERAL(15, 131, 9) // "_markerID"

    },
    "MarkerList\0cleanMarker\0\0insertMarker\0"
    "Marker*\0_marker\0lat\0lon\0type\0description\0"
    "saveMarkers\0fileName\0loadMarkers\0"
    "numMarker\0getMarker\0_markerID"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_MarkerList[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       8,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // methods: name, argc, parameters, tag, flags
       1,    0,   54,    2, 0x02 /* Public */,
       3,    1,   55,    2, 0x02 /* Public */,
       3,    4,   58,    2, 0x02 /* Public */,
       3,    3,   67,    2, 0x22 /* Public | MethodCloned */,
      10,    1,   74,    2, 0x02 /* Public */,
      12,    1,   77,    2, 0x02 /* Public */,
      13,    0,   80,    2, 0x02 /* Public */,
      14,    1,   81,    2, 0x02 /* Public */,

 // methods: parameters
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 4,    5,
    QMetaType::Void, QMetaType::QString, QMetaType::QString, QMetaType::QString, QMetaType::QString,    6,    7,    8,    9,
    QMetaType::Void, QMetaType::QString, QMetaType::QString, QMetaType::QString,    6,    7,    8,
    QMetaType::Void, QMetaType::QString,   11,
    QMetaType::Void, QMetaType::QString,   11,
    QMetaType::Int,
    0x80000000 | 4, QMetaType::Int,   15,

       0        // eod
};

void MarkerList::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        MarkerList *_t = static_cast<MarkerList *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->cleanMarker(); break;
        case 1: _t->insertMarker((*reinterpret_cast< Marker*(*)>(_a[1]))); break;
        case 2: _t->insertMarker((*reinterpret_cast< QString(*)>(_a[1])),(*reinterpret_cast< QString(*)>(_a[2])),(*reinterpret_cast< QString(*)>(_a[3])),(*reinterpret_cast< QString(*)>(_a[4]))); break;
        case 3: _t->insertMarker((*reinterpret_cast< QString(*)>(_a[1])),(*reinterpret_cast< QString(*)>(_a[2])),(*reinterpret_cast< QString(*)>(_a[3]))); break;
        case 4: _t->saveMarkers((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 5: _t->loadMarkers((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 6: { int _r = _t->numMarker();
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = std::move(_r); }  break;
        case 7: { Marker* _r = _t->getMarker((*reinterpret_cast< int(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< Marker**>(_a[0]) = std::move(_r); }  break;
        default: ;
        }
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        switch (_id) {
        default: *reinterpret_cast<int*>(_a[0]) = -1; break;
        case 1:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< Marker* >(); break;
            }
            break;
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject MarkerList::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_MarkerList.data,
      qt_meta_data_MarkerList,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *MarkerList::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *MarkerList::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_MarkerList.stringdata0))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int MarkerList::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
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
            qt_static_metacall(this, _c, _id, _a);
        _id -= 8;
    }
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
