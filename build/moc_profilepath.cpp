/****************************************************************************
** Meta object code from reading C++ file 'profilepath.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.11.3)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../src/Maplib/profilepath.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'profilepath.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.11.3. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_ProfilePath_t {
    QByteArrayData data[19];
    char stringdata0[197];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_ProfilePath_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_ProfilePath_t qt_meta_stringdata_ProfilePath = {
    {
QT_MOC_LITERAL(0, 0, 11), // "ProfilePath"
QT_MOC_LITERAL(1, 12, 12), // "colorChanged"
QT_MOC_LITERAL(2, 25, 0), // ""
QT_MOC_LITERAL(3, 26, 12), // "titleChanged"
QT_MOC_LITERAL(4, 39, 12), // "xNameChanged"
QT_MOC_LITERAL(5, 52, 12), // "yNameChanged"
QT_MOC_LITERAL(6, 65, 15), // "fontSizeChanged"
QT_MOC_LITERAL(7, 81, 17), // "fontFamilyChanged"
QT_MOC_LITERAL(8, 99, 12), // "addElevation"
QT_MOC_LITERAL(9, 112, 6), // "folder"
QT_MOC_LITERAL(10, 119, 14), // "QGeoCoordinate"
QT_MOC_LITERAL(11, 134, 10), // "startcoord"
QT_MOC_LITERAL(12, 145, 7), // "tocoord"
QT_MOC_LITERAL(13, 153, 5), // "color"
QT_MOC_LITERAL(14, 159, 5), // "title"
QT_MOC_LITERAL(15, 165, 5), // "xName"
QT_MOC_LITERAL(16, 171, 5), // "yName"
QT_MOC_LITERAL(17, 177, 8), // "fontSize"
QT_MOC_LITERAL(18, 186, 10) // "fontFamily"

    },
    "ProfilePath\0colorChanged\0\0titleChanged\0"
    "xNameChanged\0yNameChanged\0fontSizeChanged\0"
    "fontFamilyChanged\0addElevation\0folder\0"
    "QGeoCoordinate\0startcoord\0tocoord\0"
    "color\0title\0xName\0yName\0fontSize\0"
    "fontFamily"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_ProfilePath[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       7,   14, // methods
       6,   62, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       6,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    0,   49,    2, 0x06 /* Public */,
       3,    0,   50,    2, 0x06 /* Public */,
       4,    0,   51,    2, 0x06 /* Public */,
       5,    0,   52,    2, 0x06 /* Public */,
       6,    0,   53,    2, 0x06 /* Public */,
       7,    0,   54,    2, 0x06 /* Public */,

 // methods: name, argc, parameters, tag, flags
       8,    3,   55,    2, 0x02 /* Public */,

 // signals: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,

 // methods: parameters
    QMetaType::Void, QMetaType::QString, 0x80000000 | 10, 0x80000000 | 10,    9,   11,   12,

 // properties: name, type, flags
      13, QMetaType::QColor, 0x00495103,
      14, QMetaType::QString, 0x00495103,
      15, QMetaType::QString, 0x00495103,
      16, QMetaType::QString, 0x00495103,
      17, QMetaType::Int, 0x00495103,
      18, QMetaType::QString, 0x00495103,

 // properties: notify_signal_id
       0,
       1,
       2,
       3,
       4,
       5,

       0        // eod
};

void ProfilePath::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        ProfilePath *_t = static_cast<ProfilePath *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->colorChanged(); break;
        case 1: _t->titleChanged(); break;
        case 2: _t->xNameChanged(); break;
        case 3: _t->yNameChanged(); break;
        case 4: _t->fontSizeChanged(); break;
        case 5: _t->fontFamilyChanged(); break;
        case 6: _t->addElevation((*reinterpret_cast< QString(*)>(_a[1])),(*reinterpret_cast< QGeoCoordinate(*)>(_a[2])),(*reinterpret_cast< QGeoCoordinate(*)>(_a[3]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        switch (_id) {
        default: *reinterpret_cast<int*>(_a[0]) = -1; break;
        case 6:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 2:
            case 1:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QGeoCoordinate >(); break;
            }
            break;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (ProfilePath::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&ProfilePath::colorChanged)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (ProfilePath::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&ProfilePath::titleChanged)) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (ProfilePath::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&ProfilePath::xNameChanged)) {
                *result = 2;
                return;
            }
        }
        {
            using _t = void (ProfilePath::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&ProfilePath::yNameChanged)) {
                *result = 3;
                return;
            }
        }
        {
            using _t = void (ProfilePath::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&ProfilePath::fontSizeChanged)) {
                *result = 4;
                return;
            }
        }
        {
            using _t = void (ProfilePath::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&ProfilePath::fontFamilyChanged)) {
                *result = 5;
                return;
            }
        }
    }
#ifndef QT_NO_PROPERTIES
    else if (_c == QMetaObject::ReadProperty) {
        ProfilePath *_t = static_cast<ProfilePath *>(_o);
        Q_UNUSED(_t)
        void *_v = _a[0];
        switch (_id) {
        case 0: *reinterpret_cast< QColor*>(_v) = _t->color(); break;
        case 1: *reinterpret_cast< QString*>(_v) = _t->title(); break;
        case 2: *reinterpret_cast< QString*>(_v) = _t->xName(); break;
        case 3: *reinterpret_cast< QString*>(_v) = _t->yName(); break;
        case 4: *reinterpret_cast< int*>(_v) = _t->fontSize(); break;
        case 5: *reinterpret_cast< QString*>(_v) = _t->fontFamily(); break;
        default: break;
        }
    } else if (_c == QMetaObject::WriteProperty) {
        ProfilePath *_t = static_cast<ProfilePath *>(_o);
        Q_UNUSED(_t)
        void *_v = _a[0];
        switch (_id) {
        case 0: _t->setColor(*reinterpret_cast< QColor*>(_v)); break;
        case 1: _t->setTitle(*reinterpret_cast< QString*>(_v)); break;
        case 2: _t->setXName(*reinterpret_cast< QString*>(_v)); break;
        case 3: _t->setYName(*reinterpret_cast< QString*>(_v)); break;
        case 4: _t->setFontSize(*reinterpret_cast< int*>(_v)); break;
        case 5: _t->setFontFamily(*reinterpret_cast< QString*>(_v)); break;
        default: break;
        }
    } else if (_c == QMetaObject::ResetProperty) {
    }
#endif // QT_NO_PROPERTIES
}

QT_INIT_METAOBJECT const QMetaObject ProfilePath::staticMetaObject = {
    { &QQuickPaintedItem::staticMetaObject, qt_meta_stringdata_ProfilePath.data,
      qt_meta_data_ProfilePath,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *ProfilePath::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *ProfilePath::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_ProfilePath.stringdata0))
        return static_cast<void*>(this);
    return QQuickPaintedItem::qt_metacast(_clname);
}

int ProfilePath::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QQuickPaintedItem::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 7)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 7;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 7)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 7;
    }
#ifndef QT_NO_PROPERTIES
   else if (_c == QMetaObject::ReadProperty || _c == QMetaObject::WriteProperty
            || _c == QMetaObject::ResetProperty || _c == QMetaObject::RegisterPropertyMetaType) {
        qt_static_metacall(this, _c, _id, _a);
        _id -= 6;
    } else if (_c == QMetaObject::QueryPropertyDesignable) {
        _id -= 6;
    } else if (_c == QMetaObject::QueryPropertyScriptable) {
        _id -= 6;
    } else if (_c == QMetaObject::QueryPropertyStored) {
        _id -= 6;
    } else if (_c == QMetaObject::QueryPropertyEditable) {
        _id -= 6;
    } else if (_c == QMetaObject::QueryPropertyUser) {
        _id -= 6;
    }
#endif // QT_NO_PROPERTIES
    return _id;
}

// SIGNAL 0
void ProfilePath::colorChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 0, nullptr);
}

// SIGNAL 1
void ProfilePath::titleChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 1, nullptr);
}

// SIGNAL 2
void ProfilePath::xNameChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 2, nullptr);
}

// SIGNAL 3
void ProfilePath::yNameChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 3, nullptr);
}

// SIGNAL 4
void ProfilePath::fontSizeChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 4, nullptr);
}

// SIGNAL 5
void ProfilePath::fontFamilyChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 5, nullptr);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
