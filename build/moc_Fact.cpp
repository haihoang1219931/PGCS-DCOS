/****************************************************************************
** Meta object code from reading C++ file 'Fact.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.11.3)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../src/Controller/Params/Fact.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'Fact.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.11.3. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_Fact_t {
    QByteArrayData data[20];
    char stringdata0[231];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_Fact_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_Fact_t qt_meta_stringdata_Fact = {
    {
QT_MOC_LITERAL(0, 0, 4), // "Fact"
QT_MOC_LITERAL(1, 5, 15), // "selectedChanged"
QT_MOC_LITERAL(2, 21, 0), // ""
QT_MOC_LITERAL(3, 22, 11), // "nameChanged"
QT_MOC_LITERAL(4, 34, 12), // "valueChanged"
QT_MOC_LITERAL(5, 47, 11), // "unitChanged"
QT_MOC_LITERAL(6, 59, 17), // "lowerValueChanged"
QT_MOC_LITERAL(7, 77, 17), // "upperValueChanged"
QT_MOC_LITERAL(8, 95, 17), // "lowerColorChanged"
QT_MOC_LITERAL(9, 113, 17), // "upperColorChanged"
QT_MOC_LITERAL(10, 131, 18), // "middleColorChanged"
QT_MOC_LITERAL(11, 150, 8), // "selected"
QT_MOC_LITERAL(12, 159, 4), // "name"
QT_MOC_LITERAL(13, 164, 5), // "value"
QT_MOC_LITERAL(14, 170, 4), // "unit"
QT_MOC_LITERAL(15, 175, 10), // "lowerValue"
QT_MOC_LITERAL(16, 186, 10), // "upperValue"
QT_MOC_LITERAL(17, 197, 10), // "lowerColor"
QT_MOC_LITERAL(18, 208, 10), // "upperColor"
QT_MOC_LITERAL(19, 219, 11) // "middleColor"

    },
    "Fact\0selectedChanged\0\0nameChanged\0"
    "valueChanged\0unitChanged\0lowerValueChanged\0"
    "upperValueChanged\0lowerColorChanged\0"
    "upperColorChanged\0middleColorChanged\0"
    "selected\0name\0value\0unit\0lowerValue\0"
    "upperValue\0lowerColor\0upperColor\0"
    "middleColor"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_Fact[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       9,   14, // methods
       9,   68, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       9,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    0,   59,    2, 0x06 /* Public */,
       3,    0,   60,    2, 0x06 /* Public */,
       4,    0,   61,    2, 0x06 /* Public */,
       5,    0,   62,    2, 0x06 /* Public */,
       6,    0,   63,    2, 0x06 /* Public */,
       7,    0,   64,    2, 0x06 /* Public */,
       8,    0,   65,    2, 0x06 /* Public */,
       9,    0,   66,    2, 0x06 /* Public */,
      10,    0,   67,    2, 0x06 /* Public */,

 // signals: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,

 // properties: name, type, flags
      11, QMetaType::Bool, 0x00495103,
      12, QMetaType::QString, 0x00495103,
      13, QMetaType::QString, 0x00495103,
      14, QMetaType::QString, 0x00495103,
      15, QMetaType::Double, 0x00495103,
      16, QMetaType::Double, 0x00495103,
      17, QMetaType::QString, 0x00495103,
      18, QMetaType::QString, 0x00495103,
      19, QMetaType::QString, 0x00495103,

 // properties: notify_signal_id
       0,
       1,
       2,
       3,
       4,
       5,
       6,
       7,
       8,

       0        // eod
};

void Fact::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Fact *_t = static_cast<Fact *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->selectedChanged(); break;
        case 1: _t->nameChanged(); break;
        case 2: _t->valueChanged(); break;
        case 3: _t->unitChanged(); break;
        case 4: _t->lowerValueChanged(); break;
        case 5: _t->upperValueChanged(); break;
        case 6: _t->lowerColorChanged(); break;
        case 7: _t->upperColorChanged(); break;
        case 8: _t->middleColorChanged(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (Fact::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Fact::selectedChanged)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (Fact::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Fact::nameChanged)) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (Fact::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Fact::valueChanged)) {
                *result = 2;
                return;
            }
        }
        {
            using _t = void (Fact::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Fact::unitChanged)) {
                *result = 3;
                return;
            }
        }
        {
            using _t = void (Fact::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Fact::lowerValueChanged)) {
                *result = 4;
                return;
            }
        }
        {
            using _t = void (Fact::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Fact::upperValueChanged)) {
                *result = 5;
                return;
            }
        }
        {
            using _t = void (Fact::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Fact::lowerColorChanged)) {
                *result = 6;
                return;
            }
        }
        {
            using _t = void (Fact::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Fact::upperColorChanged)) {
                *result = 7;
                return;
            }
        }
        {
            using _t = void (Fact::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Fact::middleColorChanged)) {
                *result = 8;
                return;
            }
        }
    }
#ifndef QT_NO_PROPERTIES
    else if (_c == QMetaObject::ReadProperty) {
        Fact *_t = static_cast<Fact *>(_o);
        Q_UNUSED(_t)
        void *_v = _a[0];
        switch (_id) {
        case 0: *reinterpret_cast< bool*>(_v) = _t->selected(); break;
        case 1: *reinterpret_cast< QString*>(_v) = _t->name(); break;
        case 2: *reinterpret_cast< QString*>(_v) = _t->value(); break;
        case 3: *reinterpret_cast< QString*>(_v) = _t->unit(); break;
        case 4: *reinterpret_cast< double*>(_v) = _t->lowerValue(); break;
        case 5: *reinterpret_cast< double*>(_v) = _t->upperValue(); break;
        case 6: *reinterpret_cast< QString*>(_v) = _t->lowerColor(); break;
        case 7: *reinterpret_cast< QString*>(_v) = _t->upperColor(); break;
        case 8: *reinterpret_cast< QString*>(_v) = _t->middleColor(); break;
        default: break;
        }
    } else if (_c == QMetaObject::WriteProperty) {
        Fact *_t = static_cast<Fact *>(_o);
        Q_UNUSED(_t)
        void *_v = _a[0];
        switch (_id) {
        case 0: _t->setSelected(*reinterpret_cast< bool*>(_v)); break;
        case 1: _t->setName(*reinterpret_cast< QString*>(_v)); break;
        case 2: _t->setValue(*reinterpret_cast< QString*>(_v)); break;
        case 3: _t->setUnit(*reinterpret_cast< QString*>(_v)); break;
        case 4: _t->setLowerValue(*reinterpret_cast< double*>(_v)); break;
        case 5: _t->setUpperValue(*reinterpret_cast< double*>(_v)); break;
        case 6: _t->setLowerColor(*reinterpret_cast< QString*>(_v)); break;
        case 7: _t->setUpperColor(*reinterpret_cast< QString*>(_v)); break;
        case 8: _t->setMiddleColor(*reinterpret_cast< QString*>(_v)); break;
        default: break;
        }
    } else if (_c == QMetaObject::ResetProperty) {
    }
#endif // QT_NO_PROPERTIES
    Q_UNUSED(_a);
}

QT_INIT_METAOBJECT const QMetaObject Fact::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_Fact.data,
      qt_meta_data_Fact,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *Fact::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *Fact::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_Fact.stringdata0))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int Fact::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
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
        _id -= 9;
    } else if (_c == QMetaObject::QueryPropertyDesignable) {
        _id -= 9;
    } else if (_c == QMetaObject::QueryPropertyScriptable) {
        _id -= 9;
    } else if (_c == QMetaObject::QueryPropertyStored) {
        _id -= 9;
    } else if (_c == QMetaObject::QueryPropertyEditable) {
        _id -= 9;
    } else if (_c == QMetaObject::QueryPropertyUser) {
        _id -= 9;
    }
#endif // QT_NO_PROPERTIES
    return _id;
}

// SIGNAL 0
void Fact::selectedChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 0, nullptr);
}

// SIGNAL 1
void Fact::nameChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 1, nullptr);
}

// SIGNAL 2
void Fact::valueChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 2, nullptr);
}

// SIGNAL 3
void Fact::unitChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 3, nullptr);
}

// SIGNAL 4
void Fact::lowerValueChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 4, nullptr);
}

// SIGNAL 5
void Fact::upperValueChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 5, nullptr);
}

// SIGNAL 6
void Fact::lowerColorChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 6, nullptr);
}

// SIGNAL 7
void Fact::upperColorChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 7, nullptr);
}

// SIGNAL 8
void Fact::middleColorChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 8, nullptr);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
