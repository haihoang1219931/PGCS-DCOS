/****************************************************************************
** Meta object code from reading C++ file 'UASMessage.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.11.3)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../src/Controller/UAS/UASMessage.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'UASMessage.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.11.3. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_UASMessage_t {
    QByteArrayData data[8];
    char stringdata0[97];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_UASMessage_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_UASMessage_t qt_meta_stringdata_UASMessage = {
    {
QT_MOC_LITERAL(0, 0, 10), // "UASMessage"
QT_MOC_LITERAL(1, 11, 11), // "textChanged"
QT_MOC_LITERAL(2, 23, 0), // ""
QT_MOC_LITERAL(3, 24, 19), // "formatedTextChanged"
QT_MOC_LITERAL(4, 44, 20), // "formatedColorChanged"
QT_MOC_LITERAL(5, 65, 4), // "text"
QT_MOC_LITERAL(6, 70, 12), // "formatedText"
QT_MOC_LITERAL(7, 83, 13) // "formatedColor"

    },
    "UASMessage\0textChanged\0\0formatedTextChanged\0"
    "formatedColorChanged\0text\0formatedText\0"
    "formatedColor"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_UASMessage[] = {

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
       5, QMetaType::QString, 0x00495001,
       6, QMetaType::QString, 0x00495001,
       7, QMetaType::QString, 0x00495001,

 // properties: notify_signal_id
       0,
       1,
       2,

       0        // eod
};

void UASMessage::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        UASMessage *_t = static_cast<UASMessage *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->textChanged(); break;
        case 1: _t->formatedTextChanged(); break;
        case 2: _t->formatedColorChanged(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (UASMessage::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&UASMessage::textChanged)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (UASMessage::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&UASMessage::formatedTextChanged)) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (UASMessage::*)();
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&UASMessage::formatedColorChanged)) {
                *result = 2;
                return;
            }
        }
    }
#ifndef QT_NO_PROPERTIES
    else if (_c == QMetaObject::ReadProperty) {
        UASMessage *_t = static_cast<UASMessage *>(_o);
        Q_UNUSED(_t)
        void *_v = _a[0];
        switch (_id) {
        case 0: *reinterpret_cast< QString*>(_v) = _t->getText(); break;
        case 1: *reinterpret_cast< QString*>(_v) = _t->getFormatedText(); break;
        case 2: *reinterpret_cast< QString*>(_v) = _t->getFormatedColor(); break;
        default: break;
        }
    } else if (_c == QMetaObject::WriteProperty) {
    } else if (_c == QMetaObject::ResetProperty) {
    }
#endif // QT_NO_PROPERTIES
    Q_UNUSED(_a);
}

QT_INIT_METAOBJECT const QMetaObject UASMessage::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_UASMessage.data,
      qt_meta_data_UASMessage,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *UASMessage::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *UASMessage::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_UASMessage.stringdata0))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int UASMessage::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
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
void UASMessage::textChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 0, nullptr);
}

// SIGNAL 1
void UASMessage::formatedTextChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 1, nullptr);
}

// SIGNAL 2
void UASMessage::formatedColorChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 2, nullptr);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
