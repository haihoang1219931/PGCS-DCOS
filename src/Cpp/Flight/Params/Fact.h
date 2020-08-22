#ifndef FACT_H
#define FACT_H

#include <QObject>

class Fact : public QObject
{
    Q_OBJECT
    Q_PROPERTY(bool  selected       READ selected   WRITE setSelected NOTIFY selectedChanged)
    Q_PROPERTY(QString  name   READ name  WRITE setName NOTIFY nameChanged)
    Q_PROPERTY(QString  value       READ value      WRITE setValue NOTIFY valueChanged)
    Q_PROPERTY(QString  unit        READ unit       WRITE setUnit NOTIFY unitChanged)
public:
    explicit Fact(QObject *parent = nullptr){}
    Fact(bool  selected, QString  name, QString  value, QString  unit){
        _selected = selected;
        sprintf(_name,"%s",name.toStdString().c_str());
        sprintf(_value,"%s",value.toStdString().c_str());
        sprintf(_unit,"%s",unit.toStdString().c_str());
    }
    ~Fact(){}
public:
    bool selected(){ return _selected;}
    QString name(){ return QString(_name);}
    QString value(){ return QString(_value);}
    QString unit(){ return QString(_unit);}
    void setSelected(bool value){
        if(_selected!=value){
            _selected = value;
            Q_EMIT selectedChanged();
        }
    }
    void setName(QString value){
        if(_name!=value){
            sprintf(_name,"%s",value.toStdString().c_str());
            Q_EMIT nameChanged();
        }
    }
    void setValue(QString value){
        if(_value!=value){
            sprintf(_value,"%s",value.replace(",",".").toStdString().c_str());
            Q_EMIT valueChanged();
        }
    }
    void setUnit(QString value){
        if(_unit!=value){
            sprintf(_unit,"%s",value.toStdString().c_str());
            Q_EMIT unitChanged();
        }
    }
Q_SIGNALS:
    void selectedChanged();
    void nameChanged();
    void valueChanged();
    void unitChanged();
private:
    bool _selected = false;
    char _name[256];
    char _value[256];
    char _unit[256];
};

#endif // FACT_H
