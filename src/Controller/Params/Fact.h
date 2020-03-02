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
        _name = name;
        _value = value;
        _unit = unit;
    }
    ~Fact(){}
public:
    bool selected(){ return _selected;}
    QString name(){ return _name;}
    QString value(){ return _value;}
    QString unit(){ return _unit;}
    void setSelected(bool value){
        if(_selected!=value){
            _selected = value;
            Q_EMIT selectedChanged();
        }
    }
    void setName(QString value){
        if(_name!=value){
            _name = value;
            Q_EMIT nameChanged();
        }
    }
    void setValue(QString value){
        if(_value!=value){
            _value = value;
            Q_EMIT valueChanged();
        }
    }
    void setUnit(QString value){
        if(_unit!=value){
            _unit = value;
            Q_EMIT unitChanged();
        }
    }
Q_SIGNALS:
    void selectedChanged();
    void nameChanged();
    void valueChanged();
    void unitChanged();
private:
    bool _selected;
    QString _name;
    QString _value;
    QString _unit;
};

#endif // FACT_H
