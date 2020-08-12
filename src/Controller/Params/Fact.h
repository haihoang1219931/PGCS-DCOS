#ifndef FACT_H
#define FACT_H

#include <QObject>
#include <QColor>
class Fact : public QObject
{
    Q_OBJECT
    Q_PROPERTY(bool  selected       READ selected   WRITE setSelected NOTIFY selectedChanged)
    Q_PROPERTY(QString  name   READ name  WRITE setName NOTIFY nameChanged)
    Q_PROPERTY(QString  value       READ value      WRITE setValue NOTIFY valueChanged)
    Q_PROPERTY(QString  unit        READ unit       WRITE setUnit NOTIFY unitChanged)
    Q_PROPERTY(double   lowerValue  READ lowerValue WRITE setLowerValue NOTIFY lowerValueChanged)
    Q_PROPERTY(double   upperValue  READ upperValue WRITE setUpperValue NOTIFY upperValueChanged)
    Q_PROPERTY(QString   lowerColor  READ lowerColor WRITE setLowerColor NOTIFY lowerColorChanged)
    Q_PROPERTY(QString   upperColor  READ upperColor WRITE setUpperColor NOTIFY upperColorChanged)
    Q_PROPERTY(QString   middleColor READ middleColor WRITE setMiddleColor NOTIFY middleColorChanged)

public:
    explicit Fact(QObject *parent = nullptr){}
    Fact(bool  selected, QString  name, QString  value, QString  unit, double lowerValue=0, double upperValue=0,
         QString lowerColor=QString("transparent"), QString upperColor=QString("transparent"), QString middleColor=QString("transparent") ){
        _selected = selected;
        sprintf(_name,"%s",name.toStdString().c_str());
        sprintf(_value,"%s",value.toStdString().c_str());
        sprintf(_unit,"%s",unit.toStdString().c_str());
        _lowerValue = lowerValue;
        _upperValue = upperValue;
        _lowerColor = lowerColor;
        _upperColor = upperColor;
        _middleColor = middleColor;
    }
    ~Fact(){}
public:
    bool selected(){ return _selected;}
    QString name(){ return QString(_name);}
    QString value(){ return QString(_value);}
    QString unit(){ return QString(_unit);}
    double lowerValue(){return _lowerValue;}
    double upperValue(){return _upperValue;}
    QString lowerColor(){return _lowerColor;}
    QString upperColor(){return _upperColor;}
    QString middleColor(){return _middleColor;}


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
    void setLowerValue(double value){
        if(_lowerValue!=value){
            _lowerValue = value;
            Q_EMIT lowerValueChanged();
        }
    }
    void setUpperValue(double value){
        if(_upperValue!=value){
            _upperValue = value;
            Q_EMIT upperValueChanged();
        }
    }
    void setLowerColor(QString value){
        if(_lowerColor!=value){
            _lowerColor = value;
            Q_EMIT lowerColorChanged();
        }
    }
    void setUpperColor(QString value){
        if(_upperColor!=value){
            _upperColor = value;
            Q_EMIT upperColorChanged();
        }
    }
    void setMiddleColor(QString value){
        if(_middleColor!=value){
            _middleColor = value;
            Q_EMIT middleColorChanged();
        }
    }

Q_SIGNALS:
    void selectedChanged();
    void nameChanged();
    void valueChanged();
    void unitChanged();
    void lowerValueChanged();
    void upperValueChanged();
    void lowerColorChanged();
    void upperColorChanged();
    void middleColorChanged();

private:
    bool _selected = false;
    char _name[256];
    char _value[256];
    char _unit[256];
    double _lowerValue = 0;
    double _upperValue = 0;
    QString _lowerColor;
    QString _upperColor;
    QString _middleColor;
};

#endif // FACT_H
