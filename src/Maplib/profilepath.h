#ifndef PROFILEPATH_H
#define PROFILEPATH_H
#include <QObject>
#include <QtQuick>
#include <QPointF>
#include <QColor>
#include <QGeoCoordinate>
#include "Elevation.h"

class ProfilePath:public QQuickPaintedItem
{
    Q_OBJECT
    Q_PROPERTY(QColor color READ color WRITE setColor NOTIFY colorChanged)
    Q_PROPERTY(QString title READ title WRITE setTitle NOTIFY titleChanged)
    Q_PROPERTY(QString xName READ xName WRITE setXName NOTIFY xNameChanged)
    Q_PROPERTY(QString yName READ yName WRITE setYName NOTIFY yNameChanged)
    Q_PROPERTY(int fontSize READ fontSize WRITE setFontSize NOTIFY fontSizeChanged)
    Q_PROPERTY(QString fontFamily READ fontFamily WRITE setFontFamily NOTIFY fontFamilyChanged)
public:
    QColor color(){ return _color;}
    QString title(){ return _title;}
    QString xName(){ return _xName;}
    QString yName(){ return _yName;}
    int     fontSize(){ return _fontSize;}
    QString fontFamily(){ return _fontFamily;}

    void setColor(QColor value){
        if(_color != value){
            _color = value;
            Q_EMIT colorChanged();
        }
    }
    void setTitle(QString value){
        if(_title != value){
            _title = value;
            Q_EMIT titleChanged();
        }
    }
    void setXName(QString value){
        if(_xName != value){
            _xName = value;
            Q_EMIT xNameChanged();
        }
    }
    void setYName(QString value){
        if(_yName != value){
            _yName = value;
            Q_EMIT yNameChanged();
        }
    }
    void setFontSize(int value){
        if(_fontSize != value){
            _fontSize = value;
            AxisXoffset = _fontSize * 3;
            AxisYoffset = _fontSize * 3;
            Q_EMIT fontSizeChanged();
        }
    }
    void setFontFamily(QString value){
        if(_fontFamily != value){
            _fontFamily = value;
            Q_EMIT fontFamilyChanged();
        }
    }

    ProfilePath(QQuickItem *parent = 0);
    void paint(QPainter *painter);

    void changePen(QPen *pen,QString color,int width);
    void changeFont(QFont *font,QString fontFamily,int size);
    void drawPlot(QPainter *painter);
    float getAltitude(QString folder,QGeoCoordinate coord);
    void insertProfilePath(int distance,int altitude);
    void clearProfilePath();

    Q_INVOKABLE void addElevation(QString folder,QGeoCoordinate startcoord,QGeoCoordinate tocoord);
Q_SIGNALS:
    void colorChanged();
    void titleChanged();
    void xNameChanged();
    void yNameChanged();
    void fontSizeChanged();
    void fontFamilyChanged();
private:
    QColor _color;
    QString _title;
    QString _xName;
    QString _yName;
    int _fontSize;
    QString _fontFamily;
    int AxisXoffset = 45;
    int AxisYoffset = 50;

    int mMaxAltitude=0;
    int mMinAltitude=0;
    int mMinDistance=0;
    int mMaxDistance=0;

    QMap<int,int> mListAltitude;
    Elevation mElevation;

};

#endif // PROFILEPATH_H
