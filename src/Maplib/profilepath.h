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
    Q_PROPERTY(QString folderPath READ folderPath WRITE setFolderPath NOTIFY folderPathChanged)

    Q_PROPERTY(int axisXOffset READ xOffset NOTIFY xOffsetChanged)
    Q_PROPERTY(int axisYOffset READ yOffset NOTIFY yOffsetChanged)

    Q_PROPERTY(bool isShowLineOfSight READ isShowLineOfSight WRITE setIsShowLineOfSight NOTIFY isShowLineOfSightChanged)

public:
    QColor color(){ return _color;}
    QString title(){ return _title;}
    QString xName(){ return _xName;}
    QString yName(){ return _yName;}
    int     fontSize(){ return _fontSize;}
    QString fontFamily(){ return _fontFamily;}
    QString folderPath(){ return _folderPath;}

    void setColor(QColor value){
        if(_color != value){
            _color = value;
            this->update();
            Q_EMIT colorChanged();
        }
    }
    void setTitle(QString value){
        if(_title != value){
            _title = value;
            this->update();
            Q_EMIT titleChanged();

        }
    }
    void setXName(QString value){
        if(_xName != value){
            _xName = value;
            this->update();
            Q_EMIT xNameChanged();
        }
    }
    void setYName(QString value){
        if(_yName != value){
            _yName = value;
            this->update();
            Q_EMIT yNameChanged();
        }
    }
    void setFontSize(int value){
        if(_fontSize != value){
            _fontSize = value;
            mAxisXoffset = _fontSize * 3;
            mAxisYoffset = _fontSize * 3;
            this->update();
            Q_EMIT fontSizeChanged();
        }
    }
    void setFontFamily(QString value){
        if(_fontFamily != value){
            _fontFamily = value;
            this->update();
            Q_EMIT fontFamilyChanged();
        }
    }

    void setIsShowLineOfSight(bool value){
        if(_isShowLineOfSight != value){
            _isShowLineOfSight = value;
            Q_EMIT isShowLineOfSightChanged();
        }
    }

    void setFolderPath(QString path){
        if(_folderPath != path){
            _folderPath = path;
            Q_EMIT folderPathChanged();
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

    bool isShowLineOfSight(){return _isShowLineOfSight;}

    QGeoCoordinate findSlideShowCoord(QGeoCoordinate coord,QGeoCoordinate startcoord, QGeoCoordinate tocoord);

    Q_INVOKABLE QGeoCoordinate planePosPrediction(QGeoCoordinate planePos, QGeoCoordinate toPos, double planeGroundSpeed, double timeSec);
    Q_INVOKABLE bool checkAltitudeWarning(QGeoCoordinate planePos, QGeoCoordinate posPrediction);



    Q_INVOKABLE void addElevation(QGeoCoordinate startcoord,QGeoCoordinate tocoord);
    Q_INVOKABLE QPoint convertCoordinatetoXY(QGeoCoordinate coord,QGeoCoordinate startcoord, QGeoCoordinate tocoord);
    Q_INVOKABLE void setLineOfSight(double fromDis,double fromAlt,double toDis,double toAlt);


Q_SIGNALS:
    void colorChanged();
    void titleChanged();
    void xNameChanged();
    void yNameChanged();
    void fontSizeChanged();
    void fontFamilyChanged();
    void xOffsetChanged();
    void yOffsetChanged();
    void isShowLineOfSightChanged();
    void folderPathChanged();
private:
    QColor _color;
    QString _title;
    QString _xName;
    QString _yName;
    int _fontSize;
    QString _fontFamily;
    QString _folderPath;
    int mAxisXoffset = 45;
    int mAxisYoffset = 50;

    int mMaxAltitude=0;
    int mMinAltitude=0;
    int mMinDistance=0;
    int mMaxDistance=0;

    double _fromDistance = 0;
    double _fromAltitude = 0;
    double _toDistance = 0;
    double _toAltitude = 0;

    int mMaxVehicleAlt = 0;

    bool _isShowLineOfSight = false;

    QMap<int,int> mListAltitude;
    Elevation mElevation;

    int xOffset(){return mAxisXoffset;}
    int yOffset(){return mAxisYoffset;}

    double max_altitude_point_Y=0;
    double max_distance_point_X=0;

    double getAltitudeCoordinate(QGeoCoordinate getCoord, QGeoCoordinate coord1, QGeoCoordinate coord2);

};

#endif // PROFILEPATH_H
