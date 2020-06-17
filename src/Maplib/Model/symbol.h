#ifndef SYMBOL_H
#define SYMBOL_H
#include <QGeoCoordinate>

class symbol
{
public:
    symbol();
    symbol(const int& id,const int& type,const int& param1 ,const int& param2,const int& param3,const int& param4,const QString& text,const QGeoCoordinate& coordinate);

    int Id;
    int Type;

    int Param1;
    int Param2;
    int Param3;
    int Param4;

    QString Text;

    QGeoCoordinate Coordinate;

private:
    int _mid;
    int _mtype;
    int _mparam1;
    int _mparam2;
    int _mparam3;
    int _mparam4;
    QString _mtext;
    QGeoCoordinate _mcoordinate;
};

#endif // SYMBOL_H
