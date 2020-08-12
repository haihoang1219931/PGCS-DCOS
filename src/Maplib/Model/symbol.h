#ifndef SYMBOL_H
#define SYMBOL_H
#include <QGeoCoordinate>

class symbol
{
public:
    symbol();
    symbol(const int& id,const int& type,const int& alt,const int& toWP,const int& repeat,const int& dircircle,const int& timecircle,const QGeoCoordinate& coordinate);
    int Id;
    int Type;
    int Alt;
    int ToWP;
    int Repeat;
    int DirCircle;
    int TimeCircle;

    QGeoCoordinate Coordinate;

private:
    int _mid;
    int _mtype;
    int _malt;
    int _mtoWP;
    int _mrepeat;
    int _mdirCircle;
    int _mtimeCircle;

    QGeoCoordinate _mcoordinate;
};

#endif // SYMBOL_H
