#include "symbol.h"

symbol::symbol()
{

}

symbol::symbol(const int& id,const int& type,const int& alt,const int& toWP,const int& repeat,const int& dircircle,const int& timecircle,const QGeoCoordinate& coordinate)
{
    _mid  = id;
    _mtype= type;
    _malt = alt;
    _mtoWP = toWP;
    _mrepeat = repeat;
    _mdirCircle = dircircle;
    _mtimeCircle = timecircle;
    _mcoordinate=coordinate;


    Id   =_mid;
    Type =_mtype;
    Alt  = alt;
    ToWP = toWP;
    Repeat = repeat;
    DirCircle = dircircle;
    TimeCircle = timecircle;
    Coordinate=coordinate;

}
