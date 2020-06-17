#include "symbol.h"

symbol::symbol()
{

}

symbol::symbol(const int &id, const int &type, const int &param1, const int &param2, const int &param3, const int &param4,const QString &text, const QGeoCoordinate &coordinate)
{
    _mid  = id;
    _mtype= type;
    _mparam1 = param1;
    _mparam2 = param2;
    _mparam3 = param3;
    _mparam4 = param4;
    _mtext    = text;
    _mcoordinate=coordinate;

    Id     =_mid;
    Type   =_mtype;
    Param1 = _mparam1;
    Param2 = _mparam2;
    Param3 = _mparam3;
    Param4 = _mparam4;
    Text   = _mtext;
    Coordinate=_mcoordinate;
}



