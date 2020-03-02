#include "georeferencecontext.h"

GeoreferenceContext::GeoreferenceContext()
{
    _latitude = 0.0;
    _longitude = 0.0;
    _elevation = 0.0;
}
GeoreferenceContext::GeoreferenceContext(double latitude, double longitude, double elevation)
{
    this->_latitude = latitude;
    this->_longitude = longitude;
    this->_elevation = elevation;
}
GeoreferenceContext::~GeoreferenceContext(){

}

double GeoreferenceContext::getLatitude()
{
   return _latitude;
}
void GeoreferenceContext::setLatitude(double value)
{
   _latitude = value;
}
double GeoreferenceContext::getLongitude()
{
   return _latitude;
}
void GeoreferenceContext::setLongitude(double value)
{
   _latitude = value;
}
double GeoreferenceContext::getElevation()
{
   return _latitude;
}
void GeoreferenceContext::setElevation(double value)
{
   _latitude = value;
}
