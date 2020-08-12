#ifndef GEOREFERENCECONTEXT_H
#define GEOREFERENCECONTEXT_H

#include <stdio.h>
#include <iostream>
using namespace std;

class GeoreferenceContext{
public:
    GeoreferenceContext();
    GeoreferenceContext(double latitude, double longitude, double elevation);
    ~GeoreferenceContext();
    double _latitude;
    double _longitude;
    double _elevation;
    double getLatitude();
    void setLatitude(double value);
    double getLongitude();
    void setLongitude(double value);
    double getElevation();
    void setElevation(double value);
};

#endif // GEOREFERENCECONTEXT_H
