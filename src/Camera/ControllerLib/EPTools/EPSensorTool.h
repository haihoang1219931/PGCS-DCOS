#ifndef EPSENSORTOOL_H
#define EPSENSORTOOL_H

#include <iostream>
#include <math.h>
#include <vector>

class EPSensorTool
{
    public:
        EPSensorTool() {}
        virtual double zoomPos2Fov(uint16_t _zoomPos) = 0;
        virtual double zoomPos2ZoomRatio(uint16_t _zoomPos) = 0;
        virtual double zoomPos2Focal(uint16_t _zoomPos) = 0;
        virtual std::vector<unsigned char> zoomPos2DataSend(uint16_t _zoomPos) = 0;
        virtual uint16_t dataSend2ZoomPos(std::vector<unsigned char> _dataSend) = 0;
};

#endif // EPSENSORTOOL_H
