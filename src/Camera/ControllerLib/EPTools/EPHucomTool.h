#ifndef EPHUCOMTOOL_H
#define EPHUCOMTOOL_H

#include "EPSensorTool.h"

class EPHucomTool : public EPSensorTool
{
    public:
        EPHucomTool();
        virtual ~EPHucomTool();
        double zoomPos2Fov(uint16_t _zoomPos);
        double zoomPos2ZoomRatio(uint16_t _zoomPos);
        double zoomPos2Focal(uint16_t _zoomPos);
        std::vector<unsigned char> zoomPos2DataSend(uint16_t _zoomPos);
        uint16_t dataSend2ZoomPos(std::vector<unsigned char> _dataSend);

    private:
        static const uint16_t DATA_LENGTH;
        static const uint16_t MAX_ZOOM_POS;
        static const uint16_t MIN_ZOOM_POS;

        static const double WIDTH;
        static const double A1;
        static const double A2;
        static const double A3;
        static const double F1;

};
#endif // EPHUCOMTOOL_H
