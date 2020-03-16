#include "EPHucomTool.h"

uint16_t const EPHucomTool::DATA_LENGTH = 4U;
uint16_t const EPHucomTool::MAX_ZOOM_POS = 0x7AC0;
uint16_t const EPHucomTool::MIN_ZOOM_POS = 0U;

double const EPHucomTool::WIDTH = 1920.0;
double const EPHucomTool::A1 = 0.9912;
double const EPHucomTool::A2 = -0.10132 / 1024.0;
double const EPHucomTool::A3 = 0.002688 / 1048576.0;
double const EPHucomTool::F1 = 1937.046005;

EPHucomTool::EPHucomTool()
{
}

EPHucomTool::~EPHucomTool()
{
}

double EPHucomTool::zoomPos2Fov(uint16_t _zoomPos)
{
    double _fov;

    if (_zoomPos > MAX_ZOOM_POS) {
        _zoomPos = MAX_ZOOM_POS;
    }

    _fov = atan(this->WIDTH / (2.0 * this->zoomPos2Focal(_zoomPos))) * 2.0;
    return _fov;
}

double EPHucomTool::zoomPos2ZoomRatio(uint16_t _zoomPos)
{
    double _zoomRatio;

    if (_zoomPos > MAX_ZOOM_POS) {
        _zoomPos = MAX_ZOOM_POS;
    }

    _zoomRatio = zoomPos2Focal(_zoomPos) / this->F1;
    return _zoomRatio;
}

double EPHucomTool::zoomPos2Focal(uint16_t _zoomPos)
{
    double _focalLength;

    if (_zoomPos > MAX_ZOOM_POS) {
        _zoomPos = MAX_ZOOM_POS;
    }

    _focalLength = this->WIDTH / (this->A3 * _zoomPos * _zoomPos + this->A2 * _zoomPos + this->A1);
    return _focalLength;
}

std::vector<unsigned char> EPHucomTool::zoomPos2DataSend(uint16_t _zoomPos)
{
    std::vector<unsigned char> _res;
    _res.push_back(0x000F & (_zoomPos >> 12));
    _res.push_back(0x000F & (_zoomPos >> 8));
    _res.push_back(0x000F & (_zoomPos >> 4));
    _res.push_back(0x000F & (_zoomPos));
    return  _res;
}

uint16_t EPHucomTool::dataSend2ZoomPos(std::vector<unsigned char> _dataSend)
{
    uint16_t zoomPos = 0;

    if (_dataSend.size() != DATA_LENGTH) {
        return 0;
    }

    for (int i = 0; i < DATA_LENGTH; i++) {
        zoomPos += _dataSend.at(i) * pow(16, DATA_LENGTH - i - 1);
    }

    return zoomPos;
}
