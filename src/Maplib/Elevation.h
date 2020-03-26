#ifndef ELEVATION_H
#define ELEVATION_H

#include <QObject>
#include <QUrl>
#include <iostream>
#include <fstream>
#include <math.h>
using namespace std;


class Elevation : public QObject
{
    Q_OBJECT
    Q_PROPERTY(QString path READ path WRITE setPath)
public:
    explicit Elevation(QObject *parent = nullptr);
public:
    string m_path;
    static const int resolution = 90;//m
public:
    QString path(){
        return QString::fromStdString(m_path);
    }
    void setPath(QString _path){
        m_path = _path.toStdString();
    }
private:
    int secondsPerPx = 3;  //arc seconds per pixel (3 equals cca 90m)
    int totalPx = 1201;
public:
    /** Prepares corresponding file if not opened */
    bool srtmLoadTile(const char *folder, int latDec, int lonDec, unsigned char * srtmTile){
        char filename[256];
        sprintf(filename, "%s/%c%02d%c%03d.hgt", folder,
                    latDec>0?'N':'S', abs(latDec),
                    lonDec>0?'E':'W', abs(lonDec));

//        printf("Opening %s\n", filename);
        FILE* srtmFd = fopen(filename, "r");
        if(srtmFd == NULL) {
//            printf("Error opening %s\n",  filename);
            return false;
        }else{
            fread(srtmTile, 1, (2 * totalPx * totalPx), srtmFd);
            fclose(srtmFd);
            return true;
        }
    }

    void srtmReadPx(int y, int x, int* height,unsigned char * srtmTile){
        int row = (totalPx-1) - y;
        int col = x;
        int pos = (row * totalPx + col) * 2;

        //set correct buff pointer
        unsigned char * buff = & srtmTile[pos];

        //solve endianity (using int16_t)
        int16_t hgt = 0 | (buff[0] << 8) | (buff[1] << 0);

        *height = (int) hgt;
    }

    /** Returns interpolated height from four nearest points */
    Q_INVOKABLE float getAltitude(QString folder,float lat, float lon){
//        printf("Get altitude from %f,%f\r\n",lat,lon);
        int latDec = (int)floor(lat);
        int lonDec = (int)floor(lon);

        float secondsLat = (lat-latDec) * 60 * 60;
        float secondsLon = (lon-lonDec) * 60 * 60;
        unsigned char*srtmTile = new unsigned char[2 * totalPx * totalPx];
        if(srtmLoadTile(folder.toStdString().c_str(),latDec, lonDec,srtmTile) == false){
            delete srtmTile;
            return -1;
        }

        //X coresponds to x/y values,
        //everything easter/norhter (< S) is rounded to X.
        //
        //  y   ^
        //  3   |       |   S
        //      +-------+-------
        //  0   |   X   |
        //      +-------+-------->
        // (sec)    0        3   x  (lon)

        //both values are 0-1199 (1200 reserved for interpolating)
        int y = secondsLat/secondsPerPx;
        int x = secondsLon/secondsPerPx;

        //get norther and easter points
        int height[4];
        srtmReadPx(y,   x, &height[2],srtmTile);
        srtmReadPx(y+1, x, &height[0],srtmTile);
        srtmReadPx(y,   x+1, &height[3],srtmTile);
        srtmReadPx(y+1, x+1, &height[1],srtmTile);

        //ratio where X lays
        float dy = fmod(secondsLat, secondsPerPx) / secondsPerPx;
        float dx = fmod(secondsLon, secondsPerPx) / secondsPerPx;

        // Bilinear interpolation
        // h0------------h1
        // |
        // |--dx-- .
        // |       |
        // |      dy
        // |       |
        // h2------------h3
        delete srtmTile;
        return  height[0] * dy * (1 - dx) +
                height[1] * dy * (dx) +
                height[2] * (1 - dy) * (1 - dx) +
                height[3] * (1 - dy) * dx;
    }

Q_SIGNALS:

public Q_SLOTS:
};

#endif // ELEVATION_H
