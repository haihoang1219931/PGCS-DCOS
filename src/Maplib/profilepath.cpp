#include "profilepath.h"


ProfilePath::ProfilePath(QQuickItem *parent)
{

}

void ProfilePath::paint(QPainter *painter)
{
    painter->setRenderHint(QPainter::Antialiasing);
    drawPlot(painter);
}

void ProfilePath::changePen(QPen *pen, QString color, int width)
{
    pen->setColor(QColor(color));
    pen->setWidth(width);
}

void ProfilePath::changeFont(QFont *font, QString fontFamily, int size)
{
    font->setFamily(fontFamily);
    font->setPixelSize(size);
}


void ProfilePath::drawPlot(QPainter *painter)
{
    //get pen,font,size
    QPen pen = painter->pen();
    QFont font = painter->font();
    QSizeF itemSize = size();


    //set background
//    painter->setBrush(QBrush("#b334495e"));
//    pen.setColor(QColor("transparent"));
//    changePen(&pen,"transparent",1);
//    painter->setPen(pen);
//    painter->drawRoundedRect(0, 0, qRound(itemSize.width()),qRound(itemSize.height()), 10, 10);

    changePen(&pen,"white",2);
    painter->setPen(pen);
    //draw title
    //set font
    changeFont(&font,_fontFamily,_fontSize);
    font.setBold(false);
    font.setItalic(false);
    painter->setFont(font);
    painter->drawText(QRectF(0, 10, qRound(itemSize.width()),40),Qt::AlignCenter,_title);

    //draw axisX,axisY line
    QPointF O_point = QPointF(mAxisYoffset,itemSize.height()-mAxisXoffset);
    QPointF X_point = QPointF(itemSize.width()-20,itemSize.height()-mAxisXoffset);
    QPointF Y_point = QPointF(mAxisYoffset,50);
    max_altitude_point_Y=Y_point.ry()-O_point.ry() + 20;
    max_distance_point_X = X_point.rx()-O_point.rx() - 20;

    painter->drawLine(O_point,X_point);
    painter->drawLine(O_point,Y_point);

    changeFont(&font,_fontFamily,_fontSize);
    font.setBold(false);
    font.setItalic(false);
    painter->setFont(font);
    painter->drawText(qRound(Y_point.rx()-25),qRound(Y_point.ry()-10),_yName);//"Do cao(m)"
    painter->drawText(QRectF(O_point.rx(), O_point.ry() + _fontSize * 1.5 , max_distance_point_X,20),Qt::AlignCenter,_xName);//"Khoang cach(m)"

    //draw arrow
    const QPointF arrowpointX[3] = {
            QPointF(mAxisYoffset-4,57),
            QPointF(mAxisYoffset,50),
            QPointF(mAxisYoffset+4,57),
        };
    const QPointF arrowpointY[3] = {
            QPointF(itemSize.width()-27,itemSize.height()-mAxisXoffset+4),
            QPointF(itemSize.width()-20,itemSize.height()-mAxisXoffset),
            QPointF(itemSize.width()-27,itemSize.height()-mAxisXoffset-4),
        };
    painter->setBrush(QBrush(Qt::white));
    painter->drawConvexPolygon(arrowpointX, 3);
    painter->drawConvexPolygon(arrowpointY, 3);

    //translate coordinate
    painter->translate(O_point);  //
    //draw axisY value

    //set font
    changeFont(&font,_fontFamily,_fontSize);
    painter->setFont(font);
    //draw axisY value
    for(int i=1 ;i<5; i++)
    {
        double y = max_altitude_point_Y*i/4;
        QPointF p1=QPointF(0,y);
        QPointF p2=QPointF(4,y);
        QPointF p3=QPointF(max_distance_point_X,y);
        //setpen
        changePen(&pen,"white",1);
        painter->setPen(pen);
        painter->drawLine(p1,p2);
        //draw text
        QString y_text=QString("%1").arg(mMaxAltitude*i/4);
        painter->drawText(QRectF(-_fontSize*4.25,y-_fontSize * 0.75,_fontSize*4,_fontSize),Qt::AlignRight,y_text);
        //draw grid
        changePen(&pen,"gray",1);
        painter->setPen(pen);
        painter->drawLine(p2,p3);
    }
    //draw axisX value
    for(int i=1 ;i<6; i++)
    {
        double x = max_distance_point_X*i/5;
        QPointF p1=QPointF(x,0);
        QPointF p2=QPointF(x,-4);
        QPointF p3=QPointF(x,max_altitude_point_Y);
        //set pen
        changePen(&pen,"white",1);
        painter->setPen(pen);
        painter->drawLine(p1,p2);
        //draw text
        QString x_text=QString("%1").arg(mMaxDistance*i/5);
        painter->drawText(QRectF(x-_fontSize*2,4,_fontSize*4,_fontSize),Qt::AlignCenter,x_text);
        //draw grid
        changePen(&pen,"gray",1);
        painter->setPen(pen);
        painter->drawLine(p2,p3);
    }


    //plot
    //set pen
    changePen(&pen,"#8fffffff",1);
    painter->setPen(pen);
    painter->setBrush(QBrush("#4fffffff"));

    QPointF points[mListAltitude.count()+2] ;
    QMapIterator <int,int> i(mListAltitude);
    int index=1;
    while (i.hasNext()) {
        i.next();
        int distance=i.key();
        int altitude=i.value();
        double px = distance*max_distance_point_X/mMaxDistance;
        double py = altitude*max_altitude_point_Y/mMaxAltitude;
        points[index] = QPointF(px,py);
        index++;
    }

    points[0]=QPointF(points[1].rx(),0);
    points[index]=QPointF(points[index-1].rx(),0);
    painter->drawConvexPolygon(points, index+1);


    //draw UAV line of sight
    if(_isShowLineOfSight){
        double fromPx = _fromDistance*max_distance_point_X/mMaxDistance;
        double fromPy = _fromAltitude*max_altitude_point_Y/mMaxAltitude;
        double toPx   = _toDistance*max_distance_point_X/mMaxDistance;
        double toPy   = _toAltitude*max_altitude_point_Y/mMaxAltitude;
        //set pen
        changePen(&pen,"brown",2);
        painter->setPen(pen);
        painter->drawLine(QPointF(fromPx,fromPy),QPointF(toPx,toPy));
    }

    //end plot

    painter->resetMatrix();
}

void ProfilePath::insertProfilePath(int distance, int altitude)
{
    mListAltitude.insert(distance,altitude);
    if(distance>mMaxDistance)
    {
//        int x=distance/500;
//        mMaxDistance = (x+1)*500;
        mMaxDistance = distance;
    }
    if(altitude>mMaxAltitude)
    {
        int x=altitude/100;
        mMaxAltitude = (x+1)*100;
    }
}

float ProfilePath::getAltitude(QString folder,QGeoCoordinate coord)
{
    float alt=mElevation.getAltitude(folder,coord.latitude(),coord.longitude());
}

void ProfilePath::clearProfilePath()
{
    mListAltitude.clear();
    mMaxAltitude = 0;
    mMaxDistance = 0;
    this->update();
}

QGeoCoordinate ProfilePath::findSlideShowCoord(QGeoCoordinate coord, QGeoCoordinate startcoord, QGeoCoordinate tocoord)
{
    double azimuth = startcoord.azimuthTo(tocoord);
    double azimuth1 = startcoord.azimuthTo(coord);
    double cosAlpha = cos((azimuth - azimuth1) / 57.3);
    double distance = startcoord.distanceTo(coord) * abs(cosAlpha);
    QGeoCoordinate hCoord = cosAlpha>0 ? startcoord.atDistanceAndAzimuth(distance,azimuth) : startcoord.atDistanceAndAzimuth(distance,-azimuth);
    return hCoord;
}



void ProfilePath::addElevation(QGeoCoordinate startcoord, QGeoCoordinate tocoord)
{
    double distance =startcoord.distanceTo(tocoord);
    double azimuth = startcoord.azimuthTo(tocoord);//bearing

    int numOfSegments = qRound(distance)/Elevation::resolution;
    int bufferSize = distance > numOfSegments * Elevation::resolution + 1 ? numOfSegments + 2 : numOfSegments + 1;

    clearProfilePath();
    for (int i = 0; i < bufferSize; i++)
    {
        int distance_i = i * Elevation::resolution;
        QGeoCoordinate coord_i = startcoord.atDistanceAndAzimuth(distance_i,azimuth);
        int alt_i = qRound(getAltitude(_folderPath,coord_i));
        insertProfilePath(distance_i,alt_i);
    }
    this->update();
}

QPoint ProfilePath::convertCoordinatetoXY(QGeoCoordinate coord,QGeoCoordinate startcoord, QGeoCoordinate tocoord)
{

    //int X = distance / Elevation::resolution;
    //int Y =

    QGeoCoordinate hCoord = findSlideShowCoord(coord,startcoord,tocoord);
    double distance =startcoord.distanceTo(hCoord);
    double azimuth = startcoord.azimuthTo(tocoord);
    double azimuth1 = startcoord.azimuthTo(coord);
    double cosAlpha = cos((azimuth - azimuth1) / 57.3);
    double px = cosAlpha>0 ? distance*max_distance_point_X/mMaxDistance : -distance*max_distance_point_X/mMaxDistance;
    double py = coord.altitude()*max_altitude_point_Y/mMaxAltitude;

    double altVehicle = coord.altitude();

//    if(altVehicle > 0.95 * static_cast<double>(mMaxAltitude))
//    {
//        mMaxAltitude = static_cast<int>(altVehicle / 0.77);
//        this->update();
//    }
//    else if(altVehicle < 0.75 * static_cast<double>(mMaxAltitude))
//    {
//        mMaxAltitude = static_cast<int>(altVehicle / 0.93);
//        this->update();
//    }

    if(altVehicle > 0.95 * static_cast<double>(mMaxVehicleAlt))
    {
        mMaxVehicleAlt = static_cast<int>(altVehicle / 0.77);

    }
    else if(altVehicle < 0.75 * static_cast<double>(mMaxVehicleAlt))
    {
        mMaxVehicleAlt = static_cast<int>(altVehicle / 0.93);
    }
    if(mMaxVehicleAlt > mMaxAltitude)
    {
        mMaxAltitude = mMaxVehicleAlt;
        this->update();
        //printf("update profile path\r\n");
    }

    return QPoint(static_cast<int>(px),static_cast<int>(py));
}

void ProfilePath::setLineOfSight(double fromDis, double fromAlt, double toDis, double toAlt)
{
    _fromDistance = fromDis;
    _fromAltitude = fromAlt;
    _toDistance   = toDis;
    _toAltitude   = toAlt;

    if(_fromAltitude>mMaxAltitude)
    {
        int x=_fromAltitude/100;
        mMaxAltitude = (x+1)*100;
    }
    if(_toAltitude>mMaxAltitude)
    {
        int x=_toAltitude/100;
        mMaxAltitude = (x+1)*100;
    }

    this->update();
}



//update warning
bool ProfilePath::checkAltitudeWarning(QGeoCoordinate planePos,QGeoCoordinate posPrediction)
{
    double distance =planePos.distanceTo(posPrediction);
    double azimuth = planePos.azimuthTo(posPrediction);//bearing

    int numOfSegments = qRound(distance)/Elevation::resolution;
    int bufferSize = distance > numOfSegments * Elevation::resolution + 1 ? numOfSegments + 2 : numOfSegments + 1;
    if(bufferSize > 100) return false;
    for (int i = 0; i < bufferSize; i++)
    {
        int distance_i = i * Elevation::resolution;
        QGeoCoordinate coord_i = planePos.atDistanceAndAzimuth(distance_i,azimuth);
        int alt_i = qRound(getAltitude(_folderPath,coord_i));
        double stepPlaneAlt = getAltitudeCoordinate(coord_i,planePos,posPrediction);
        if(stepPlaneAlt <= alt_i) return true;
    }
    return false;
}

QGeoCoordinate ProfilePath::planePosPrediction(QGeoCoordinate planePos, QGeoCoordinate toPos, double planeGroundSpeed, double timeSec)
{
    double deltaAltToPos = toPos.altitude() - planePos.altitude();
    double tanAlpha = deltaAltToPos /  planePos.distanceTo(toPos);

    double azmithAngle = planePos.azimuthTo(toPos);

    //planeGroundSpeed : m/s //distance : m
    double preDistance = planeGroundSpeed * timeSec ;
    double preDeltaAlt = tanAlpha * preDistance;
    double preAltitude = planePos.altitude() + preDeltaAlt;

    QGeoCoordinate preCoord = planePos.atDistanceAndAzimuth(preDistance,azmithAngle);
    preCoord.setAltitude(preAltitude);

    return preCoord;
}

double ProfilePath::getAltitudeCoordinate(QGeoCoordinate getCoord, QGeoCoordinate coord1, QGeoCoordinate coord2)
{
    double deltaAltToPos = coord2.altitude() - coord1.altitude();
    double tanAlpha = deltaAltToPos /  coord1.distanceTo(coord2);

    //double azmithAngle = coord1.azimuthTo(coord2);

    //planeGroundSpeed : m/s //distance : m
    double distance = coord1.distanceTo(getCoord) ;
    double deltaAlt = tanAlpha * distance;
    double altitude = coord1.altitude() + deltaAlt;

    return  altitude;
}


