#ifndef TRACKOBJECTINFO_H
#define TRACKOBJECTINFO_H

#include <QObject>
#include <QSize>
#include <QRect>
class TrackObjectInfo : public QObject
{
    Q_OBJECT
public:
    //---------- Expose struct properties to qml
    Q_PROPERTY(QSize sourceSize READ sourceSize WRITE setSourceSize NOTIFY sourceSizeChanged)
    Q_PROPERTY(QRect rect READ rect WRITE setRect NOTIFY rectChanged)
    Q_PROPERTY(QString userId READ userId WRITE setUserId NOTIFY userIdChanged)
    Q_PROPERTY(int screenX READ screenX WRITE setScreenX NOTIFY screenXChanged)
    Q_PROPERTY(int screenY READ screenY WRITE setScreenY NOTIFY screenYChanged)
    Q_PROPERTY(float speed READ speed WRITE setSpeed NOTIFY speedChanged)
    Q_PROPERTY(float angle READ angle WRITE setAngle NOTIFY angleChanged)
    Q_PROPERTY(float latitude READ latitude WRITE setLatitude NOTIFY latitudeChanged)
    Q_PROPERTY(float longitude READ longitude WRITE setLongitude NOTIFY longitudeChanged)
    Q_PROPERTY(QString name READ name WRITE setName NOTIFY nameChanged)
    Q_PROPERTY(bool isSelected READ isSelected WRITE setIsSelected NOTIFY isSelectedChanged)
    explicit TrackObjectInfo(QObject *parent = nullptr){}
    TrackObjectInfo(
            const QSize &_sourceSize,
            const QRect &_rect,
            const QString &_userId,
            const float &_latitude, const float &_longitude,
            const float &_speed,
            const float &_angle,
            const QString &_name) :
        m_sourceSize(_sourceSize),
        m_rect(_rect),
        m_userId(_userId),
        m_latitude(_latitude),
        m_longitude(_longitude),
        m_speed(_speed),
        m_angle(_angle),
        m_name(_name)
    {}
    ~TrackObjectInfo(){}
    QSize sourceSize(){
        return m_sourceSize;
    }
    void setSourceSize(QSize sourceSize){
        m_sourceSize = sourceSize;
        Q_EMIT sourceSizeChanged();
    }
    QRect rect(){
        return m_rect;
    }
    void setRect(QRect rect){
        m_rect = rect;
        Q_EMIT rectChanged();
    }
    QString userId()
    {
        return m_userId;
    }
    void setUserId(QString userId)
    {
        m_userId = userId;
        Q_EMIT userIdChanged(userId);
    }

    int screenX()
    {
        return m_screenX;
    }
    void setScreenX(int screenX)
    {
        m_screenX = screenX;
        Q_EMIT screenXChanged();
    }
    int screenY()
    {
        return m_screenY;
    }
    void setScreenY(int screenY)
    {
        m_screenY = screenY;
        Q_EMIT screenYChanged();
    }

    float latitude()
    {
        return m_latitude;
    }
    void setLatitude(float latitude)
    {
        m_latitude = latitude;
        Q_EMIT latitudeChanged(latitude);
    }

    float longitude()
    {
        return m_longitude;
    }
    void setLongitude(float longitude)
    {
        m_longitude = longitude;
        Q_EMIT longitudeChanged(longitude);
    }

    float speed()
    {
        return m_speed;
    }
    void setSpeed(float speed)
    {
        if(static_cast<int>(m_speed - speed) != 0){
            m_speed = speed;
            Q_EMIT speedChanged(speed);
        }
    }
    float angle()
    {
        return m_angle;
    }
    void setAngle(float angle)
    {
        if(static_cast<int>(m_angle - angle) != 0){
            m_angle = angle;
            Q_EMIT angleChanged(angle);
        }
    }

    QString name()
    {
        return m_name;
    }
    void setName(QString name)
    {
        m_name = name;
        Q_EMIT nameChanged(name);
    }

    bool isSelected()
    {
        return m_isSelected;
    }
    void setIsSelected(bool isSelected)
    {
        m_isSelected = isSelected;
        Q_EMIT isSelectedChanged();
    }

Q_SIGNALS:
    void userIdChanged(QString userId);
    void screenXChanged();
    void screenYChanged();
    void latitudeChanged(float latitude);
    void longitudeChanged(float longtitude);
    void speedChanged(float speed);
    void angleChanged(float angle);
    void nameChanged(QString name);
    void isSelectedChanged();
    void rectChanged();
    void sourceSizeChanged();
private:
    QSize m_sourceSize;
    QRect m_rect;
    QString m_userId;
    int m_screenX;
    int m_screenY;
    float m_latitude;
    float m_longitude;
    float m_speed;
    float m_angle;
    QString m_name;
    bool m_isSelected = false;
};

#endif // TRACKOBJECTINFO_H
