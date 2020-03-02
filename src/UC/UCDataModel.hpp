#ifndef UCDATAMODEL_HPP
#define UCDATAMODEL_HPP
//--------------- Including preloading C++ Libraries
#include <iostream>
#include <sstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <chrono>
#include <condition_variable>
#include <thread>
#include <deque>
#include <unistd.h>
#include <mutex>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <map>
#include <algorithm>
#include <string>
#include <regex>

//---------------- Including Qt lib
#include <QObject>
#include <QDebug>
#include <QMutex>
#include <QWaitCondition>
#include <QThread>
#include <QVariantList>
#include <QVariantMap>
#include <QtQuick>
#include <QQmlApplicationEngine>
#include <QQmlListProperty>

namespace UC{
class User : public QObject
{
        Q_OBJECT
    public:
        //---------- Expose struct properties to qml
        Q_PROPERTY(QString ipAddress READ ipAddress WRITE setIPAddress NOTIFY ipAddressChanged)
        Q_PROPERTY(QString userId READ userId WRITE setUserId NOTIFY userIdChanged)
        Q_PROPERTY(QString roomName READ roomName WRITE setRoomName NOTIFY roomNameChanged)
        Q_PROPERTY(bool available READ available WRITE setAvailable NOTIFY availableChanged)
        Q_PROPERTY(int role READ role WRITE setRole NOTIFY roleChanged)
        Q_PROPERTY(bool shared READ shared WRITE setShared NOTIFY sharedChanged)
        Q_PROPERTY(float latitude READ latitude WRITE setLatitude NOTIFY latitudeChanged)
        Q_PROPERTY(float longitude READ longitude WRITE setLongitude NOTIFY longitudeChanged)
        Q_PROPERTY(QString uid READ uid WRITE setUid NOTIFY uidChanged);
        Q_PROPERTY(QString name READ name WRITE setName NOTIFY nameChanged);
        Q_PROPERTY(bool isSelected READ isSelected WRITE setIsSelected NOTIFY isSelectedChanged);
        Q_PROPERTY(bool connectionState READ connectionState WRITE setConnectionState NOTIFY connectionStateChanged);

        User(const QString& _ipAddress, const QString& _userId,
             const QString& _roomName, const bool& _available,
             const int& _role, const bool& _shared,
             const float& _latitude, const float& _longitude,
             const QString& _uid, const QString& _name, const bool& _connectionState) :

            m_ipAddress(_ipAddress),
            m_userId(_userId),
            m_roomName(_roomName),
            m_available(_available),
            m_role(_role),
            m_shared(_shared),
            m_latitude(_latitude),
            m_longitude(_longitude),
            m_uid(_uid),
            m_name(_name),
            m_connectionState(_connectionState)
        {}
        bool connectionState(){ return m_connectionState;}
        void setConnectionState(bool connectionState){
            m_connectionState = connectionState;
            Q_EMIT connectionStateChanged();
        }
        QString ipAddress(){return m_ipAddress;}
        void setIPAddress(QString ipAddress){
            m_ipAddress = ipAddress;
            Q_EMIT ipAddressChanged(ipAddress);
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

        QString roomName()
        {
            return m_roomName;
        }
        void setRoomName(QString roomName)
        {
            m_roomName = roomName;
            Q_EMIT roomNameChanged(roomName);
        }

        bool available()
        {
            return m_available;
        }
        void setAvailable(bool available)
        {
            m_available = available;
            Q_EMIT availableChanged(available);
        }

        int role()
        {
            return m_role;
        }
        void setRole(int role)
        {
            m_role = role;
            Q_EMIT roleChanged(role);
        }

        bool shared()
        {
            return m_shared;
        }
        void setShared(bool shared)
        {
            m_shared = shared;
            Q_EMIT sharedChanged(shared);
        }

        float latitude(){return m_latitude;}
        void setLatitude(float latitude){
            printf("%s = %f\r\n",__func__,latitude);
            m_latitude = latitude;
//            printf("setLatitude = %f\r\n", m_latitude);
            Q_EMIT latitudeChanged(latitude);
        }

        float longitude(){return m_longitude;}
        void setLongitude(float longitude){
            printf("%s = %f\r\n",__func__,longitude);
            m_longitude = longitude;
            printf("setLongitude = %f\r\n", m_longitude);
            Q_EMIT longitudeChanged(longitude);
        }

        QString uid()
        {
            return m_uid;
        }
        void setUid(QString uid)
        {
            m_uid = uid;
            Q_EMIT uidChanged(uid);
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

        bool isSelected() {return m_isSelected;}
        void setIsSelected(bool isSelected){
            printf("%s[%s] = %s\r\n",__func__,m_uid.toStdString().c_str(),isSelected?"true":"false");
            m_isSelected = isSelected;
            Q_EMIT isSelectedChanged();
        }

    Q_SIGNALS:
        void ipAddressChanged(QString ipAddress);
        void userIdChanged(QString userId);
        void roomNameChanged(QString roomName);
        void availableChanged(bool available);
        void roleChanged(int role);
        void sharedChanged(bool shared);
        void latitudeChanged(float latitude);
        void longitudeChanged(float longtitude);
        void uidChanged(QString uid);
        void nameChanged(QString name);
        void isSelectedChanged();
        void connectionStateChanged();
    private:
        QString m_ipAddress;
        QString m_userId;
        QString m_roomName;
        bool m_available;
        int m_role;
        bool m_shared;
        float m_latitude;
        float m_longitude;
        QString m_uid;
        QString m_name;
        bool m_isSelected = false;
        bool m_connectionState = true;
};

class Room : public QObject
{
        Q_OBJECT
    public:
        //---------- Expose struct properties to qml
        Q_PROPERTY(QString roomName READ roomName WRITE setRoomName NOTIFY roomNameChanged)
        Q_PROPERTY(int onlineUsers READ onlineUsers WRITE setOnlineUsers NOTIFY onlineUsersChanged)

        Room(const QString &_roomName, const int &_onlineUsers) :
            m_roomName(_roomName),
            m_onlineUsers(_onlineUsers)
        {}


        QString roomName()
        {
            return m_roomName;
        }
        void setRoomName(QString roomName)
        {
            m_roomName = roomName;
            Q_EMIT roomNameChanged(roomName);
        }

        int onlineUsers()
        {
            return m_onlineUsers;
        }
        void setOnlineUsers(int onlineUsers)
        {
            m_onlineUsers = onlineUsers;
            Q_EMIT onlineUsersChanged(onlineUsers);
        }

    Q_SIGNALS:
        void roomNameChanged(QString roomName);
        void onlineUsersChanged(int onlineUsers);

    private:
        QString m_roomName;
        int m_onlineUsers;
};

class UserAttribute : public QObject {
    Q_OBJECT
public:
    UserAttribute(QObject* parent=0) {}
    enum class Attribute
    {
        IP_ADDRESS = 0,
        USER_ID = 1,
        ROOM_NAME = 2,
        AVAILABLE = 3,
        ROLE = 4,
        SHARED = 5,
        LATITUDE = 6,
        LONGITUDE = 7,
        UID = 8,
        NAME = 9,
        SELECTED = 10,
        CONNECTION_STATE = 11
    };
    Q_ENUMS(Attribute)

    static void expose()
    {
        qmlRegisterType<UC::User>();
        qmlRegisterType<UC::Room>();
        qmlRegisterUncreatableType<UC::UserAttribute>("io.qdt.dev", 1, 0, "UserAttribute", "UserAttribute");
    }
};

class UserRoles : public QObject
{
        Q_OBJECT
    public:
        UserRoles(QObject *parent = 0) {}
        enum class Roles {
            FCS = 1,
            PCD = 2
        };
        Q_ENUMS(Roles)

        static void expose()
        {
            qmlRegisterUncreatableType<UserRoles>("io.qdt.dev", 1, 0, "UserRoles", "UserRoles");
        }
};

class RedoActionAfterReloadWebView : public QObject
{
        Q_OBJECT
    public:
        RedoActionAfterReloadWebView(QObject *parent = 0) {}
        enum class RedoAction {
            ADD_PCD_TO_ROOM = 1,
            OPEN_SINGLE_PCD_VIDEO = 2
        };
        Q_ENUMS(RedoAction)

        static void expose()
        {
            qmlRegisterUncreatableType<RedoActionAfterReloadWebView>("io.qdt.dev", 1, 0, "RedoActionAfterReloadWebView", "RedoActionAfterReloadWebView");
        }
};

class UCDataModel : public QObject
{
        Q_OBJECT
    public:
        Q_PROPERTY( QQmlListProperty<UC::User> listUsers READ listUsers NOTIFY listUsersChanged);
        Q_PROPERTY( QQmlListProperty<UC::Room> listRooms READ listRooms NOTIFY listRoomsChanged);
        Q_PROPERTY(int  selectedID READ selectedID WRITE setSelectedID NOTIFY selectedIDChanged);
        Q_INVOKABLE void addUser(const QVariantMap& userDataObj);
        Q_INVOKABLE bool isUserExist(const QString& name);
        Q_INVOKABLE void removeUser(const int& sequence);
        Q_INVOKABLE void removeUser(const QString& userUid);
        Q_INVOKABLE void updateUser(const QString& userUid, const int& attr, const QVariant& newValue);
        Q_INVOKABLE void clean();
        Q_INVOKABLE int  isContainRoom(QString roomName);
        Q_INVOKABLE void addRoom(const QVariantMap& roomDataObj);
        Q_INVOKABLE void removeRoom(const int& sequence);
        Q_INVOKABLE void removeRoom(const QString& roomName);
        Q_INVOKABLE void updateRoom(const QString& roomName, const int& onlineUsers);
        Q_INVOKABLE void newUserJoinedRoom(const QString& roomName);
        Q_INVOKABLE void userLeftRoom(const QString& roomName);
        Q_INVOKABLE void cleanRoom();

        static UCDataModel *instance();
        QQmlListProperty<UC::User> listUsers()
        {
            return QQmlListProperty<UC::User>(this, _listUsers);
        }
        QQmlListProperty<UC::Room> listRooms()
        {
            return QQmlListProperty<UC::Room>(this, _listRooms);
        }
        int selectedID(){
            return m_selectedID;
        }
        void setSelectedID(int selectedID){ m_selectedID = selectedID; Q_EMIT selectedIDChanged(selectedID);}
    Q_SIGNALS:
        void listUsersChanged();
        void listRoomsChanged();
        void selectedIDChanged(int selectedID);
    protected:
        UCDataModel(QObject *parent) : QObject(parent) {}
        ~UCDataModel() {}
        UCDataModel(const UCDataModel &) = delete;
        UCDataModel &operator = (const UCDataModel &) = delete;

    private:
        QList<UC::User*> _listUsers;
        QList<UC::Room*> _listRooms;
        static UCDataModel* inst;
        int m_selectedID = -1;
};
};
#endif // UCDATAMODEL_HPP
