#ifndef UCEVENTLISTENER_HPP
#define UCEVENTLISTENER_HPP

#include <iostream>
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

class UCEventEnums : public QObject {
    Q_OBJECT
    public:
    UCEventEnums() : QObject() {}

    enum class OPEN_PCD_VIDEO_INVALID {
        PCD_VIDEO_DUPLICATE,
        USER_NOT_IN_ROOM
    };
    Q_ENUMS(OPEN_PCD_VIDEO_INVALID)

    static void expose() {
        qmlRegisterUncreatableType<UCEventEnums>("io.qdt.dev", 1, 0, "UCEventEnums", "UCEventEnums");
    }
};

class UCEventListener : public QObject {
    Q_OBJECT
    public:
        static UCEventListener* instance();

        Q_INVOKABLE void invalidOpenPcdVideo(int invalidCase);

        Q_INVOKABLE void pointToPcdFromSidebar(QString pcdUid, bool activeStatus);

    Q_SIGNALS:
        void invalidOpenPcdVideoFired(int invalidCase);

        void userIsPointed(QString pcdUid, bool activeStatus);

    private:
        UCEventListener() : QObject() {}
        ~UCEventListener() {}
        UCEventListener(const UCEventListener& ) = delete;
        UCEventListener& operator = (const UCEventListener& ) = delete;

       static UCEventListener* inst;
};
#endif // UCEVENTLISTENER_HPP
