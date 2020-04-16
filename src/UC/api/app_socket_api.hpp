/**
 * =======================================================================================================================
 * Project: FCS-GCS
 * Module: FcsAppSocketApi
 * Module Short Description: FcsAppSocketApi is module that belongs to FCS-PGCS system. With main responsibility is
 *                           providing apis that help FCS could communicate with DCOS webserver include sending requests,
 *                           receiving data from server to expose, notify to user.
 *
 * Author: Trung Ng
 * Date: 08/02/2019
 * Viettel Aerospace Institude - Viettel Group
 * =========================================================================================================================
 */

#ifndef APP_SOCKET_API_HPP
#define APP_SOCKET_API_HPP

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

//--------------- Socket
#include <sioclient/src/sio_client.h>

//--------------- Json
#include "../json/json.h"

class AppSocketApi : public QObject {
    Q_OBJECT
    public:
        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        //          List apis for module initialization: connect to server, check connection, notify qml ready.
        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        /**
         * @brief connectToServer: This function help init a module instance to call other apis, connect socket to DCOS server.
         * @param serverIp - Server ip address
         * @param serverPort - Server port
         * @param initStationName (optional) - FCS station name to identify joined session. To help other participants
         *                                     in DCOS system can recognize who are you, try to make it as descriptive as posible
         * @return AppSocketApi object
         */
        static AppSocketApi* connectToServer(const QString& serverIp, const int& serverPort, const QString& initStationName = "");

        /**
         * @brief getConnectionStatus: Get the connection status: on/off
         * @return true/false
         */
        Q_INVOKABLE virtual bool getConnectionStatus() = 0;

        /**
         * @brief notifyQmlReady: Notify to back-end api that the front-end interface is already loaded. Ready to do other tasks
         */
        Q_INVOKABLE virtual void notifyQmlReady() = 0;

    Q_SIGNALS:
        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        //          List signals that listened from server. It means the data flow is from server -> you.
        //          Note: The params in functions are the data/messages we get from server. Those one will be used to display
        //                on interface.
        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        /**
         * @brief connectedToServer: This signal inform that module successfully connected to server
         */
        void connectedToServer();

        /**
         * @brief receivedListRoomsSignal: This signal is fired when server send to us list of all the available rooms.
         * @param listRoomMsg: Json object contains list of room with format:
         *                            listRoomMsg = [{roomName: String, participants: number}, {}, {},...]
         */
        void receivedListRoomsSignal(const QString& listRoomMsg);

        /**
         * @brief receivedListActiveUsers: This signal is fired when server send to us list of all authenticated participants
         *                                 joined to system.
         * @param listUsers: Json object contains list of authenticated users with format:
         *                            listUsers = [  {
         *                                              ip_address: String,
         *                                              user_id: String,
         *                                              room_name: string,
         *                                              available: boolean,
         *                                              role: Enum,
         *                                              shared: boolean,
         *                                              uid: String,
         *                                              name: String
         *                                            },
         *                                            {...},
         *                                            {...} ]
         */
        void receivedListActiveUsers(const QString& listUsers);

        /**
         * @brief specUserLocationUpdated: This signal fired when an user changed his location position on map.
         * @param userUid - user's uid that changed position
         * @param lat - Latitude position
         * @param lng - Longitude position
         */
        void specUserLocationUpdated(const QString& userUid, const float& lat, const float& lng);

        /**
         * @brief newRoomMessage: This signal fired when an user in system drop a message to chat room.
         * @param msg - String: The message is touched.
         * @param sourceUserUid - String: The uid of user that creates this message.
         * @param sourceUserName - String: The name of user that creates this message
         */
        void newRoomMessage(const QString& msg, const QString& sourceUserUid, const QString& sourceUserName);

        /**
         * @brief newUserOnline: This signal fired when an user just signed in to system.
         * @param newUserUid - String: The new user's uid
         * @param newUserName - String: The new user's name
         */
        void newUserOnline(const QString& newUserUid, const QString& newUserName);

        /**
         * @brief pcdRequestJoinRoom: This signal fired when a pcd want/request to join the room
         *                            that you created/owned.
         * @param pcdUid - String: requested pcd's uid
         */
        void pcdRequestJoinRoom(const QString& pcdUid);

        /**
         * @brief pcdRequestShareVideo: This signal fired when a pcd in your created room want to
         *                              alert you in some emergency case with purpose that you
         *                              will open the video to see the sight.
         * @param pcdUid - The uid of user that wanna share video.
         */
        void pcdRequestShareVideo(const QString& pcdUid);

        /**
         * @brief pcdReplyRequestAddToRoom: This signal is fired after pcd responses the request that you
         *                                  added this one to room.
         * @param room - String: The room you created.
         * @param pcdUid - String: The pcd's uid you requested adding to room.
         * @param accepted - Boolean: Pcd accepted join room or not.
         */
        void pcdReplyRequestAddToRoom(const QString& room, const QString& pcdUid, const bool& accepted);

        /**
         * @brief updateRoom: This signal fired when something in room changed like participant join or leave.
         * @param dataObjectStr: The JSON data object with format:
         *                       {
         *                          roomInfo: {roomName: string, participants: number},
         *                          listParticipants: Array of participants,
         *                          action: string (join/leave),
         *                          participant: object -  The participant that cause the room change.
         *                       }
         */
        void updateRoom(const QString& dataObjectStr);

        /**
         * @brief specUserChangeAvailable: This signal fired when someone change their available state
         *                                 If available state is false, you must not modify or send request to
         *                                 this user and otherwise
         * @param userAttrObj - JSON object contains user attributes with format:
         *                      {
         *                         ip_address: String,
         *                         user_id: String,
         *                         room_name: string,
         *                         available: boolean,
         *                         role: Enum,
         *                         shared: boolean,
         *                         uid: String,
         *                         name: String
         *                      }
         */
        void specUserChangeAvailable(const QString& userAttrObj);

        /**
         * @brief userUpdateRole: This signal fired when someone is updated their role.
         * @param userUid - The user that has role updated.
         * @param role - int: Enum integer: 1- FCS, 2-PCD
         */
        void userUpdateRole(const QString &userUid, const int& role);

        /**
         * @brief userUpdateConnectionState: This signal fired when someone is updated connection (incase ping timeout or not - network connection stability)
         * @param userUid
         * @param connectionState
         */
        void userUpdateConnectionState(const QString &userUid, const bool& connectionState);

        /**
         * @brief userInactive: This signal is fired when someone is losed connection
         *                      or turn off connection.
         * @param userUid - The user that leaved
         */
        void userInactive(const QString &userUid);

        /**
         * @brief pcdSharingVideoStatus: This signal fired when you receive the result
         *                               of action share pcd video to room
         * @param status - Boolean : Success or not
         * @param pcdUid - Pcd that is shared.
         */
        void pcdSharingVideoStatus(const bool& status, const QString& pcdUid);

        /**
         * @brief shouldReloadWebEngineView: This signal fired when server realized that the connection of web engine view
         *                                   client (web engine view is an embeded browser is attact to QML interface with
         *                                   purpurse receive and display video/audio media) hasn't established yet.
         *                                   May be the server start after web engine client connect to, so it should be
         *                                   refreshed.
         *
         * @param redoAction - The action that just sent to server. And when server handle this request, it cause error and request
         *                     reload web engine view. We should resend this action after web engine view already reloaded.
         * @param redoData - The data of action just sent. It support for resending work.
         */
        void shouldReloadWebEngineView(const int& redoAction, QString const& redoData);

        /**
         * @brief networkCrash
         */
        void networkCrash();

        /**
         * @brief mediaError
         * @param sourceUid - Uid of user who has some error while accessing media.
         */
        void mediaError(const QString& sourceUid, const int& errorType);
    public Q_SLOTS:
        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        //          List signals that you use to send request to server. The data flow is from you -> server in reverse with
        //          list signals above.
        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        /**
         * @brief createNewRoom: The request to create your own new conference room
         * @param rtspLink - The sharing quard video link.
         * @param room - Room name
         */
        virtual void createNewRoom(QString const& rtspLink, QString const& room) = 0;

        /**
         * @brief shareVideoToRoom: Share the video received from UAV/quard to all participants in room.
         */
        virtual void shareVideoToRoom() = 0;

        /**
         * @brief addPcdToRoom: After room created, you can add member into room from list of all authenticated users.
         * @param pcdUid - The user's uid you wanna add.
         */
        virtual void addPcdToRoom(QString const& pcdUid) = 0;

        /**
         * @brief replyRequestJoinRoom: When an user send a join room request, you can accept or not.
         * @param accepted - You decision, accepted or not.
         * @param pcdUid - The pcd that sent join room request to you.
         */
        virtual void replyRequestJoinRoom(bool accepted, QString const& pcdUid) = 0;

        /**
         * @brief requestOpenParticularPcdVideo: You open the particular pcd 's video to see what 's going on at
         *                                       this one side.
         * @param pcdUid - The pcd that you opened video to see.
         */
        virtual void requestOpenParticularPcdVideo(QString const& pcdUid) = 0;

        /**
         * @brief closePcdVideo: You can close the pcd video after open it.
         * @param pcdUid - The pcd you close video.
         */
        virtual void closePcdVideo(QString const& pcdUid) = 0;

        /**
         * @brief sharePcdVideoToRoom: After you open a particiular pcd's video, you can share it to all members in room.
         * @param pcdUid  - Pcd that you share video.
         */
        virtual void sharePcdVideoToRoom(QString const& pcdUid) = 0;

        /**
         * @brief stopSharePcdVideoFromRoom: Stop sharing a particular pcd's video from room.
         * @param pcdUid - The pcd that you stop share video.
         */
        virtual void stopSharePcdVideoFromRoom(QString const& pcdUid) = 0;

        /**
         * @brief removePcdFromRoom: Kick out an user away from room.
         * @param pcdUid - pcd has been baned.
         */
        virtual void removePcdFromRoom(QString const& pcdUid) = 0;

        /**
         * @brief sendMsgToRoom: Drop a message to room.
         * @param msg - Message you sent.
         */
        virtual void sendMsgToRoom(QString const& msg) = 0;

        /**
         * @brief getRoomName: Get the room name
         * @return
         */
        virtual QString getRoomName() = 0;

        /**
         * @brief getServerIp: Get server ip address
         * @return
         */
        virtual QString getServerIp() = 0;

        /**
         * @brief getStationName: Get name of fcs station
         * @return
         */
        virtual QString getStationName() = 0;
    //############################################ Prevented methods
    protected:
        explicit AppSocketApi(QObject* parent = 0) : QObject(parent){}
        virtual ~AppSocketApi() {}

        //--- Prevent copy contructor and assignment operator
        AppSocketApi( const AppSocketApi& ) = delete;
        AppSocketApi& operator = (const AppSocketApi& ) = delete;
    private:
        static AppSocketApi* inst;

};

#endif // APP_SOCKET_API_HPP
