/**
 * ===========================================================
 * Project: FCS-GCS
 * Module: FcsAppSocketApi
 * Module Short Description:
 * Author: Trung Nguyen
 * Date: 08/02/2019
 * Viettel Aerospace Institude - Viettel Group
 * ===========================================================
 */

#include "app_socket_api.hpp"

#define BIND_EVENT(IO, EV, FN) \
    IO->on(EV, FN)

enum class ServerMessage {
    NewUserOnline,
    PcdRequestJoinRoom,
    PcdRequestShareVideo,
    UpdateRoom,
    NewMessage,
    AddToRoomResponse,
    PcdSharingStatus,
    ReloadWebEngineView,
    MediaError
};

const std::map<std::string, ServerMessage> mapMessage = {
    { "new_user_online", ServerMessage::NewUserOnline },
    { "pcd_request_join_room", ServerMessage::PcdRequestJoinRoom },
    { "pcd_request_share_video", ServerMessage::PcdRequestShareVideo },
    { "update_room", ServerMessage::UpdateRoom },
    { "new_message", ServerMessage::NewMessage },
    { "add_to_room_response", ServerMessage::AddToRoomResponse },
    { "pcd_video_sharing_status", ServerMessage::PcdSharingStatus },
    { "reload_web_engine_view", ServerMessage::ReloadWebEngineView },
    { "media_error", ServerMessage::MediaError }
};

AppSocketApi* AppSocketApi::inst = nullptr;

class AppSocketApiImpl : public AppSocketApi {
public:
    AppSocketApiImpl(const QString& serverIp, const int& serverPort, const QString& initStationName = "");
    ~AppSocketApiImpl();
    bool getConnectionStatus();
    void notifyQmlReady();
    void createNewRoom(QString const& rtspLink, QString const& room);
    void shareVideoToRoom();
    void addPcdToRoom(QString const& pcdUid);
    void replyRequestJoinRoom(bool accepted, QString const& pcdUid);
    void requestOpenParticularPcdVideo(QString const& pcdUid);
    void closePcdVideo(QString const& pcdUid);
    void sharePcdVideoToRoom(QString const& pcdUid);
    void stopSharePcdVideoFromRoom(QString const& pcdUid);
    void removePcdFromRoom(QString const& pcdUid);
    void sendMsgToRoom(QString const& msg);
    QString getRoomName();
    QString getServerIp();
    QString getStationName();

protected:
    void listenSeverSignal();
    void onConnected(std::string const& nsp);
    void onClosed(sio::client::close_reason const& reason);
    void onFailed();

private:
    //----Private methods
    void doAuthentication();
    void _createNewRoom();
    void handleListRoomsSignal(sio::event const& ev);
    void handleListActiveUsers(sio::event const& ev);
    void hanldeSpecUserChangeAvailable(sio::event const& ev);
    void handleUserUpdateRole(sio::event const& ev);
    void handleUserUpdateConnectionState(sio::event const& ev);
    void handleServerMessages(sio::event const& ev);
    void handleUserInactive(sio::event const& ev);
    void handleUserLocationUpdate(sio::event const& ev);
    void sendMessage(const std::string& cmd, const std::string& msg);
    void sendMessage(const std::string& msg);
    Json::Value parseData(const std::string& package);

    //---- Properties
    QString localIpAddress;
    QString rtspLink;
    QString room;
    QString serverIp;
    QString stationName;
    bool isConnected;
    bool isQmlReady;
    std::unique_ptr<sio::client> sockInst;
};

using namespace std::placeholders;

AppSocketApiImpl::AppSocketApiImpl(const QString& serverIp, const int& serverPort, const QString& initStationName) :
    serverIp(serverIp),
    stationName(initStationName),
    isConnected(false),
    isQmlReady(false),
    sockInst(new sio::client("https://" + serverIp.toStdString() + ":" + std::to_string(serverPort)))
{
    qDebug("Init socket connection ....");
    sockInst->connect();
    sockInst->set_socket_open_listener(std::bind(&AppSocketApiImpl::onConnected, this, _1));
    sockInst->set_close_listener(std::bind(&AppSocketApiImpl::onClosed, this, _1));
    sockInst->set_fail_listener(std::bind(&AppSocketApiImpl::onFailed, this));
    listenSeverSignal();
}

AppSocketApiImpl::~AppSocketApiImpl() {
    sockInst->socket()->off_all();
    sockInst->socket()->off_error();
}

bool AppSocketApiImpl::getConnectionStatus() {
    return isConnected;
}

void AppSocketApiImpl::notifyQmlReady() {
    isQmlReady = true;
}

void AppSocketApiImpl::createNewRoom(const QString &rtspLink, const QString &room) {
    this->room = room;
    this->rtspLink = rtspLink;
}

void AppSocketApiImpl::shareVideoToRoom() {
    Json::Value send_package;
    send_package["mess"] = "shareVideoToAllPcdsInRoom";
    send_package["data"] = "";
    qDebug("%s", send_package.toStyledString().c_str());
    sendMessage("client_message", send_package.toStyledString());
}

void AppSocketApiImpl::addPcdToRoom(const QString &pcdUid) {
    Json::Value send_package;
    send_package["mess"] = "addPcdToRoom";
    send_package["data"]["pcd_uid"] = pcdUid.toStdString();
    qDebug("%s", send_package.toStyledString().c_str());
    sendMessage(send_package.toStyledString());
}

void AppSocketApiImpl::replyRequestJoinRoom(bool accepted, const QString &pcdUid) {
    Json::Value send_package;
    send_package["mess"] = "replyRequestJoinRoom";
    send_package["data"]["accepted"] = accepted;
    send_package["data"]["pcdUid"] = pcdUid.toStdString();
    qDebug("%s", send_package.toStyledString().c_str());
    sendMessage(send_package.toStyledString());
}

void AppSocketApiImpl::requestOpenParticularPcdVideo(const QString &pcdUid) {
    Json::Value send_package;
    send_package["mess"] = "fcsWantToViewVideoFromSinglePcd";
    send_package["data"]["pcdUid"] = pcdUid.toStdString();
    qDebug("%s", send_package.toStyledString().c_str());
    sendMessage(send_package.toStyledString());
}

void AppSocketApiImpl::closePcdVideo(const QString &pcdUid) {
    Json::Value send_package;
    send_package["mess"] = "fcsCloseVideoShareOfSinglePcd";
    send_package["data"]["pcdUid"] = pcdUid.toStdString();
    qDebug("%s", send_package.toStyledString().c_str());
    sendMessage(send_package.toStyledString());
}

void AppSocketApiImpl::sharePcdVideoToRoom(const QString &pcdUid) {
    Json::Value send_package;
    send_package["mess"] = "sharePcdVideoToRoom";
    send_package["data"]["pcdUid"] = pcdUid.toStdString();
    qDebug("%s", send_package.toStyledString().c_str());
    sendMessage(send_package.toStyledString());
}

void AppSocketApiImpl::stopSharePcdVideoFromRoom(const QString &pcdUid) {
    Json::Value send_package;
    send_package["mess"] = "stopSharePcdVideoFromRoom";
    send_package["data"]["pcdUid"] = pcdUid.toStdString();
    qDebug("%s", send_package.toStyledString().c_str());
    sendMessage(send_package.toStyledString());
}

void AppSocketApiImpl::removePcdFromRoom(QString const& pcdUid) {
    Json::Value send_package;
    send_package["mess"] = "removeUserFromRoom";
    send_package["data"]["pcdUid"] = pcdUid.toStdString();
    send_package["data"]["room"] = this->room.toStdString();
    qDebug("%s", send_package.toStyledString().c_str());
    sendMessage(send_package.toStyledString());
}

void AppSocketApiImpl::sendMsgToRoom(QString const& msg) {
    Json::Value send_package;
    send_package["mess"] = "roomNewMessage";
    send_package["data"]["mess"] = msg.toStdString();
    send_package["data"]["room"] = this->room.toStdString();
    qDebug("%s", send_package.toStyledString().c_str());
    sendMessage(send_package.toStyledString());
}

QString AppSocketApiImpl::getRoomName() {
    return this->room;
}

QString AppSocketApiImpl::getServerIp() {
    return this->serverIp;
}

QString AppSocketApiImpl::getStationName() {
    return this->stationName;
}

void AppSocketApiImpl::listenSeverSignal() {
    sockInst->socket()->on("list_rooms", std::bind(&AppSocketApiImpl::handleListRoomsSignal, this, _1));
    sockInst->socket()->on("list_active_users", std::bind(&AppSocketApiImpl::handleListActiveUsers, this, _1));
    sockInst->socket()->on("user_change_available", std::bind(&AppSocketApiImpl::hanldeSpecUserChangeAvailable, this, _1));
    sockInst->socket()->on("user_update_role", std::bind(&AppSocketApiImpl::handleUserUpdateRole, this, _1));
    sockInst->socket()->on("user_update_connection_state", std::bind(&AppSocketApiImpl::handleUserUpdateConnectionState, this, _1));
    sockInst->socket()->on("user_inactive", std::bind(&AppSocketApiImpl::handleUserInactive, this, _1));
    sockInst->socket()->on("spec_user_location_updated", std::bind(&AppSocketApiImpl::handleUserLocationUpdate, this, _1));
    sockInst->socket()->on("server_message", std::bind(&AppSocketApiImpl::handleServerMessages, this, _1));
}

//void AppSocketApiImpl::handleListRoomsSignal(const std::string &name, const sio::message::ptr &data, bool hasAck, sio::message::ptr &ack_resp) {
//    qDebug("Data received: %s", data->get_string());
//}

void AppSocketApiImpl::doAuthentication() {
    Json::Value send_package;
    send_package["isFcs"] = true;
    send_package["mediaExchange"] = false;
    send_package["fcsStationName"] = this->stationName.toStdString();
    qDebug("%s", send_package.toStyledString().c_str());
    while( !isConnected ) {
        sleep(1);
    }
    sendMessage("authentication", send_package.toStyledString());
}

void AppSocketApiImpl::_createNewRoom() {
    Json::Value send_package;
    send_package["mess"] = "fcsCreateRoom";
    send_package["data"]["rtsp_link"] = this->rtspLink.toStdString();
    send_package["data"]["room"] = this->room.toStdString();
    qDebug("%s", send_package.toStyledString().c_str());
    while( !isConnected ) {
        sleep(1);
    }
    sendMessage("client_message", send_package.toStyledString());
}

void AppSocketApiImpl::handleListRoomsSignal(sio::event const& ev) {
    qDebug("Data received: %s", ev.get_message()->get_string().c_str());
    while( !isQmlReady ) {
        sleep(1);
    }
    Q_EMIT receivedListRoomsSignal(QString::fromStdString(ev.get_message()->get_string()));
}

void AppSocketApiImpl::handleListActiveUsers(sio::event const& ev) {
    qDebug("Data received: %s", ev.get_message()->get_string().c_str());
    while( !isQmlReady ) {
        sleep(1);
    }
    Q_EMIT receivedListActiveUsers(QString::fromStdString(ev.get_message()->get_string()));
}

void AppSocketApiImpl::hanldeSpecUserChangeAvailable(sio::event const& ev) {
    qDebug("Data received: %s", ev.get_message()->get_string().c_str());
    while( !isQmlReady ) {
        sleep(1);
    }
    Q_EMIT specUserChangeAvailable(QString::fromStdString(ev.get_message()->get_string()));
}

void AppSocketApiImpl::handleUserUpdateRole(sio::event const& ev) {
    qDebug("Data received: %s", ev.get_message()->get_string().c_str());
    while( !isQmlReady ) {
        sleep(1);
    }
    Json::Value  jsonDataObj = parseData(ev.get_message()->get_string());
    Q_EMIT userUpdateRole(QString::fromStdString(jsonDataObj["uid"].toStyledString()), jsonDataObj["role"].asInt());
}

void AppSocketApiImpl::handleUserUpdateConnectionState(sio::event const& ev) {
    qDebug("Data received: %s", ev.get_message()->get_string().c_str());
    while( !isQmlReady ) {
        sleep(1);
    }
    Json::Value  jsonDataObj = parseData(ev.get_message()->get_string());
    Q_EMIT userUpdateConnectionState(QString::fromStdString(jsonDataObj["uid"].toStyledString()), jsonDataObj["connection"].asBool());
}

void AppSocketApiImpl::handleUserInactive(sio::event const& ev) {
    qDebug("Data received: %s", ev.get_message()->get_string().c_str());
    while( !isQmlReady ) {
        sleep(1);
    }
    Json::Value data =  parseData(ev.get_message()->get_string());
    Q_EMIT userInactive(QString::fromStdString(data["userUid"].toStyledString()));
}

void AppSocketApiImpl::handleUserLocationUpdate(sio::event const& ev) {
    qDebug("Data received: %s", ev.get_message()->get_string().c_str());
    while( !isQmlReady ) {
        sleep(1);
    }
    Json::Value data =  parseData(ev.get_message()->get_string());
    Q_EMIT specUserLocationUpdated(QString::fromStdString(data["uid"].toStyledString()), data["lat"].asFloat(), data["lng"].asFloat());
}

void AppSocketApiImpl::handleServerMessages(const sio::event &ev) {
    Json::Value jsonDataObj = parseData(ev.get_message()->get_string());
    qDebug("%s - %s", jsonDataObj["mess"].toStyledString().c_str(), jsonDataObj["data"].toStyledString().c_str());
    std::string key =  jsonDataObj["mess"].toStyledString();
    //key.erase(std::remove(key.begin(), key.end(), '\"'), key.end());
    std::regex reg("[\"\n]?");
    try {
        //------------ Handle server_message
        switch( mapMessage.at(std::regex_replace(key, reg, "")) ) {
        //---1. new_message
        case ServerMessage::NewMessage:
            while( !isQmlReady ) {
                sleep(1);
            }
            Q_EMIT newRoomMessage(QString::fromStdString(jsonDataObj["data"]["msg"].toStyledString()),
                    QString::fromStdString(jsonDataObj["data"]["sourcePcdUid"].toStyledString()), QString::fromStdString(jsonDataObj["data"]["sourcePcdName"].toStyledString()));
            break;

        //---2. new_user_online
        case ServerMessage::NewUserOnline:
            while( !isQmlReady ) {
                sleep(1);
            }
            Q_EMIT newUserOnline(QString::fromStdString(jsonDataObj["data"]["uid"].toStyledString()), QString::fromStdString(jsonDataObj["data"]["name"].toStyledString()));
            break;

        //---3. pcd_request_join_room
        case ServerMessage::PcdRequestJoinRoom:
            while( !isQmlReady ) {
                sleep(1);
            }
            Q_EMIT pcdRequestJoinRoom(QString::fromStdString(jsonDataObj["data"]["pcdUid"].toStyledString()));
            break;

        //---4. pcd_request_share_video
        case ServerMessage::PcdRequestShareVideo:
            while( !isQmlReady ) {
                sleep(1);
            }
            Q_EMIT pcdRequestShareVideo(QString::fromStdString(jsonDataObj["data"]["pcdUid"].toStyledString()));
            break;

        //---5. Update room
        case ServerMessage::UpdateRoom:
            while( !isQmlReady ) {
                sleep(1);
            }
            Q_EMIT updateRoom(QString::fromStdString(jsonDataObj["data"].toStyledString()));
            break;

        //---6. Pcd reply request add to room
        case ServerMessage::AddToRoomResponse:
            while( !isQmlReady ) {
                sleep(1);
            }
            //qDebug("Pcd reply: %s", jsonDataObj["data"]["accepted"].type());
            Q_EMIT pcdReplyRequestAddToRoom(QString::fromStdString(jsonDataObj["data"]["roomName"].toStyledString()), QString::fromStdString(jsonDataObj["data"]["pcdUid"].toStyledString()), jsonDataObj["data"]["accepted"].asBool());
            break;

        //---7. Pcd sharing media status
        case ServerMessage::PcdSharingStatus:
            while( !isQmlReady ) {
                sleep(1);
            }
            Q_EMIT pcdSharingVideoStatus(jsonDataObj["data"]["status"].asBool(), QString::fromStdString(jsonDataObj["data"]["pcdUid"].toStyledString()));
            break;

        //---8. Reload webengine view if it still haven't connected to server yet
        case ServerMessage::ReloadWebEngineView:
            while( !isQmlReady ) {
                sleep(1);
            }
            Q_EMIT shouldReloadWebEngineView(jsonDataObj["data"]["redoAction"].asInt(), QString::fromStdString(jsonDataObj["data"]["redoData"].toStyledString()));
            break;
        case ServerMessage::MediaError:
            while( !isQmlReady ) {
                sleep(1);
            }
            Q_EMIT mediaError(QString::fromStdString(jsonDataObj["data"]["sourceUid"].toStyledString()), jsonDataObj["data"]["type"].asInt());
            break;

        default:
            qDebug("Server message not found/invalid !!!");
            break;
        }
    } catch(...) {
        qDebug("Server message not found/invalid !!! It has to be in format like {mess: ..., data: ...}");
    }
}

void AppSocketApiImpl::onConnected(const std::string &nsp) {
    isConnected = true;
    doAuthentication();
    sleep(1);
    _createNewRoom();
    Q_EMIT connectedToServer();
}

void AppSocketApiImpl::onFailed() {
    qDebug("Socket connection on failed !!!!");
}

void AppSocketApiImpl::onClosed(const sio::client::close_reason &reason) {
    qDebug("Socket connection on closed !!!");
    isConnected = false;
    Q_EMIT networkCrash();
}

void AppSocketApiImpl::sendMessage(const std::string &cmd, const std::string &msg) {
    sockInst->socket()->emit_(cmd, msg);
}

void AppSocketApiImpl::sendMessage(const std::string &msg) {
    sockInst->socket()->emit_("client_message", msg);
}

Json::Value AppSocketApiImpl::parseData(const std::string &package) {
    Json::Reader jsonReader;
    Json::Value  jsonDataObj;
    if( !jsonReader.parse(package, jsonDataObj) ) {
        qDebug("Error in parsing json object from std::string - %s", jsonReader.getFormatedErrorMessages().c_str());
    }
    return jsonDataObj;
}
AppSocketApi* AppSocketApi::connectToServer(const QString& serverIp, const int& serverPort, const QString& initStationName) {
    if( inst == nullptr ) {
        inst = (AppSocketApiImpl*) new AppSocketApiImpl(serverIp, serverPort, initStationName);
    }
    return inst;
}
