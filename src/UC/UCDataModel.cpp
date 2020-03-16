#include "UCDataModel.hpp"

namespace UC{
UCDataModel *UCDataModel::inst = nullptr;

void UCDataModel::addUser(const QVariantMap &userDataObj)
{
    //--- 1. Check user data object valid
    // [TODO ]
    //--- 2. Create new user and add to list if passed.
    this->_listUsers.append(new User(userDataObj["ipAddress"].value<QString>() ,
                                userDataObj["userId"].value<QString>(), userDataObj["roomName"].value<QString>(),
                                userDataObj["available"].value<bool>(), userDataObj["role"].value<int>(),
                                userDataObj["shared"].value<bool>(),  userDataObj["latitude"].value<float>(),
                                userDataObj["longitude"].value<float>(), userDataObj["uid"].value<QString>(),
                                userDataObj["name"].value<QString>(),
                                userDataObj["connectionState"].value<bool>()
            ));

    Q_EMIT listUsersChanged();
}
bool UCDataModel::isUserExist(const QString& name){
    bool result = false;
    for(int i = 0; i < this->_listUsers.size(); i++ ){
        User* user = this->_listUsers[i];
        if(name.contains(user->name())){
            result = true;
            break;
        }
    }
    return result;
}
void UCDataModel::removeUser(const int& sequence) {
    if(sequence < 0 || sequence >= this->_listUsers.size()){
        return;
    }

    // remove user on list
    this->_listUsers.removeAt(sequence);
    Q_EMIT listUsersChanged();
}

void UCDataModel::removeUser(const QString &userUid)
{
    // check room contain user
    int sequence = -1;

    for (int i = 0; i < this->_listUsers.size(); i++) {
        if (this->_listUsers[i]->uid() == userUid) {
            sequence = i;
            break;
        }
    }

    removeUser(sequence);
    Q_EMIT listUsersChanged();
}

void UCDataModel::updateUser(const QString& userUid, const int& attr, const QVariant& newValue) {

    for(int i = 0; i < this->_listUsers.size(); i++ ) {
        User* user = this->_listUsers[i];
        if(userUid.contains(this->_listUsers.at(i)->uid())) {
            switch ((UserAttribute::Attribute) attr) {
                case UserAttribute::Attribute::IP_ADDRESS:
                    user->setIPAddress(newValue.value<QString>());
                    break;
                case UserAttribute::Attribute::USER_ID:
                    user->setUserId(newValue.value<QString>());
                    break;
                case UserAttribute::Attribute::ROOM_NAME:
                    user->setRoomName(newValue.value<QString>());
                    break;
                case UserAttribute::Attribute::AVAILABLE:
                    user->setAvailable(newValue.value<bool>());
                    break;
                case UserAttribute::Attribute::ROLE:
                    user->setRole(newValue.value<int>());
                    break;
                case UserAttribute::Attribute::SHARED:
                    user->setShared(newValue.value<bool>());
                    break;
                case UserAttribute::Attribute::LATITUDE:
                    user->setLatitude(newValue.value<float>());
                    break;
                case UserAttribute::Attribute::LONGITUDE:
                    user->setLongitude(newValue.value<float>());
                    break;
                case UserAttribute::Attribute::UID:
                    user->setUid(newValue.value<QString>());
                    break;
                case UserAttribute::Attribute::NAME:
                    user->setName(newValue.value<QString>());
                    break;
                case UserAttribute::Attribute::SELECTED:
                    user->setIsSelected(newValue.value<bool>());
                    this->setSelectedID(i);
                    break;
                case UserAttribute::Attribute::CONNECTION_STATE:
                    user->setConnectionState(newValue.value<bool>());
                    break;
            }
        }else{
            if((UserAttribute::Attribute) attr ==
                    UserAttribute::Attribute::SELECTED){
                user->setIsSelected(false);
            }
        }
    }
//    Q_EMIT listUsersChanged();
}

void UCDataModel::clean()
{
    this->_listUsers.clear();
}
int  UCDataModel::isContainRoom(QString roomName)
{
    int result = -1;

    for (int i = 0; i < this->_listRooms.size(); i++) {
        if (this->_listRooms[i]->roomName() == roomName) {
            result = i;
            break;
        }
    }

    return result;
}
void UCDataModel::addRoom(const QVariantMap &roomDataObj)
{
    //    if(isContainRoom(roomDataObj["roomName"].value<QString>()) < 0)
    {
        this->_listRooms.append(new Room(roomDataObj["roomName"].value<QString>(),
                                         roomDataObj["onlineUsers"].value<int>()
                                        ));
        Q_EMIT listRoomsChanged();
    }
}
void UCDataModel::removeRoom(const int &sequence)
{
    this->_listRooms.removeAt(sequence);
    Q_EMIT listRoomsChanged();
}
void UCDataModel::removeRoom(const QString &roomName)
{
    for (int i = 0; i < this->_listRooms.size(); i++) {
        if (this->_listRooms[i]->roomName() == roomName) {
            this->_listRooms.removeAt(i);
            break;
        }
    }

    Q_EMIT listRoomsChanged();
}
void UCDataModel::updateRoom(const QString &roomName, const int &onlineUsers)
{
    for (int i = 0; i < this->_listRooms.size(); i++) {
        if (this->_listRooms.at(i)->roomName() == roomName) {
            Room *room = this->_listRooms[i];
            room->setOnlineUsers(onlineUsers);
        }
    }

    Q_EMIT listRoomsChanged();
}
void UCDataModel::newUserJoinedRoom(const QString &roomName)
{
    for (int i = 0; i < this->_listRooms.size(); i++) {
        if (this->_listRooms.at(i)->roomName() == roomName) {
            Room *room = this->_listRooms[i];
            room->setOnlineUsers(room->onlineUsers() + 1);
        }
    }

    Q_EMIT listRoomsChanged();
}
void UCDataModel::userLeftRoom(const QString& roomName) {
    for(int i = 0; i < this->_listRooms.size(); i++ ) {
        if(this->_listRooms.at(i)->roomName() == roomName ) {
            Room* room = this->_listRooms[i];
            room->setOnlineUsers(room->onlineUsers() - 1);
        }
    }
    Q_EMIT listRoomsChanged();
}
void UCDataModel::cleanRoom(){
    this->_listRooms.clear();
    Q_EMIT listRoomsChanged();
}
UCDataModel *UCDataModel::instance()
{
    if (inst == nullptr) {
        inst = new UCDataModel(0);
    }

    return inst;
}
}
