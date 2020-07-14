import QtQuick 2.9
import QtQuick.Controls 2.4
import QtQuick.Layouts 1.3
import QtWebEngine 1.7

import QSyncable    1.0

import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0
import io.qdt.dev   1.0
// This is available in all editors.
Flickable {
    id: rootItem
    clip: true
    width: UIConstants.sRect * 12
    height: UIConstants.sRect * 22
    property var listUsers: UCDataModel.listUsers
    //[ {ipAddress: String, user_id: String, available: bool, ... }, {...}, {...} ]

    //signal doubleClicked(string ipAddress)
    Rectangle{
        anchors.fill: parent
        color: UIConstants.transparentBlue
    }

    ListView {
        id: listUsersView
        anchors.fill: parent
        clip: true
        model: JsonListModel {
           keyField: "uid"
           source: JSON.parse(JSON.stringify(Object.keys(UCDataModel.listUsers).map(function(key) {return UCDataModel.listUsers[key]})))
           fields: ["ipAddress", "userId", "roomName", "available", "role", "shared", "latitude", "longitude", "uid", "name", "isSelected", "warning", "connectionState"]
        }

        delegate: Rectangle {
            id: userItem
            width: rootItem.width
            height: UIConstants.sRect * 2
            color: model.isSelected ? UIConstants.cSelectedColor : UIConstants.transparentColor
            Text {
                text: UIConstants.iPatrolMan
                font{ pixelSize: 20;
                    weight: Font.Bold;
                    // family: ExternalFontLoader.solidFont
                }
                color: model.connectionState ? UIConstants.greenColor: UIConstants.cDisableColor
                anchors {
                    left: parent.left;
                    leftMargin: 20;
                    verticalCenter: parent.verticalCenter
                }
            }

            Text {
                text: "<p><strong>" + model.name + "</strong></p>"
                color: connectionState?UIConstants.textColor:UIConstants.cDisableColor
                textFormat: Text.RichText
                font.family: UIConstants.appFont
                font.pixelSize: UIConstants.fontSize
                anchors {
                    left: parent.left;
                    leftMargin: 60;
                    verticalCenter: parent.verticalCenter
                }
            }

            Text {
                text: model.role === UserRoles.FCS ?
                          UIConstants.iCircle : (model.available ?
                            UIConstants.iAddUser  : (model.shared ?
                                UIConstants.iPlayState :
                                    UIConstants.iRemoveUser + (model.roomName ? "<span style='font-size: 10px'>(" + model.roomName + ")</span>" : "")))
                color: model.connectionState ?
                           (model.available ? "#2ecc71" : "#c0392b") : UIConstants.cDisableColor
                font{ pixelSize: 16;
                    weight: Font.Bold;
                    //family: ExternalFontLoader.solidFont
                }
                textFormat: Text.RichText
                anchors {
                    right:parent.right;
                    rightMargin: 30;
                    verticalCenter: parent.verticalCenter
                }

                MouseArea {
                    visible: model.connectionState ? (model.available ? (model.role === UserRoles.FCS ? false : true)  : (model.shared ? true : false)): false
                    anchors.fill: parent
                    onClicked: {
                        if (model.shared) {
                            UcApi.stopSharePcdVideoFromRoom(model.uid);
                        } else {
                            UcApi.addPcdToRoom(model.uid);
                        }
                    }
                }
            }
            Canvas{
                anchors.fill: parent
                onPaint: {
                    // create canvas context
                    var ctx = getContext("2d");
                    // set drawing stype
                    ctx.strokeStyle = "gray";
                    ctx.fillStyle = "gray";
                    ctx.lineWidth = 1;
                    ctx.moveTo(0,0);
                    ctx.lineTo(0,height)
                    ctx.lineTo(width,height)
                    ctx.lineTo(width,0)
                    ctx.stroke();
                }
            }
            MouseArea {
                width: parent.width * 3 / 4
                height: parent.height
                anchors.top: parent.top
                anchors.left: parent.left
                onClicked: {
                    //--- Reset state
                    UCDataModel.updateUser(model.uid, 10, !model.isSelected);
                    UCEventListener.pointToPcdFromSidebar(model.uid, !model.isSelected);
                }
            }

            SequentialAnimation {
                running: warning
                loops: Animation.Infinite
                PropertyAnimation {
                    target: userItem
                    properties: "color"
                    to: "#fc5c65"
                    duration: 700
                    easing.type: Easing.InOutBounce
                }
                PropertyAnimation {
                    target: userItem
                    properties: "color"
                    to: UIConstants.transparentColor
                    duration: 300
                    easing.type: Easing.InOutBounce
                }
                onRunningChanged: {
                    if (!model.warning) {
                        UCDataModel.updateUser(model.id, 10, false);
                    }
                }
            }
        }
        Canvas{
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.top: parent.top
            height: 1
            onPaint: {
                // create canvas context
                var ctx = getContext("2d");
                // set drawing stype
                ctx.strokeStyle = "gray";
                ctx.fillStyle = "gray";
                ctx.lineWidth = 1;
                ctx.moveTo(0,0);
                ctx.lineTo(width,0)
                ctx.stroke();
            }
        }
    }
}
