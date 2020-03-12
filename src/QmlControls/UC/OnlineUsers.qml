import QtQuick 2.9
import QtQuick.Controls 2.4
import QtQuick.Layouts 1.3
import QtWebEngine 1.7
import QtGraphicalEffects 1.0

import CustomViews.UIConstants 1.0
import io.qdt.dev   1.0
// This is available in all editors.
Flickable {
    id: rootItem
    clip: true
    width: UIConstants.sRect * 12
    height: UIConstants.sRect * 22
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
        model: UC_API?UCDataModel.listUsers:[]

        delegate: Rectangle {
            width: rootItem.width
            height: UIConstants.sRect * 2
            color: isSelected ? UIConstants.cSelectedColor : UIConstants.transparentColor
            Text {
                text: UIConstants.iPatrolMan
                font{
                    pixelSize: UIConstants.fontSize
                    weight: Font.Bold;
                    family: ExternalFontLoader.solidFont}
                color: connectionState ? UIConstants.greenColor: UIConstants.cDisableColor
                anchors {
                    left: parent.left;
                    leftMargin: 20;
                    verticalCenter: parent.verticalCenter
                }
            }

            Text {
                text: "<p>" + name + "</p>"
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
                text: role === UserRoles.FCS ?
                          UIConstants.iCircle : (available ?
                            UIConstants.iAddUser  : (shared ?
                                UIConstants.iShareVideo :
                                    UIConstants.iRemoveUser + (roomName ? "<span style='font-size: 10px'>(" + roomName + ")</span>" : "")))
                color: connectionState ?
                           (available ? "#2ecc71" : "#c0392b") : UIConstants.cDisableColor
                font{ pixelSize: 16;
                    weight: Font.Bold;
                    family: ExternalFontLoader.solidFont}
                textFormat: Text.RichText
                anchors {
                    right:parent.right;
                    rightMargin: 30;
                    verticalCenter: parent.verticalCenter
                }

                MouseArea {
                    visible: connectionState ? (available ? (role === UserRoles.FCS ? false : true)  : false): false
                    anchors.fill: parent
                    onClicked: {
                        UcApi.addPcdToRoom(uid);
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
                    console.log("Selected = "+UCDataModel.listUsers[index].isSelected);
                    console.log("role["+role+"] vs UserRoles.FCS["+UserRoles.FCS+"]");
                    UCDataModel.updateUser(uid, 10, !isSelected);
                    UCEventListener.pointToPcdFromSidebar(uid, !isSelected);
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
