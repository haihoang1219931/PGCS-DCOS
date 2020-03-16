import QtQuick 2.9
import QtQuick.Controls 2.4
import QtQuick.Layouts 1.3
import QtWebEngine 1.7
import QtGraphicalEffects 1.0

import CustomViews.UIConstants 1.0
// This is available in all editors.
Flickable {
    id: rootItem
    clip: true
    width: UIConstants.sRect * 12
    height: UIConstants.sRect * 3
    //[ {ipAddress: String, user_id: String, available: bool, ... }, {...}, {...} ]
    Rectangle{
        anchors.fill: parent
        color: UIConstants.transparentBlue
    }
    ListView {
        id: listUsersView
        anchors.fill: parent
        clip: true
        model: UC_API?UCDataModel.listRooms:[]

        delegate: Rectangle {
            width: rootItem.width
            height: UIConstants.sRect * 3/2
//            border.color: UIConstants.grayLighterColor
//            border.width: 1
            color: UIConstants.transparentColor
            state: "normal"
            Text {
                text: UIConstants.iDrone
                font.bold: true
                font.pixelSize: UIConstants.fontSize
                color: UIConstants.greenColor
                anchors {
                    left: parent.left;
                    leftMargin: 20;
                    verticalCenter: parent.verticalCenter
                }
            }

            Text {
                text: roomName
                color: UIConstants.textColor
                font.family: UIConstants.appFont
                font.pixelSize: UIConstants.fontSize
                anchors {
                    left: parent.left;
                    leftMargin: 60;
                    verticalCenter: parent.verticalCenter
                }
            }

            Text {
                text: onlineUsers
                color: UIConstants.textColor
                font.pixelSize: UIConstants.fontSize
                font.family: UIConstants.appFont
                textFormat: Text.RichText
                anchors {
                    right:parent.right;
                    rightMargin: 30;
                    verticalCenter: parent.verticalCenter
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
