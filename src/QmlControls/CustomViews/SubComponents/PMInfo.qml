/**
 * ==============================================================================
 * @Project: VCM01TargetViewer
 * @Component: Center Commander Info
 * @Breif:
 * @Author: Hai Nguyen Hoang
 * @Date: 23/05/201
9
 * @Language: QML
 * @License: (c) Viettel Aerospace Institude - Viettel Group
 * ============================================================================
 */

//----------------------- Include QT libs -------------------------------------
//------------------ Include QT libs ------------------------------------------
import QtQuick.Window 2.2
import QtQuick 2.6
import QtQuick.Controls 2.1
import QtQuick.Layouts 1.3
import QtGraphicalEffects 1.0
import QtQuick 2.0
//---------------- Include custom libs ----------------------------------------
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0

Rectangle {
    id: rootItem
    width: UIConstants.sRect * 12
    height: UIConstants.sRect * 2
    property string pmState: "PM_NORMAL"
    property int pcdID: 0
    property bool isConnected: false
    property bool isShowOnMap: false
    signal visibleClicked();
    signal chatClicked();
    signal shareClicked();
    color: UIConstants.transparentBlue
    Rectangle {
        id: rectID
        width: UIConstants.sRect * 2
        height: UIConstants.sRect * 2
        color: pmState === "PM_NORMAL"?UIConstants.cPatrolManNormal:UIConstants.cPatrolManNeedHelp

        Rectangle {
            id: rectConnection
            x: 0
            y: 0
            width: 10
            height: 40
            color: isConnected?UIConstants.cConnected:UIConstants.cDisConnected
        }

        Label {
            id: lblIcon
            x: 8
            y: 0
            width: 32
            height: 40
            text: UIConstants.iPatrolMan
            font{ pixelSize: 18; bold: true;  family: ExternalFontLoader.solidFont }
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignHCenter
        }

        Label {
            id: lblID
            x: 1
            width: 10
            height: 10
            text: Number(pcdID).toFixed(0).toString()
            anchors.top: parent.top
            anchors.topMargin: 0
            anchors.right: parent.right
            anchors.rightMargin: 0
            font.pixelSize: UIConstants.fontSize
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignHCenter
        }
    }

    Label {
        id: lblInfo
        x: 46
        y: 0
        width: 68
        height: 40
        text: qsTr("")
        wrapMode: Text.WrapAnywhere
        color: "white"
    }
    FlatButtonIcon {
        id: btnShow
        isShowRect: false
        x: 140
        iconSize: 16
        width: UIConstants.sRect
        height: UIConstants.sRect
        anchors.top: parent.top
        anchors.topMargin: 0
        icon: isShowOnMap?UIConstants.iVisible:UIConstants.iInvisible
//        iconColor:
        anchors.right: btnShare.left
        anchors.rightMargin: 2
        onClicked: {
//            rootItem.isShowOnMap =! rootItem.isShowOnMap;
//            rootItem.visibleClicked();
            rootItem.isShowOnMap = !rootItem.isShowOnMap;
        }
    }
    FlatButtonIcon {
        id: btnShare
        x: 160
        isShowRect: false
        iconSize: 15
        width: UIConstants.sRect
        height: UIConstants.sRect
        anchors.top: parent.top
        anchors.topMargin: 0
        icon: UIConstants.iShare
        anchors.right: btnChat.left
        anchors.rightMargin: 2
        onClicked: {
            rootItem.shareClicked();
        }
    }
    FlatButtonIcon {
        id: btnChat
        isShowRect: false
        iconSize: 15
        width: UIConstants.sRect
        height: UIConstants.sRect
        anchors.top: parent.top
        anchors.topMargin: 0
        icon: UIConstants.iChatIcon
        anchors.right: parent.right
        anchors.rightMargin: 0
        onClicked: {
            rootItem.chatClicked();
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

/*##^## Designer {
    D{i:6;anchors_y:0}
}
 ##^##*/
