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
    width: UIConstants.sRect * 10
    height: UIConstants.sRect * 2
    property bool isConnected: false
    property bool isShowOnMap: true
    signal visibleClicked();
    signal chatClicked();
    color: UIConstants.transparentBlue
    Rectangle {
        id: rectID
        width: UIConstants.sRect * 2
        height: UIConstants.sRect * 2
        color: "white"
        Rectangle {
            id: rectConnection
            x: 0
            y: 0
            width: 10
            height: 40
            color: isConnected?UIConstants.cConnected:UIConstants.cDisConnected

            Label {
                id: lblID
                width: 8
                height: 16
                anchors.verticalCenterOffset: 0
                anchors.horizontalCenterOffset: 0
                anchors.horizontalCenter: parent.horizontalCenter
                anchors.verticalCenter: parent.verticalCenter
                verticalAlignment: Text.AlignVCenter
                horizontalAlignment: Text.AlignHCenter
            }
        }

        Label {
            id: lblIcon
            x: 8
            y: 0
            width: 32
            height: 40
            font{ pixelSize: 18; bold: true;  family: ExternalFontLoader.solidFont }
            text: UIConstants.iCenterCommander
            color: UIConstants.cCenterCommander
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
        text: qsTr("Informations")
        color: "white"
        wrapMode: Text.WrapAnywhere
    }
    FlatButtonIcon {
        id: btnShow
        x: 114
        iconSize: 15
        isShowRect: false
        width: UIConstants.sRect
        height: UIConstants.sRect
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 0
        icon: UIConstants.iConnect
        iconColor: isConnected?"green":"gray"
        anchors.right: parent.right
        anchors.rightMargin: 0
        onClicked: {
            rootItem.isShowOnMap =! rootItem.isShowOnMap;
            rootItem.visibleClicked();
            rootItem.isConnected = !rootItem.isConnected;
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
