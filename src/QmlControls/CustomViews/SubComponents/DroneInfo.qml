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
import CustomViews.Components 1.0
Rectangle {
    id: rootItem
    width: UIConstants.sRect * 12
    height: UIConstants.sRect * 2
    property string droneState: "DRONE_OTHER"
    property int droneID: 0
    property bool isConnected: false
    property bool isShowOnMap: true
    signal visibleClicked();
    color: UIConstants.transparentBlue
    Rectangle {
        id: rectID
        width: 36
        color: "#00000000"
        anchors.top: parent.top
        anchors.topMargin: 2
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 2
        anchors.left: parent.left
        anchors.leftMargin: 5

        Rectangle {
            id: rectangle
            width: 36
            color: "#808080"
            radius: height/2
            border.color: "#808080"
            anchors.fill: parent

            Label {
                id: lblIcon
                text: UIConstants.iDrone
                anchors.rightMargin: 2
                anchors.bottomMargin: 2
                anchors.topMargin: 2
                anchors.leftMargin: 2
                anchors.fill: parent
                color: "white"
                font{ pixelSize: 18; bold: true;  family: ExternalFontLoader.solidFont }
                verticalAlignment: Text.AlignVCenter
                horizontalAlignment: Text.AlignHCenter
            }
        }

        Label {
            id: lblID
            x: 1
            width: 10
            height: 10
            text: Number(droneID).toFixed(0).toString()
            visible: false
            anchors.top: parent.top
            anchors.topMargin: 0
            anchors.right: parent.right
            anchors.rightMargin: 0
            font.pointSize: 10
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignHCenter
        }
    }

    Label {
        id: lblInfo
        y: 0
        width: 122
        height: 40
        text: qsTr("Informations")
        verticalAlignment: Text.AlignVCenter
        anchors.left: rectID.right
        anchors.leftMargin: 6
        wrapMode: Text.WrapAnywhere
        color: "white"
        font.family: "monospace"

    }
    FlatButtonIcon {
        id: btnShow
        x: 174
        iconSize: 20
        isShowRect: false
        width: UIConstants.sRect*2
        height: UIConstants.sRect*2
        anchors.verticalCenterOffset: 0
        anchors.verticalCenter: parent.verticalCenter
        icon: isShowOnMap?UIConstants.iVisible:UIConstants.iInvisible
        anchors.right: parent.right
        anchors.rightMargin: 26
        onClicked: {
            rootItem.isShowOnMap =! rootItem.isShowOnMap;
            rootItem.visibleClicked();
//            rootItem.isConnected =! rootItem.isConnected;
        }
    }
    FlatIcon{
        id: iconOnline
        x: 176
        width: 7
        height: 7
        text: UIConstants.iEnabled
        anchors.rightMargin: 15
        color: "green"
        anchors.right: parent.right
        anchors.verticalCenter: parent.verticalCenter
        font.pixelSize: 7
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
    D{i:3;anchors_height:40;anchors_width:32;anchors_x:8;anchors_y:0}D{i:1;anchors_height:40}
D{i:6;anchors_x:46}
}
 ##^##*/
