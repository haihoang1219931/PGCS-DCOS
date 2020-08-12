/**
 * ==============================================================================
 * @Project: FCS-Groundcontrol-based
 * @Module: PreflightCheck page
 * @Breif:
 * @Author: Hai Nguyen Hoang
 * @Date: 14/05/2019
 * @Language: QML
 * @License: (c) Viettel Aerospace Institude - Viettel Group
 * ============================================================================
 */

//------------------ Include QT libs ------------------------------------------
import QtQuick.Window 2.2
import QtQuick 2.6
import QtQuick.Controls 2.1
import QtQuick.Layouts 1.3
import QtGraphicalEffects 1.0
import QtQuick 2.0

//------------------ Include Custom modules/plugins
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0
import CustomViews.SubComponents 1.0
Rectangle{
    id: root
    width: root.showCam ? UIConstants.sRect*12:UIConstants.sRect*2
    height: !root.smallSize ? UIConstants.sRect*14:UIConstants.sRect*2
    color: "transparent"
    property int pcdID: 1
    property string pcdName: "PM A"
    property bool smallSize: true
    property bool showCam: false
    property color pcdStateColor: "green"
    signal showOnTop(var show);
    border.color: "gray"
    border.width: 1
    PMGraphics {
        id: rectID
        width: UIConstants.sRect*2
        height: UIConstants.sRect*2
        iconColor: pcdStateColor
        Label {
            id: lblID
            x: 24
            y: 0
            width: 16
            height: 16
            text: Number(pcdID).toFixed(0).toString()
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignHCenter
        }
        MouseArea{
            drag.target: root
            hoverEnabled: true
            anchors.fill: parent
            onPressed: {
                root.showOnTop(true);
//                root.showCam=!root.showCam;
            }
            onClicked: {
                root.showCam=!root.showCam;
                root.smallSize = !root.showCam;
            }
        }
    }
    FlatButtonIcon{
        id: btnCall
        anchors.top: parent.top
        anchors.right: parent.right
        width: UIConstants.sRect*2
        height: UIConstants.sRect*2
        icon: !root.smallSize? UIConstants.iUp:UIConstants.iDown
        iconColor: "gray"
        color: "white"
        border.color: "gray"
        border.width: 1
        visible: showCam
        onClicked: {
//            if(root.showCam === false) root.showCam = true;
            root.smallSize=!root.smallSize;
            root.showOnTop(true);
        }
//        onEntered: {
//            root.showOnTop(true);
//        }
//        onExited: {
//            root.showOnTop(false);
//        }
    }
    FlatButtonIcon{
        id: btnShareAll
        anchors.top: parent.top
        anchors.right: btnCall.left
        icon: UIConstants.iShare
        iconColor: "gray"
        color: "white"
        border.color: "gray"
        border.width: 1
        visible: showCam
        width: UIConstants.sRect*2
        height: UIConstants.sRect*2
        onClicked: {
            root.showOnTop(true);
        }

//        onEntered: {
//            root.showOnTop(true);
//        }
//        onExited: {
//            root.showOnTop(false);
//        }
    }
    Rectangle {
        id: rectName
        anchors.left: rectID.right
        anchors.top: parent.top
        anchors.right: btnShareAll.left
        height: UIConstants.sRect*2
        color: "white"
        border.color: "gray"
        border.width: 1
        visible: showCam
        Label {
            id: lblName
            text: pcdName
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.verticalCenter: parent.verticalCenter
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignHCenter
        }
        MouseArea{
            drag.target: root
            hoverEnabled: true
            anchors.fill: parent
            onPressed: {
                root.showOnTop(true);
//                root.showCam=!root.showCam;
            }
        }
    }

    Rectangle{
        id: rectVideo
        anchors.left: parent.left
        anchors.bottom: parent.bottom
        anchors.right: parent.right
        anchors.top: rectName.bottom
        visible: root.showCam && !root.smallSize
        border.color: "gray"
        border.width: 1
        MouseArea{
            drag.target: root
            hoverEnabled: true
            anchors.fill: parent
            onPressed: {
                root.showOnTop(true);
//                root.showCam=!root.showCam;
            }
        }
    }
//    Rectangle{
//        id: rectBound
////        anchors.fill: parent
//        width: root.showCam ? UIConstants.sRect*12:UIConstants.sRect*2
//        height: UIConstants.sRect*2
//        color: UIConstants.transparentColor
//        border.color: "gray"
//        border.width: 1
//    }
}
