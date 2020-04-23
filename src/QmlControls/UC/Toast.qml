/**
 * ==============================================================================
 * @Project: UC-FCS
 * @Module: Toast
 * @Breif:
 * @Author: Trung Ng
 * @Date: 26/08/2019
 * @Language: QML
 * @License: (c) Viettel Aerospace Institude - Viettel Group
 * ============================================================================
 */

import QtQuick 2.9
import QtQuick.Window 2.2
import QtQuick.Controls 2.4
import QtQuick.Layouts 1.3
import QtWebEngine 1.7
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0
import io.qdt.dev   1.0
//-------- :Toast notification
Rectangle{
    id: toast

    //--- Properties
    property alias toastContent: toastContent.text
    property string actionType: "add_to_room"
    property var injectData: null
    property alias callActionAppearance: callAction.visible
    property alias rejectButtonAppearance: rejectBtn.visible
    property int pauseAnimationTime: pauseAnimation.duration

    //--- Signals
//    signal callAction(var actionType);
//    signal ignoreRequestJoinRoom();

    //--- Dimensions
    width: toastContent.paintedWidth * 1.1 + 30
    height: toastContent.paintedHeight + callAction.height + 30

    //--- Attributes
    color: "#487eb0"
    border { width: 1; color: "#40739e" }
    opacity: 0
    z: 100000
    radius: 10
    //--- Content
    Column {
        spacing: 10
        anchors.verticalCenter: parent.verticalCenter
        anchors.horizontalCenter: parent.horizontalCenter
        Text {
            id: toastContent
            anchors.horizontalCenter: parent.horizontalCenter
            text: "This is sample notification"
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
            color: UIConstants.textColor
            wrapMode: Text.NoWrap
            z: 1000
        }

        Row {
            id: btnsGroup
            anchors.horizontalCenter: parent.horizontalCenter
            spacing: 5
            Rectangle {
                id: rejectBtn
                color: "#c0392b"
                width: callAction.width
                height: callAction.height
                radius: 5
                visible: false
                z: 1000
                Text {
                    text: "Bỏ qua"
                    font.pixelSize: UIConstants.fontSize
                    font.family: UIConstants.appFont
                    color: "#f5f6fa"
                    anchors.centerIn: parent
                }

                MouseArea {
                    anchors.fill: parent
                    onClicked: {
                        ignoreCallaction();
                    }
                }
            }


            Rectangle {
                id: callAction
                color: "#27ae60"
                width: btnText.paintedWidth + 20
                height: btnText.paintedHeight + 20
                radius: 5
                visible: false
                z: 1000
                Text {
                    id: btnText
                    text: "Thêm"
                    font.pixelSize: UIConstants.fontSize
                    font.family: UIConstants.appFont
                    color: "#f5f6fa"
                    anchors.centerIn: parent
                }

                MouseArea {
                    anchors.fill: parent
                    onClicked: {
                        hanldeCallaction();
                    }
                }
            }

        }
    }
    SequentialAnimation on opacity {
        id: animation
        running: false

        PauseAnimation {
            duration: 200
        }

        NumberAnimation {
            to: 0.95
            duration: 1000
        }


        PauseAnimation {
            id: pauseAnimation
            duration: 2500
        }

        NumberAnimation {
            to: 0
            duration: 500
        }

    }

    function show() {
        animation.running = true;
    }

    //-------- :JS supported methods
    function hanldeCallaction() {
        console.log(toast.actionType, toast.injectData);
        toast.opacity = 0;
        pauseAnimation.duration = 1500;
        callAction.visible = false;
        rejectBtn.visible = false;
//        toast.callAction(toast.actionType);

        if( toast.actionType === "request_join_room") {
            UcApi.replyRequestJoinRoom(true, toast.injectData);
        }
        if( toast.actionType === "add_to_room") {
            UcApi.addPcdToRoom(toast.injectData);
        }


    }

    function ignoreCallaction() {
        console.log(toast.actionType, toast.injectData);
        toast.opacity = 0;
        pauseAnimation.duration = 1500;
        callAction.visible = false;
        rejectBtn.visible = false;
        if (toast.actionType === "request_join_room") {
            UcApi.replyRequestJoinRoom(false, toast.injectData);
//            toast.ignoreRequestJoinRoom();
        }
    }
}
