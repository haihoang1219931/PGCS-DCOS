import QtQuick 2.5
import QtQuick.Controls 2.1
import QtQuick.Layouts 1.3
import QtGraphicalEffects 1.0
import CustomViews.UIConstants 1.0
import QGroundControl.Controls 1.0


Rectangle {
    id: root
    property var bgColorNoti: new Object({
                                             "error": "#ff7675",
                                             "warning": "#fdcb6e",
                                             "info": "#0984e3",
                                             "success": "#1dd1a1"
                                         })
    property variant typeNoti
    property alias notiMes: mainNotiMes.text
    property alias running_: notiShow.running

    color: bgColorNoti[typeNoti]
    anchors.topMargin: -height - 10
    Rectangle {
        id: mainNoti
        width: childrenRect.width + 50
        height: parent.height
        opacity: .6
        anchors.centerIn: parent
        color: UIConstants.transparentColor
        Text {
            id: mainNotiMes
            color: "#fff"
            text: qsTr("Notification Message...")
            anchors.verticalCenter: parent.verticalCenter
            x: 10
        }
//        Text {
//            text: "\uf057"
//            font{ pointSize: 18; family: ExternalFontLoader.solidFont }
//            color: "#fff"
//            x: -10
//            y: -10
//            opacity: .8
//            MouseArea {
//                anchors.fill: parent
//                cursorShape: Qt.OpenHandCursor
//                onClicked: {
//                    notiShowReverse.running = true;
//                }
//            }
//        }
    }

    NumberAnimation {
        id: notiShow
        target: root
        property: "anchors.topMargin"
        from: -height - 10
        to: 0
        duration: 1000
        running: false
        easing.type: Easing.InOutCubic
        onStopped: {
            timer.running = true
        }
    }

    NumberAnimation {
        id: notiShowReverse
        target: root
        property: "anchors.topMargin"
        from: 0
        to: -height - 10
        duration: 1000
        running: false
        easing.type: Easing.InOutCubic
    }

    Timer {
        id: timer
        interval: 3000
        running: false
        repeat: false
        onTriggered: {
            notiShowReverse.running = true
        }
    }

    function startShowNoti()
    {
        notiShow.running = true;
    }
}
