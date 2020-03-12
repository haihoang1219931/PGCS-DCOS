import QtQuick 2.0

Item {
    id: root
    width: 100
    height: 30
    property alias color: txt.color
    property alias text: txt.text
    property alias font: txt.font
    signal showFinished()
    signal hideFinished()
    function showInfo(){
        animShow.start()
    }
    function hideInfo(){
        animHide.start()
    }

    Item{
        clip: true
        anchors.top: parent.top
        anchors.topMargin: 2
        anchors.left: parent.left
        anchors.leftMargin: 2
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 2
        width: 0
        Text{
            id: txt
            anchors.fill: parent
        }
        SequentialAnimation on width {
            id: animShow
            running: false
            NumberAnimation {
                to: root.width - 2
                duration: 1000
            }
            onStopped: {
                root.showFinished();
            }
        }
        SequentialAnimation on width {
            id: animHide
            running: false
            NumberAnimation {
                to: 0
                duration: 1000
            }
            onStopped: {
                root.hideFinished();
            }
        }
    }
}
