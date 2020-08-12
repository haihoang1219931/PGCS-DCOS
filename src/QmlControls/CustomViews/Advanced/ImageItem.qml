import QtQuick 2.0
import QtQuick.Controls 2.2
Rectangle {
    width: 95
    height: 60
    color: "#00000000"
    border.color: "gray"
    border.width: 1
    property string iconSource
    property bool needValue: false
    signal removed()
    signal needed()
    id: root

    Image {
        id: img
        source: iconSource
        anchors.top: parent.top
        anchors.topMargin: 0
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 0
        anchors.right: parent.right
        anchors.rightMargin: 0
        anchors.left: parent.left
        anchors.leftMargin: 0
        visible: !needValue
        verticalAlignment: Text.AlignVCenter
        horizontalAlignment: Text.AlignHCenter
    }

    Button {
        id: btnRemove
        x: 75
        width: 20
        height: 16
        text: qsTr("-")
        visible: !needValue
        anchors.right: parent.right
        anchors.rightMargin: 0
        anchors.top: parent.top
        anchors.topMargin: 0
        font.pointSize: 24
        font.bold: true
        onClicked: {
            root.removed();
        }
    }
    Button {
        id: btnAdd
        anchors.fill: parent
        text: qsTr("+")
        visible: needValue
        font.pointSize: 24
        font.bold: true
        onClicked: {
            needValue=!needValue;
            root.needed();
        }
    }
}
