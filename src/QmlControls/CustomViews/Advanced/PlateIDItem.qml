import QtQuick 2.0
import QtQuick.Controls 2.2
Rectangle {
    width: 190
    height: 40
    color: "#00000000"
    border.color: "gray"
    border.width: 1
    property string txt
    property bool needValue: false
    signal removed()
    signal needed()
    id: root

    TextInput {
        id: lblID
        color: "#ffffff"
        text: txt
        anchors.top: parent.top
        anchors.topMargin: 6
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 6
        maximumLength: 10
        font.bold: true
        anchors.right: btnRemove.left
        anchors.rightMargin: 6
        anchors.left: parent.left
        anchors.leftMargin: 6
        visible: !needValue
        verticalAlignment: Text.AlignVCenter
        horizontalAlignment: Text.AlignHCenter
    }

    Button {
        id: btnRemove
        x: 123
        width: 30
        text: qsTr("-")
        visible: !needValue
        anchors.right: parent.right
        anchors.rightMargin: 8
        anchors.top: parent.top
        anchors.topMargin: 5
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 6
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
