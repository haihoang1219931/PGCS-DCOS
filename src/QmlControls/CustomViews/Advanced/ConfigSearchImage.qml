import QtQuick 2.9
import QtQuick.Controls 2.2
//------------------ Include Custom modules/plugins
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0
Rectangle {
    id: root
    color: "transparent"
    property color textColor: "white"


    Rectangle {
        id: rectangle
        color: "#00000000"
        anchors.top: parent.top
        anchors.topMargin: 8
        anchors.right: parent.right
        anchors.rightMargin: 8
        anchors.bottom: rectControl.top
        anchors.bottomMargin: 8
        anchors.left: parent.left
        anchors.leftMargin: 8

        GridView {
            id: gridView
            anchors.fill: parent
            cellWidth: 97
            cellHeight: 62
            delegate: ImageItem{
                iconSource: icon
                needValue: need
                onRemoved: {
                    gridView.model.remove(index)
                }
                onNeeded: {
                    gridView.model.append({"need":true,"icon":""})
                }
            }
            model: ListModel {

                ListElement {
                    need: true
                    icon: ""
                }
            }
//            ScrollBar.vertical: ScrollBar {
//                active: true
//            }
        }
    }
    Item {
        id: rectControl
        y: 267
        height: 45
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 8
        anchors.left: parent.left
        anchors.leftMargin: 8
        anchors.right: parent.right
        anchors.rightMargin: 8

        Button {
            id: btnStart
            x: 524
            text: qsTr("Start")
            anchors.right: parent.right
            anchors.rightMargin: 0
            anchors.top: parent.top
            anchors.topMargin: 4
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 1
            font.bold: true
        }

        Button {
            id: btnLoad
            x: 0
            y: 3
            text: qsTr("Load")
            font.bold: true
        }
    }
}

/*##^## Designer {
    D{i:0;autoSize:true;height:480;width:640}D{i:1;anchors_height:253;anchors_width:584;anchors_x:8;anchors_y:8}
D{i:8;anchors_y:3}D{i:7;anchors_width:586;anchors_x:8}
}
 ##^##*/
