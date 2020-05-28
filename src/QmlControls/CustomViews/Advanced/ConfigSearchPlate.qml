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
        color: "transparent"
        anchors.rightMargin: 8
        anchors.leftMargin: 8
        anchors.topMargin: 8
        anchors.bottom: rectControl.top
        anchors.right: parent.right
        anchors.left: parent.left
        anchors.top: parent.top
        anchors.bottomMargin: 8

        GridView {
            id: gridView
            anchors.fill: parent
            delegate: PlateIDItem{
                txt: name
                needValue: need
                onRemoved: {
                    gridView.model.remove(index)
                }
                onNeeded: {
                    gridView.model.append({"need":true,"name":""})
                }
            }
            cellHeight: 42
            cellWidth: 192
            model: ListModel {
                ListElement {
                    name: "51F-68057"
                    need: false
                }

                ListElement {
                    name: "42A-13541"
                    need: false
                }

                ListElement {
                    name: "12G-98620"
                    need: false
                }

                ListElement {
                    name: ""
                    need: true
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
//        color: "#1A1A1A"
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 8
        anchors.right: parent.right
        anchors.rightMargin: 8
        anchors.left: parent.left
        anchors.leftMargin: 8

        Button {
            id: btnStart
            x: 516
            text: qsTr("Start")
            anchors.top: parent.top
            anchors.topMargin: 4
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 1
            anchors.right: parent.right
            anchors.rightMargin: 0
            font.bold: true
            onClicked: {
                var selectedList = [];
                for(var i=0; i< gridView.model.count; i++){
                    var id = gridView.model.get(i).name
                    selectedList.push(id);
                }
                console.log(selectedList);
            }
        }

        Button {
            id: btnLoad
            x: 0
            y: 3
            text: qsTr("Load")
            font.bold: true
        }

        Button {
            id: btnSave
            x: 106
            y: 3
            text: qsTr("Save")
            font.bold: true
        }

        Button {
            id: btnClear
            x: 212
            y: 3
            text: qsTr("Clear")
            font.bold: true
        }
    }
}

/*##^## Designer {
    D{i:0;autoSize:true;height:480;width:640}D{i:1;anchors_height:253;anchors_width:584;anchors_x:8;anchors_y:8}
D{i:33;anchors_y:4}
}
 ##^##*/
