import QtQuick 2.9
import QtQuick.Controls 2.0
//------------------ Include Custom modules/plugins
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0

Rectangle {
    id: root
    color: "transparent"
    property color textColor: "white"
    property real iconRatio: 6/8

    signal searchByClass(var listClasses)

    Rectangle {
        id: rectObject
        width: 290
        color: UIConstants.transparentColor
        clip: true
        anchors.bottom: rectControl.top
        anchors.bottomMargin: 8
        anchors.top: parent.top
        anchors.topMargin: 8
        anchors.left: parent.left
        anchors.leftMargin: 8

        DropArea{
            anchors.top: rectangle.bottom
            anchors.topMargin: 8
            anchors.right: parent.right
            anchors.rightMargin: 2
            anchors.left: parent.left
            anchors.leftMargin: 2
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 0
            Rectangle{
                anchors.fill: parent
                color: "#80afacac"
                visible: parent.containsDrag
                anchors.rightMargin: 10
            }
            ListView {
                id: lstObject
                anchors.fill: parent
                delegate: DragItem{
                    width: parent.width-10
                    txt: name
                    icon: iconSource
                    num: number
                }
                spacing: 2
                model: ListModel {
                    ListElement {name: "people";iconSource: "qrc:/qmlimages/objects/human.png"; number: "1,2"}
                    ListElement {name: "car";iconSource: "qrc:/qmlimages/objects/car.png"; number: "4,5,6,9"}
                    ListElement {name: "moto";iconSource: "qrc:/qmlimages/objects/bike.png"; number: "10"}
                    ListElement {name: "bicycle";iconSource: "qrc:/qmlimages/objects/bike.png"; number: "3"}
                    ListElement {name: "tricycle";iconSource: "qrc:/qmlimages/objects/bike.png"; number: "7,8"}
                    ListElement {name: "others";iconSource: "qrc:/qmlimages/objects/tree.png"; number: "11"}
                }
//                ScrollBar.vertical: ScrollBar {
//                    activeFocus: true
//                }
            }
            onDropped: {
                console.log("to object drop.text="+drop.text);
                var temp = JSON.parse(drop.text);
                var willAdd = true;
                for(var i=0; i< lstObject.model.count; i++){
                    if(lstObject.model.get(i).name === temp["name"]){
                        willAdd = false;
                        break;
                    }
                }
                if(willAdd === true){
                    lstObject.model.append({"name":temp["name"],
                                               "iconSource":temp["iconSource"], "number":temp["number"]});
                }
                for(i=0; i< lstSelected.model.count; i++){
                    if(lstSelected.model.get(i).name === temp["name"]){
                        lstSelected.model.remove(i);
                        break;
                    }
                }
            }
        }


        Rectangle {
            id: rectangle
            width: 290
            height: 44
            color: UIConstants.bgAppColor

            Label {
                id: lblSearch
                x: 2
                y: 13
                width: 64
                height: 19
                text: qsTr("Object")
                anchors.left: parent.left
                anchors.leftMargin: 2
                color: UIConstants.textColor
                font.family: UIConstants.appFont
                font.pixelSize: UIConstants.fontSize
            }

            TextField {
                id: txtSearch
                x: 72
                y: 2
                width: 205
                height: 40
                anchors.right: parent.right
                anchors.rightMargin: 14
                placeholderText: "Object type"
            }
        }
    }

    Rectangle {
        id: rectSelect
        x: 3
        y: 0
        color: UIConstants.transparentColor
        clip: true
        anchors.right: parent.right
        anchors.rightMargin: 21
        anchors.bottom: rectControl.top
        anchors.bottomMargin: 6
        anchors.leftMargin: 329
        anchors.left: parent.left
        anchors.top: parent.top
        anchors.topMargin: 10
        DropArea {
            anchors.top: rectangle2.bottom
            anchors.topMargin: 8
            anchors.leftMargin: 2
            anchors.left: parent.left
            anchors.rightMargin: 2
            anchors.bottom: parent.bottom
            anchors.right: parent.right
            anchors.bottomMargin: 0
            Rectangle{
                anchors.fill: parent
                anchors.rightMargin: 10
                color: "#80afacac"
                visible: parent.containsDrag
            }
            ListView {
                id: lstSelected
                anchors.fill: parent
                spacing: 2
                delegate: DragItem {
                    width: parent.width-10
                    txt: name
                    icon: iconSource
                    num: number
                }
                model: ListModel {}
//                ScrollBar.vertical: ScrollBar {
//                    activeFocus: true
//                }
            }
            onDropped: {
                console.log("to selected drop.text="+drop.text);
                var temp = JSON.parse(drop.text);
                var willAdd = true;
                for(var i=0; i< lstSelected.model.count; i++){
                    if(lstSelected.model.get(i).name === temp["name"]){
                        willAdd = false;
                        break;
                    }
                }
                if(willAdd === true){
                    lstSelected.model.append({"name":temp["name"],
                                                "iconSource":temp["iconSource"],
                                                "number":temp["number"]});
                }
                for(i=0; i< lstObject.model.count; i++){
                    if(lstObject.model.get(i).name === temp["name"]){
                        lstObject.model.remove(i);
                        break;
                    }
                }
            }
        }

        Rectangle {
            id: rectangle2
            width: 290
            height: 44
            color: UIConstants.bgAppColor

            Label {
                id: lblSelect
                x: 0
                y: 9
                width: 290
                height: 21
                text: qsTr("Selected")
                anchors.horizontalCenterOffset: 0
                anchors.horizontalCenter: parent.horizontalCenter
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                color: UIConstants.textColor
                font.family: UIConstants.appFont
                font.pixelSize: UIConstants.fontSize
            }
        }
    }

    Item {
        id: rectControl
        x: 8
        y: 267
        width: 586
        height: 45
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 8
        anchors.left: parent.left
        anchors.leftMargin: 8
        anchors.right: parent.right
        anchors.rightMargin: 8

        Button {
            id: button
            x: 486
            text: !searchByClass?qsTr("Start"):qsTr("Stop")
            anchors.right: parent.right
            anchors.rightMargin: 0
            anchors.top: parent.top
            anchors.topMargin: 4
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 1
            font.bold: true
            property bool searchByClass: false
            onClicked: {
                //Get classes for search
                searchByClass = !searchByClass
                var selectedList = [];
                if(searchByClass){
                    for(var i=0; i< lstSelected.model.count; i++){
                        var ids = lstSelected.model.get(i).number.split(",")
                        for(var j = 0; j < ids.length; j++){
                            selectedList.push(Number(ids[j]));
                        }
                    }
                }
//                console.log(selectedList);
                videoPane.searchByClass(selectedList);
            }
        }
    }
}

/*##^## Designer {
    D{i:0;autoSize:true;height:480;width:640}D{i:27;anchors_y:3}
}
 ##^##*/
