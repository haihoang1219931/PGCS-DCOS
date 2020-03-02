import QtQuick 2.3
import QtQuick.Controls 1.2
import QtQuick.Controls.Styles 1.4
import QtQuick.Dialogs 1.2
import QtQuick.Layouts 1.1
import QtQuick.Window 2.0

//---------------- Include custom libs ----------------------------------------
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0
import io.qdt.dev               1.0
Rectangle {
    id: root
    color: UIConstants.transparentBlue
    radius: UIConstants.rectRadius
    border.color: "gray"
    border.width: 1
    property var vehicle
    property alias title: txtDialog.text
    property string type: ""
    property color fontColor: UIConstants.textColor
    property int fontSize: UIConstants.fontSize
    signal clicked(string type,string func)
    signal died()
    width: UIConstants.sRect * 50
    height: UIConstants.sRect * 24
    function setFocus(enable){
        rectangle.focus = enable
    }
    MouseArea {
        id: rectangle
        anchors.fill: parent
        hoverEnabled: true
        focus: true
        Keys.onPressed: {
            console.log("Key pressed "+event.key);
            console.log("Qt.Key_Return "+Qt.Key_Return);
            console.log("Qt.Key_Escape "+Qt.Key_Escape);
            if(event.key === Qt.Key_Return){
                console.log("Yes");
                btnConfirm.state = "Pressed";
                root.clicked(root.type,"DIALOG_OK");
                root.destroy(1000);
            }else if(event.key === Qt.Key_Escape){
                console.log("No");
                btnCancel.state = "Pressed";
                root.clicked(root.type,"DIALOG_CANCEL");
                root.destroy();
            }
        }
        FlatButtonIcon{
            id: btnCancel
            x: 580
            width: 30
            height: 30
            icon: UIConstants.iMouse
            isSolid: true
            color: "red"
            isAutoReturn: true
            radius: root.radius
            anchors.top: parent.top
            anchors.topMargin: 8
            anchors.right: parent.right
            anchors.rightMargin: 8
            onClicked: {
                root.clicked(root.type,"DIALOG_CANCEL");
                root.destroy();
            }
        }
        GridView {
            id: gridView
            clip: true
            anchors.top: parent.top
            anchors.topMargin: 44
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 8
            anchors.right: parent.right
            anchors.rightMargin: 8
            anchors.left: parent.left
            anchors.leftMargin: 8
            cellWidth: 150
            cellHeight: 30
            model: vehicle.propertiesModel
            layoutDirection: Qt.LeftToRight
            flow: GridView.FlowTopToBottom
            delegate: Item {
                height: gridView.cellHeight
                width: gridView.cellWidth
//                color: UIConstants.transparentColor
//                border.color: "gray"
//                border.width: 1
                CheckBox{
                    id: control
                    anchors.fill: parent
                    anchors.margins: 2
                    style: CheckBoxStyle{
                        indicator: Rectangle{
                            implicitWidth: 16
                            implicitHeight: 16
                            radius: 3
                            border.color: control.activeFocus? "darkblue":"gray"
                            Rectangle{
                                visible: control.checked
                                color: "#555"
                                border.color: "#333"
                                radius: 1
                                anchors.margins: 4
                                anchors.fill: parent
                            }
                        }
                        label: Label{
                            verticalAlignment: Label.AlignVCenter
                            horizontalAlignment: Label.AlignLeft
                            font.pixelSize: UIConstants.fontSize
                            font.family: UIConstants.appFont
                            color: UIConstants.textColor
                            text: name
                        }
                    }

                    checked: selected
                    onPressedChanged: {
                        vehicle.activeProperty(name,checked);
                    }
                }
            }
        }

        Label {
            id: txtDialog
            height: 30
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
            color: UIConstants.textColor
            anchors.top: parent.top
            anchors.topMargin: 8
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignHCenter
            anchors.right: btnCancel.left
            anchors.rightMargin: 8
            anchors.left: parent.left
            anchors.leftMargin: 8
        }
    }
    Component.onCompleted: {
        console.log("Set Focus true");
        setFocus(true)
    }
}

/*##^## Designer {
    D{i:3;anchors_width:60;anchors_x:580;anchors_y:8}
}
 ##^##*/
