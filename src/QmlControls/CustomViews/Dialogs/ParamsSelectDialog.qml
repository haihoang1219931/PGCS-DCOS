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

        GridView {
            id: gridView
            clip: true
            anchors.top: btnCancel.bottom
            anchors.topMargin: 8
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 8
            anchors.right: parent.right
            anchors.rightMargin: 8
            anchors.left: parent.left
            anchors.leftMargin: 8
            cellWidth: UIConstants.sRect * 7
            cellHeight: UIConstants.sRect * 3 / 2

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
                    anchors.margins: 4
                    style: CheckBoxStyle{
                        indicator: Rectangle{
                            implicitWidth: UIConstants.sRect
                            implicitHeight: UIConstants.sRect
                            radius: 3
                            border.color: control.activeFocus? "darkblue":"gray"
                            Rectangle{
                                visible: control.checked
                                color: "#555"
                                border.color: "#333"
                                radius: 1
                                anchors.margins: UIConstants.sRect / 4
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
                        if(vehicle !== null)
                            vehicle.activeProperty(name,checked);
                    }
                }
            }
        }
    }
    Label {
        id: txtDialog
        height: btnCancel.height
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
    FlatButtonIcon{
        id: btnCancel
        height: UIConstants.sRect * 2
        width: UIConstants.sRect * 2
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
            console.log("pressed");
            root.clicked(root.type,"DIALOG_CANCEL");
        }
    }
    Component.onCompleted: {
        console.log("Set Focus true");
        setFocus(true);
        if(vehicle !== null)
            gridView.model = vehicle.propertiesModel;
    }
}
