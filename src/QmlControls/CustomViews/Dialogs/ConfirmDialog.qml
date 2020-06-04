import QtQuick 2.3
import QtQuick.Controls 1.2
import QtQuick.Dialogs 1.2
import QtQuick.Layouts 1.1
import QtQuick.Window 2.0

//---------------- Include custom libs ----------------------------------------
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0

Rectangle {
    id: root
    color: UIConstants.transparentBlue
    height: UIConstants.sRect * 6
    width: UIConstants.sRect * 13
    radius: UIConstants.rectRadius
    border.color: "gray"
    border.width: 1
    property alias title: txtDialog.text
    property string type: ""
    property color fontColor: UIConstants.textColor
    property int fontSize: UIConstants.fontSize
    signal clicked(string type,string func)
    signal died()
    function setFocus(enable){
        console.log("Set ConfirmDialog to "+enable);
        rectangle.focus = enable;
    }
    MouseArea {
        id: rectangle
        anchors.fill: parent
        focus: true
        Keys.onPressed: {
            console.log("Key pressed "+event.key);
            console.log("Qt.Key_Return "+Qt.Key_Return);
            console.log("Qt.Key_Escape "+Qt.Key_Escape);
            if(event.key === Qt.Key_Return){
                console.log("Yes");
                btnConfirm.state = "Pressed";
                root.clicked(root.type,"DIALOG_OK");
            }else if(event.key === Qt.Key_Escape){
                console.log("No");
                btnCancel.state = "Pressed";
                root.clicked(root.type,"DIALOG_CANCEL");
            }
        }

        FlatButtonIcon{
            id: btnConfirm
            height: UIConstants.sRect * 2
            width: UIConstants.sRect * 4
            icon: UIConstants.iChecked
            isSolid: true
            color: UIConstants.greenColor
            anchors.left: parent.left
            anchors.leftMargin: 8
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 8
            isAutoReturn: true
            radius: root.radius
            onClicked: {
                root.clicked(root.type,"DIALOG_OK");
            }
        }
        FlatButtonIcon{
            id: btnCancel
            height: UIConstants.sRect * 2
            width: UIConstants.sRect * 4
            icon: UIConstants.iMouse
            isSolid: true
            color: "red"
            anchors.right: parent.right
            anchors.rightMargin: 8
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 8
            isAutoReturn: true
            radius: root.radius
            onClicked: {
                root.clicked(root.type,"DIALOG_CANCEL");
            }
        }
        Text {
            id: txtDialog
            color: fontColor
            anchors.bottom: btnConfirm.top
            anchors.bottomMargin: 8
            anchors.top: parent.top
            anchors.topMargin: 8
            wrapMode: Text.WordWrap
            anchors.right: parent.right
            anchors.rightMargin: 8
            anchors.left: parent.left
            anchors.leftMargin: 8
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignHCenter
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
        }
    }
    Component.onCompleted: {
        console.log("Set Focus true");
        setFocus(true)
    }
}

/*##^## Designer {
    D{i:4;anchors_height:148;anchors_y:8}
}
 ##^##*/
