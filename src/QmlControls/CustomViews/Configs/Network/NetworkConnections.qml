import QtQuick 2.11
import QtQuick.Controls 2.4
import QtQuick.Layouts 1.3
import io.qdt.dev 1.0
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0
Rectangle {
    id: root
    width: 640
    height: 480
    color: UIConstants.transparentColor
    property alias model: lstSetting.model
    property string selectedSetting
    signal addClicked()
    signal editClicked(string selectedSetting)
    signal deleteClicked(string selectedSetting)
    Rectangle {
        id: rectangle
        color: UIConstants.transparentColor
        anchors.right: clmButton.left
        anchors.rightMargin: 8
        anchors.left: parent.left
        anchors.leftMargin: 8
        anchors.top: parent.top
        anchors.topMargin: 8
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 54
        border.color: UIConstants.grayColor
        ListView{
            id: lstSetting
            anchors.fill: parent
            anchors.margins: 5
            spacing: 5
            delegate: Column{
                height: rowInterface.height +
                        (lstConnections.visible?lstConnections.height:0)
                spacing: 5
                Row{
                    id: rowInterface
                    height: UIConstants.sRect
                    FlatButtonText{
                        isAutoReturn: true
                        width: UIConstants.sRect * 7
                        height: parent.height
                        border.width: 1
                        border.color: UIConstants.grayColor
                        color: UIConstants.grayColor
                        text: bearerTypeName
                        onClicked: {
                            lstConnections.visible=!lstConnections.visible;
                        }
                    }
                }
                ListView{
                    id: lstConnections
                    anchors.left: parent.left
                    anchors.leftMargin: UIConstants.sRect
                    width: UIConstants.sRect * 16
                    height: model.length * (UIConstants.sRect + spacing)
                    model: listNetwork
                    spacing: 5
                    clip: true

                    delegate: Item{
                        width: parent.width
                        height: UIConstants.sRect
                        QLabel{
                            anchors.fill: parent
                            text: " "+name
                            color: selectedSetting === setting?
                                   UIConstants.orangeColor: UIConstants.transparentColor
                            horizontalAlignment: Label.AlignLeft
                        }
                        MouseArea{
                            anchors.fill: parent
                            hoverEnabled: true
                            onClicked: {
                                console.log("Clicked at "+setting);
                                if(selectedSetting !== setting){
                                    selectedSetting = setting
                                }else{
                                    selectedSetting = ""
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Column{
        id: clmButton
        width: UIConstants.sRect*3
        anchors.right: parent.right
        anchors.rightMargin: 8
        anchors.top: parent.top
        anchors.topMargin: 8
        spacing: 8
        FlatButtonText {
            id: btnAdd
            isAutoReturn: true
            text: qsTr("Add")
            border.width: 1
            border.color: UIConstants.grayColor
            width: UIConstants.sRect*3
            height: UIConstants.sRect
        }

        FlatButtonText {
            id: btnEdit
            isAutoReturn: true
            width: UIConstants.sRect*3
            height: UIConstants.sRect
            border.width: 1
            border.color: UIConstants.grayColor
            isEnable: selectedSetting.includes("Setting")
            text: qsTr("Edit")
            onClicked: {
                root.editClicked(selectedSetting);
            }
        }


        FlatButtonText {
            id: btnDelete
            isAutoReturn: true
            width: UIConstants.sRect*3
            height: UIConstants.sRect
            border.width: 1
            border.color: UIConstants.grayColor
            text: qsTr("Delete")
            isEnable: selectedSetting.includes("Setting")
            onClicked: {
                root.deleteClicked(selectedSetting);
                selectedSetting = "";
            }
        }
    }
}

/*##^## Designer {
    D{i:4;anchors_height:418;anchors_width:518;anchors_x:8;anchors_y:8}D{i:2;anchors_y:8}
}
 ##^##*/
