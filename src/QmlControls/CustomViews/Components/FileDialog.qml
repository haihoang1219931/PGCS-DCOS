import QtQuick 2.2
import QtQuick.Dialogs 1.3
import QtQuick.Controls 2.4
import Qt.labs.platform 1.0
import Qt.labs.folderlistmodel 2.1

//------------------ Include Custom modules/plugins
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0

Rectangle {
    id: root
    width: UIConstants.sRect*10
    height: UIConstants.sRect*15
    color: UIConstants.transparentBlue
    border.color: UIConstants.grayColor
    radius: UIConstants.rectRadius
    signal clicked(string type,string func)
    property alias nameFilters: folderModel.nameFilters
    property string folder
    property var fileMode
    property string currentFile: ""
    signal fileSelected(string file)
    signal addFileList(var lstFile)
    function selectFile(file) {
        if (file !== ""){
            fileSelected(file)
        }
    }
    Label {
        id: lblTitle
        y: 8
        height: UIConstants.sRect
        text: qsTr("Label")
        color: UIConstants.textColor
        font.pixelSize: UIConstants.fontSize
        font.family: UIConstants.appFont
        verticalAlignment: Text.AlignVCenter
        horizontalAlignment: Text.AlignHCenter
        anchors.horizontalCenterOffset: 0
        anchors.horizontalCenter: parent.horizontalCenter
    }

    Component {
        id: fileDelegate
        Item {
            height: 40
            width: listView.width
            property variant folders: folderModel
            function launch() {
                var path = "file://";
                if (filePath.length > 2 && filePath[1] === ':') // Windows drive logic, see QUrl::fromLocalFile()
                    path += '/';
                path += filePath;
                if (folders.isFolder(index)){
                    folders.folder = path;
//                            console.log("Load folder: "+folders.folder);
                }else{
                    selectFile(path)
                }
            }

            Label {
                id: txtName
                anchors.left: parent.left;
                anchors.right: parent.right
                verticalAlignment: Label.AlignLeft
                text: fileName
                color: UIConstants.textColor
                font.pixelSize: UIConstants.fontSize
                font.family: UIConstants.appFont
            }

            MouseArea{
                hoverEnabled: true
                anchors.left: parent.left;
                anchors.right: parent.right
                anchors.bottom: parent.bottom
                anchors.top: parent.top
                onEntered: {
                    rect.color = colorHover;
                }
                onExited: {
                    rect.color = colorNormal;
                }
                onClicked: {
                    currentFile = txtName.text
                    fileSelected(currentFile);
                    console.log("fileSelected "+currentFile)
                }
            }
        }
    }
    FolderListModel {
        id: folderModel
        showDirs: true
        showDotAndDotDot: false
        folder: "file://"+root.folder
    }
    ListView{
        id: listView
        anchors.rightMargin: 8
        anchors.leftMargin: 8
        anchors.bottomMargin: 8
        anchors.top: lblTitle.bottom
        anchors.right: parent.right
        anchors.bottom: btnConfirm.top
        anchors.left: parent.left
        anchors.topMargin: 8
        clip: true
        model: folderModel
        delegate: fileDelegate
    }

    FlatButtonIcon{
        id: btnConfirm
        height: 30
        width: 60
        icon: UIConstants.iChecked
        isSolid: true
        color: "green"
        anchors.left: parent.left
        anchors.leftMargin: 8
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 8
        isAutoReturn: true
        radius: root.radius
        onClicked: {
            root.clicked(root.type,"DIALOG_OK");
            root.destroy();
        }
    }
    FlatButtonIcon{
        id: btnCancel
        height: 30
        width: 60
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
            root.destroy();
        }
    }

}
