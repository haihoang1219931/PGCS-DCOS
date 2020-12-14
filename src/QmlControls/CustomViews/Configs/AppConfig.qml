/****************************************************************************
 *
 *   (c) 2009-2016 QGROUNDCONTROL PROJECT <http://www.qgroundcontrol.org>
 *
 * QGroundControl is licensed according to the terms in the file
 * COPYING.md in the root of the source code directory.
 *
 ****************************************************************************/


import QtQuick          2.3
import QtQuick.Dialogs  1.2
import QtQuick.Layouts 1.3
import QtQuick.Window 2.11
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0
import QtQuick.Controls 1.2 as OldCtrl
import QtQuick.Controls 2.4
import QtQuick.Controls.Styles 1.4
/// ConfigPage
Rectangle {
    id: root
    color: UIConstants.transparentColor
    width: 1376
    height: 768
    property string type: ""
    property var itemListName:
        UIConstants.itemTextMultilanguages["CONFIGURATION"]["APPLICATION"]
    signal clicked(string type, string action)
    Column{
        id: clmLanguage
        anchors.topMargin: 8
        anchors.left: parent.left
        anchors.leftMargin: 8
        anchors.top: parent.top
        width: UIConstants.sRect* 15
        spacing:    UIConstants.sRect/2
        Row{
            Label {
                text: itemListName["LANGUAGE"]
                      [UIConstants.language[UIConstants.languageID]]
                color: UIConstants.textColor
                font.pixelSize: UIConstants.fontSize
                font.family: UIConstants.appFont
            }
        }
        Row{
            spacing: UIConstants.sRect/2
            QComboBox{
                id: cbxListLanguage
                width: clmLanguage.width - btnSelectLanguage.width - parent.spacing
                height: UIConstants.sRect * 1.5
                model: ["English","Việt Nam"]
            }
            OldCtrl.Button{
                id: btnSelectLanguage
                width: UIConstants.sRect * 4
                height: UIConstants.sRect * 1.5
                text: itemListName["SELECT"]
                      [UIConstants.language[UIConstants.languageID]]
                style: ButtonStyle{
                    background: Rectangle{
                        color: UIConstants.info
                    }
                    label: Label{
                        color: UIConstants.textColor
                        font.pixelSize: UIConstants.fontSize
                        font.family: UIConstants.appFont
                        verticalAlignment: Label.AlignVCenter
                        horizontalAlignment: Label.AlignHCenter
                        text: btnSelectLanguage.text
                    }
                }

                onClicked: {
                    if(cbxListLanguage.currentIndex === 0){
                        UIConstants.languageID = "EN";
                    }else if(cbxListLanguage.currentIndex === 1){
                        UIConstants.languageID = "VI";
                    }
                    ApplicationConfig.changeData("Language",UIConstants.languageID);
                }
            }
        }
        FlatButton {
            id: btnMissionFolder
            btnText: itemListName["MISSION_FOLDER"][UIConstants.language[UIConstants.languageID]]
            btnTextColor: UIConstants.textFooterColor
            height: UIConstants.sRect * 2
            width: UIConstants.sRect * 5
            iconVisible: true
            icon: UIConstants.iOpenFolder
            color: UIConstants.sidebarActiveBg
            radius: UIConstants.rectRadius
            onClicked: {
                computer.openFolder("missions")
            }
        }
        FlatButton {
            id: btnLogFolder
            btnText: itemListName["LOGS_FOLDER"][UIConstants.language[UIConstants.languageID]]
            btnTextColor: UIConstants.textFooterColor
            height: UIConstants.sRect * 2
            width: UIConstants.sRect * 5
            iconVisible: true
            icon: UIConstants.iOpenFolder
            color: UIConstants.sidebarActiveBg
            radius: UIConstants.rectRadius
            onClicked: {
                computer.openFolder("logs")
            }
        }
        FlatButton {
            id: btnVideoFolder
            visible: USE_VIDEO_CPU || USE_VIDEO_GPU
            btnText: itemListName["VIDEO_FOLDER"][UIConstants.language[UIConstants.languageID]]
            btnTextColor: UIConstants.textFooterColor
            height: UIConstants.sRect * 2
            width: UIConstants.sRect * 5
            iconVisible: true
            icon: UIConstants.iOpenFolder
            color: UIConstants.sidebarActiveBg
            radius: UIConstants.rectRadius
            onClicked: {
                computer.openFolder("flights")
            }
        }
    }
    ColumnLayout{
        anchors.topMargin: 8
        anchors.right: parent.right
        anchors.rightMargin: 8
        anchors.top: parent.top
        spacing: 3
        layoutDirection: Qt.RightToLeft
        RowLayout{
            Layout.preferredHeight: UIConstants.sRect * 2
            Layout.fillWidth: true
            layoutDirection: Qt.RightToLeft
            FlatButtonIcon{
                Layout.preferredWidth: UIConstants.sRect * 2
                Layout.preferredHeight: UIConstants.sRect * 2
                width: UIConstants.sRect * 2
                height: width
                icon: mainWindow.visibility === ApplicationWindow.FullScreen?
                          UIConstants.iWindowsMinimize:UIConstants.iWindowStore
                isSolid: true
                isShowRect: false
                iconSize: UIConstants.sRect*3/2 - 5
                iconColor: UIConstants.textColor
                color: UIConstants.transparentBlue
                radius: UIConstants.rectRadius
                Layout.alignment: Qt.AlignRight | Qt.AlignVCenter
                isAutoReturn: true
                onClicked: {
                    if(mainWindow.visibility === ApplicationWindow.FullScreen){                        
                        mainWindow.visibility = ApplicationWindow.Windowed;
                        mainWindow.width = Screen.width/2;
                        mainWindow.height = Screen.height/2;
                    }else{
                        mainWindow.visibility = ApplicationWindow.FullScreen;
                    }
                }
            }
            Label{
                font.family: UIConstants.appFont
                font.pixelSize: UIConstants.fontSize
                text: mainWindow.visibility === ApplicationWindow.FullScreen?
                          itemListName["WINDOWS_NORMAL"]
                            [UIConstants.language[UIConstants.languageID]]:
                          itemListName["WINDOWS_FULL"]
                            [UIConstants.language[UIConstants.languageID]]
                color: UIConstants.textColor
            }
        }
        RowLayout{
            Layout.preferredHeight: UIConstants.sRect * 2
            Layout.fillWidth: true
            layoutDirection: Qt.RightToLeft
            FlatButtonIcon{
                Layout.preferredWidth: UIConstants.sRect * 2
                Layout.preferredHeight: UIConstants.sRect * 2
                width: UIConstants.sRect * 2
                height: width
                icon: UIConstants.iClose
                isSolid: true
                isShowRect: false
                iconSize: UIConstants.sRect*3/2 - 5
                iconColor: UIConstants.textColor
                color: UIConstants.redColor
                radius: UIConstants.rectRadius
                Layout.alignment: Qt.AlignRight | Qt.AlignVCenter
                isAutoReturn: true
                onClicked: {
                    root.clicked(root.type,"QUIT_APP");
                }
            }
            Label{
                font.family: UIConstants.appFont
                font.pixelSize: UIConstants.fontSize
                text: itemListName["QUIT_APP"]
                      [UIConstants.language[UIConstants.languageID]]
                color: UIConstants.textColor
            }
        }
        RowLayout{
            Layout.preferredHeight: UIConstants.sRect * 2
            Layout.fillWidth: true
            layoutDirection: Qt.RightToLeft
            FlatButtonIcon{
                Layout.preferredWidth: UIConstants.sRect * 2
                Layout.preferredHeight: UIConstants.sRect * 2
                width: UIConstants.sRect * 2
                height: width
                icon: UIConstants.iRefresh
                isSolid: true
                isShowRect: false
                iconSize: UIConstants.sRect*3/2 - 5
                iconColor: UIConstants.textColor
                color: UIConstants.redColor
                radius: UIConstants.rectRadius
                Layout.alignment: Qt.AlignRight | Qt.AlignVCenter
                isAutoReturn: true
                onClicked: {
                    root.clicked(root.type,"RESTART_APP");
                }
            }
            Label{
                font.family: UIConstants.appFont
                font.pixelSize: UIConstants.fontSize
                text: itemListName["RESTART_APP"]
                      [UIConstants.language[UIConstants.languageID]]
                color: UIConstants.textColor
            }
        }
        RowLayout{
            Layout.fillWidth: true
            Layout.preferredHeight: UIConstants.sRect * 2
            layoutDirection: Qt.RightToLeft
            FlatButtonIcon{
                Layout.preferredWidth: UIConstants.sRect * 2
                Layout.preferredHeight: UIConstants.sRect * 2
                width: UIConstants.sRect * 2
                height: width
                icon: UIConstants.iPowerOff
                isSolid: true
                isShowRect: false
                iconSize: UIConstants.sRect*3/2 - 5
                iconColor: UIConstants.textColor
                color: UIConstants.redColor
                radius: UIConstants.rectRadius
                Layout.alignment: Qt.AlignRight | Qt.AlignVCenter
                isAutoReturn: true
                onClicked: {
                    root.clicked(root.type,"QUIT_COM");
                }
            }
            Label{
                font.family: UIConstants.appFont
                font.pixelSize: UIConstants.fontSize
                text: itemListName["SHUTDOWN_COM"]
                      [UIConstants.language[UIConstants.languageID]]
                color: UIConstants.textColor
            }
        }
    }


} // ConfigPage



