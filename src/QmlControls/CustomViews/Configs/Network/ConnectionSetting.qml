import QtQuick 2.0
import QtQuick.Controls 2.4
import QtQuick.Layouts 1.3

import QtQuick 2.3
import QtQuick.Controls 1.2
import QtQuick.Controls.Styles 1.4
import QtQuick.Dialogs 1.2
import QtQuick.Layouts 1.1
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0
Rectangle {
    id: root
    width: UIConstants.sRect*29
    height: UIConstants.sRect*21.5
    color: UIConstants.bgAppColor
    border.color: UIConstants.grayColor
    border.width: 1
    property string deviceType: "Ethernet"
    property var selectedSetting
    property var settingMap: undefined
    signal saveClicked(string selectedSetting,var setting)
    signal cancelClicked(string selectedSetting,var setting)

    function loadSetting(selectedSetting,setting){
        root.selectedSetting = selectedSetting;
        settingMap=setting;
        if(setting["connection"]["type"].includes("wireless")){
            deviceType = "Wi-Fi";
        }else if(setting["connection"]["type"].includes("bridge")){
            deviceType = "Bridge";
        }else if(setting["connection"]["type"].includes("ethernet")){
            deviceType = "Ethernet";
        }else{
            deviceType = "Unknown";
        }
    }
    MouseArea{
        anchors.fill: parent
        hoverEnabled: true
        drag.target: parent
    }

    Column{
        anchors.bottomMargin: 5
        anchors.bottom: itmButtons.top
        anchors.right: parent.right
        anchors.rightMargin: 5
        anchors.left: parent.left
        anchors.leftMargin: 5
        anchors.top: parent.top
        anchors.topMargin: 5
        spacing: 5
        Item{
            id: itmName
            height: UIConstants.sRect*1.5
            width: parent.width
            QLabel{
                id: lblName
                height: parent.height
                text: "Connection name"
                anchors.left: parent.left
                anchors.leftMargin: 0
                width: UIConstants.sRect*6
                border.width: 0
            }
            QTextInput{
                id: txtName
                anchors.right: parent.right
                height: parent.height
                anchors.left: lblName.right
                anchors.leftMargin: 5
                anchors.rightMargin: 5
                text: settingMap !== undefined?settingMap["connection"]["id"]:""
                onTextChanged: {
                    if(settingMap !== undefined)
                        settingMap["connection"]["id"] = text;
                }
            }
        }
        QTabBar{
            id: barSettings
            width: parent.width
            currentIndex: 0
            height: UIConstants.sRect*1.5
            contentHeight: UIConstants.sRect*1.5-1
            QTabButton{
                text: "General"
                width: visible?implicitWidth:-1
                visible: true
            }
            QTabButton{
                text: "Wi-Fi"
                width: visible?implicitWidth:-1
                visible: false
            }
            QTabButton{
                text: "Wi-Fi Security"
                width: visible?implicitWidth:-1
                visible: deviceType === "Wi-Fi"
            }
            QTabButton{
                text: "Ethernet"
                width: visible?implicitWidth:-1
                visible: false
            }
            QTabButton{
                text: "802.1x Security"
                width: visible?implicitWidth:-1
                visible: false
            }
            QTabButton{
                text: "DCB"
                width: -1
                visible: false
            }
            QTabButton{
                text: "IPv4 Settings"
            }
            QTabButton{
                text: "IPv6 Settings"
                width: -1
                visible: false
            }
        }
        Rectangle{
            width: parent.width
            height: parent.height - itmName.height - barSettings.height - parent.spacing * 2
            color: UIConstants.transparentColor
            border.color: UIConstants.grayColor
            border.width: 1
            StackLayout{
                id: stkSetting
                anchors.fill: parent
                currentIndex: barSettings.currentIndex
                Item {
                    id: tabGeneral
                    GeneralSetting{
                        anchors.fill: parent
                    }
                }
                Item {
                    id: tabWifi
                    WiFiSetting{
                        anchors.fill: parent
                        settingMap: root.settingMap

                    }
                }
                Item {
                    id: tabWifiSecurity
                    WiFiSecurity{
                        anchors.fill: parent
                        settingMap: root.settingMap
                    }
                }
                Item {
                    id: tabEthernet
                    EthernetSetting{
                        anchors.fill: parent
                    }
                }
                Item {
                    id: tabSecurity
                    ESecuritySetting{
                        anchors.fill: parent
                    }
                }
                Item {
                    id: tabDCB
                    DCBSetting{
                        anchors.fill: parent
                    }
                }
                Item {
                    id: tabIPv4
                    IPV4Setting{
                        anchors.fill: parent
                        settingMap: root.settingMap
                    }
                }
                Item {
                    id: tabIPv6
                    IPV6Setting{
                        anchors.fill: parent
                        settingMap: root.settingMap
                    }
                }
            }
        }
    }
    Item{
        id: itmButtons
        anchors.bottom: parent.bottom
        anchors.right: parent.right
        height: UIConstants.sRect*1.5
        anchors.rightMargin: 5
        anchors.bottomMargin: 5
        width: parent.width
        Row{
            spacing: UIConstants.sRect
            anchors.fill: parent
            layoutDirection: Qt.RightToLeft
            FlatButtonText{
                width: parent.height * 2
                height: parent.height
                isAutoReturn: true
                border.width: 1
                border.color: UIConstants.grayColor
                text: "Save"
                onClicked: {
                    root.saveClicked(root.selectedSetting,root.settingMap);
                }
            }
            FlatButtonText{
                width: parent.height * 2
                height: parent.height
                isAutoReturn: true
                border.width: 1
                border.color: UIConstants.grayColor
                text: "Cancel"
                onClicked: {
                    root.cancelClicked(root.selectedSetting,root.settingMap);
                }
            }
        }


    }
}

/*##^## Designer {
    D{i:19;anchors_width:614}D{i:35;anchors_height:400;anchors_width:200}D{i:33;anchors_width:614;anchors_x:8}
}
 ##^##*/
