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
import QtQuick.Controls 1.2 as OldCtrl
import QtQuick.Controls 2.4
import QtQuick.Controls.Styles 1.4
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0
import CustomViews.Configs.Network 1.0
import io.qdt.dev 1.0
/// ConfigPage
Rectangle {
    id: root
    color: UIConstants.transparentColor
    width: 1376
    height: 768
    property var itemListName:
            UIConstants.itemTextMultilanguages["CONFIGURATION"]["NETWORK"]
    NetworkManager{
        id: networkManager
        onNeedWLANPass: {
            openPassEditor();
        }
        onSettingSaved:{
            networkManager.reloadListSetting();
        }
    }
    function openPassEditor(){
        rectWLANPass.visible = true;
        txtPassWord.text = "";
        txtPassWord.focus = true;
    }
    function closePassEditor(){
        rectWLANPass.visible = false;
        txtPassWord.text = "";
        txtPassWord.focus = false;
    }

    Column{
        anchors.topMargin: 8
        anchors.left: parent.left
        anchors.leftMargin: 8
        anchors.top: parent.top
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 8
        spacing: 5
        Row{
            id: rowRefresh
            spacing: 5
            FlatButtonIcon{
                icon: UIConstants.iRefresh
                width: UIConstants.sRect * 1.5
                height: UIConstants.sRect * 1.5
                isAutoReturn: true
                isSolid: true
                onClicked: {
                    networkManager.reloadListAccess();
                }
            }
            FlatButtonText{
                id: btnNetworkList
                width: UIConstants.sRect * 8
                height: UIConstants.sRect * 1.5
                isAutoReturn: true
                text: itemListName["ACCESS_LIST"]
                      [UIConstants.language[UIConstants.languageID]]
                onClicked: {
                    stkNetwork.currentIndex = 0;
                }
            }
            FlatButtonText{
                id: btnNetworkConnections
                width: UIConstants.sRect * 8
                height: UIConstants.sRect * 1.5
                isAutoReturn: true
                text: itemListName["CONNECTIONS"]
                      [UIConstants.language[UIConstants.languageID]]
                onClicked: {
                    stkNetwork.currentIndex = 1;
                    networkManager.reloadListSetting();
                }
            }
        }
        StackLayout{
            id: stkNetwork
            anchors.left: parent.left
            anchors.leftMargin: 10
            width: UIConstants.sRect * 21.5
            height: parent.height-rowRefresh.height - 5
            clip: true
            NetworkList{
                id: accessList
                model: networkManager.listAccess
                networkEnabled: networkManager.networkEnabled
                wifiEnabled: networkManager.wifiEnabled
                onConnectNetwork: {
                    networkManager.connectNetwork(bearerTypeName,name,activated);
                }
                onEnableNetwork: {
                    networkManager.networkEnabled = enable;
                }
                onEnableWiFi: {
                    networkManager.wifiEnabled = enable;
                }
            }
            NetworkConnections{
                id: networkConnections
                model: networkManager.listSetting
                onDeleteClicked: {
                    networkManager.deleteSetting(selectedSetting);
                }
                onEditClicked: {
                    if(!connetionSetting.visible){
                        connetionSetting.visible = true;
                        connetionSetting.x = root.width/2 - connetionSetting.width/2
                        connetionSetting.y = root.height/2 - connetionSetting.height/2
                    }
                    if(selectedSetting.includes("/org/freedesktop/NetworkManager/Settings"))
                        connetionSetting.loadSetting(selectedSetting,networkManager.getConnectionSetting(selectedSetting));
                }
            }
        }

    }
    FlatRectangle{
        id: rectWLANPass
        visible: false
        anchors.centerIn: parent
        width: UIConstants.sRect * 11
        height: UIConstants.sRect * 6
        Column{
            anchors.verticalCenter: parent.verticalCenter
            anchors.horizontalCenter: parent.horizontalCenter
            spacing: 5
            Row{
                spacing: 5
                QLabel{
                    width: UIConstants.sRect*4
                    height: UIConstants.sRect*2
                    text: "Password"
                }
                QTextInput{
                    id: txtPassWord
                    width: UIConstants.sRect*6
                    height: UIConstants.sRect*2
                    text: ""
                }
            }
            Row{
                id: row
                spacing: UIConstants.sRect*2
                FlatButtonIcon{
                    width: UIConstants.sRect*4
                    height: UIConstants.sRect*2
                    isSolid: true
                    isShowRect: false
                    isAutoReturn: true
                    color: UIConstants.redColor
                    icon: UIConstants.iClose
                    onClicked: {
                        closePassEditor();
                    }
                }
                FlatButtonIcon{
                    width: UIConstants.sRect*4
                    height: UIConstants.sRect*2
                    isSolid: true
                    isShowRect: false
                    isAutoReturn: true
                    color: UIConstants.greenColor
                    icon: UIConstants.iChecked
                    onClicked: {
                        networkManager.insertWLANPass(txtPassWord.text);
                        closePassEditor();
                    }
                }
            }
        }
    }
    ConnectionSetting{
        id: connetionSetting
        visible: false
        onSaveClicked: {
            connetionSetting.visible = false;
            networkManager.saveSetting(selectedSetting,settingMap);
        }
        onCancelClicked: {
            connetionSetting.visible = false;
        }
    }
} // ConfigPage



