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
import io.qdt.dev 1.0
/// ConfigPage
Rectangle {
    id: root
    color: UIConstants.transparentColor
    width: 1376
    height: 768
    property string type: ""
    property var itemListName:
        UIConstants.itemTextMultilanguages["CONFIGURATION"]["NETWORK"]
    signal clicked(string type, string action)
    NetworkManager{
        id: networkManager
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
                    networkManager.reload();
                }
            }
            QLabel{
                id: lblInterface
                width: UIConstants.sRect * 5
                height: UIConstants.sRect * 1.5
                text: itemListName["INTERFACES"]
                      [UIConstants.language[UIConstants.languageID]]
            }
        }

        ListView{
            id: lstInterfaceIP
            anchors.left: parent.left
            anchors.leftMargin: 10
            width: UIConstants.sRect * 15 + 4 * 5
            height: parent.height-rowRefresh.height - 5
            spacing: 5
            model: networkManager.listInterface
            clip: true
            delegate: Column{
                height: rowInterface.height + lstConnections.height
                spacing: 5
                Row{
                    id: rowInterface
                    height: UIConstants.sRect
                    spacing: 5
                    FlatIcon{
                        width: UIConstants.sRect
                        height: parent.height
                        iconSize: UIConstants.fontSize * 1.5
                        icon: bearerTypeName === "Ethernet"?UIConstants.iEthernet:
                                (bearerTypeName === "WLAN"?UIConstants.iWireless: UIConstants.iUnknown)

                        isSolid: true
                        color: UIConstants.greenColor
                    }

                    QLabel{
                        width: UIConstants.sRect * 7
                        height: parent.height
                        text: name
                    }
                    Label{
                        width: 5
                        height: parent.height
                        text:":"
                        color: UIConstants.textColor
                    }
                    QLabel{
                        width: UIConstants.sRect * 7
                        height: parent.height
                        text: address
                    }
                }
                ListView{
                    id: lstConnections
                    anchors.left: parent.left
                    anchors.leftMargin: UIConstants.sRect + rowInterface.spacing
                    width: UIConstants.sRect * 15
                    height: model.length * (UIConstants.sRect + spacing)
                    model: listNetwork
                    spacing: 5
                    clip: true
                    delegate: Row{
                        height: UIConstants.sRect
                        spacing: 5
                        FlatButtonText{
                            isAutoReturn: true
                            width: UIConstants.sRect * 7
                            height: parent.height
                            text: name
                            color: activated ?
                            UIConstants.greenColor: UIConstants.grayColor
                            onClicked: {
                                networkManager.connectNetwork(name,!activated);
                            }
                        }
                        QLabel{
                            visible: bearerTypeName === "WLAN"
                            width: UIConstants.sRect*3
                            text: frequency+"MHz"
                        }
                        QLabel{
                            visible: bearerTypeName === "WLAN"
                            width: UIConstants.sRect*3
                            text: strength + "/100"
                        }
                        FlatIcon{
                            visible: bearerTypeName === "WLAN" && hasPass
                            width: UIConstants.sRect
                            height: parent.height
                            iconSize: UIConstants.fontSize * 1.5
                            icon: UIConstants.iLock
                            isSolid: true
                            color: UIConstants.textColor
                        }
                    }
                }
            }
        }
    }
} // ConfigPage



