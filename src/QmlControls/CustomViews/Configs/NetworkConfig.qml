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
        spacing: 5
        Row{
            FlatButtonIcon{
                icon: UIConstants.iRefresh
                width: UIConstants.sRect * 2
                height: UIConstants.sRect * 2
                isAutoReturn: true
                isSolid: true
                onClicked: {
                    networkManager.reload();
                }
            }
        }
        QLabel{
            width: UIConstants.sRect * 5
            height: UIConstants.sRect * 1.5
            text: itemListName["INTERFACES"]
                  [UIConstants.language[UIConstants.languageID]]
        }

        ListView{
            id: lstInterfaceIP
            anchors.left: parent.left
            anchors.leftMargin: 10
            width: UIConstants.sRect * 13
            height: model.length * (UIConstants.sRect + spacing)
            spacing: 5
            model: networkManager.listInterface
            clip: true
            delegate: Row{
                height: UIConstants.sRect
                spacing: 5
                QLabel{
                    width: UIConstants.sRect * 5
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
        }
        QLabel{
            width: UIConstants.sRect * 5
            height: UIConstants.sRect * 1.5
            text: itemListName["CONNECTIONS"]
                  [UIConstants.language[UIConstants.languageID]]
        }

        ListView{
            id: lstConnections
            anchors.left: parent.left
            anchors.leftMargin: 10
            width: UIConstants.sRect * 13
            height: model.length * (UIConstants.sRect + spacing)
            model: networkManager.listNetwork
            spacing: 5
            clip: true
            delegate: Row{
                height: UIConstants.sRect
                spacing: 5
                QLabel{
                    width: UIConstants.sRect * 5
                    height: parent.height
                    text: bearerTypeName
                    color: activated?UIConstants.greenColor:UIConstants.grayColor
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
                    text: name
                }
            }
        }
    }
} // ConfigPage



