import QtQuick 2.0
import QtQuick 2.2
import QtQuick.Layouts 1.1
import QtQuick.Controls 1.2
import QtQuick.Controls.Styles 1.3
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0
Item {
    width: 630
    height: 320
    property var settingMap
    Column {
        spacing: 5
        anchors.fill: parent
        anchors.margins: 8
        Item {
            id: itmMethod
            width: parent.width
            height: UIConstants.sRect * 1.5
            QLabel {
                id: lblMethod
                width: UIConstants.sRect * 5
                height: parent.height
                horizontalAlignment: Text.AlignLeft
                anchors.verticalCenter: parent.verticalCenter
                border.width: 0
                text: "Method:"
            }

            QComboBox{
                id: cbxMethod
                height: parent.height
                anchors.left: lblMethod.right
                anchors.leftMargin: 8
                anchors.right: parent.right
                anchors.rightMargin: 8
                anchors.verticalCenter: parent.verticalCenter
                model: [
                    "Automatic (DHCP)",
                    "Automatic (DHCP) address only",
                    "Manual",
                    "Link-Local Only",
                    "Shared to other computer",
                    "Disabled"]
            }
        }
        Item{
            id: itmAddress
            width: parent.width
            height: parent.height - parent.spacing - itmMethod.height
            opacity: enabled?1:0.5
            QLabel{
                id: lblAddress
                width: parent.width
                height: UIConstants.sRect
                horizontalAlignment: Text.AlignLeft
                text: "Addresses"
                border.width: 0
            }

            Column{
                anchors.rightMargin: 10
                anchors.left: parent.left
                anchors.bottom: parent.bottom
                anchors.right: parent.right
                anchors.top: lblAddress.bottom
                anchors.topMargin: 5
                anchors.leftMargin: 10
                spacing: 5
                Row{
                    width: parent.width
                    height: parent.height -
                            UIConstants.sRect * 1.5 *5 -
                            parent.spacing*4
                    spacing: 5
                    Rectangle{
                        height: parent.height
                        width: parent.width - UIConstants.sRect*4 - parent.spacing
                        color: UIConstants.transparentColor
                        border.width: 1
                        border.color: UIConstants.grayColor
                        clip: true

                        TableView{
                            id: tblAdresses
                            width: parent.width
                            height: parent.height
                            frameVisible: false
                            sortIndicatorVisible: true
                            backgroundVisible: true
                            alternatingRowColors: false
                            headerDelegate: QLabel{
                                text: " "+styleData.value
                                horizontalAlignment: Label.AlignLeft
                                radius: 0
                                color: UIConstants.grayColor
                            }

                            TableViewColumn {
                                id: clmAddress
                                title: "Address"
                                role: "address"
                                movable: false
                                resizable: false
                                width: tblAdresses.viewport.width -
                                       clmPrefix.width -
                                       clmGateway.width
                            }
                            TableViewColumn {
                                id: clmPrefix
                                title: "Prefix"
                                role: "prefix"
                                movable: false
                                resizable: false
                                width: tblAdresses.viewport.width / 3
                            }
                            TableViewColumn {
                                id: clmGateway
                                title: "Gateway"
                                role: "gateway"
                                movable: false
                                resizable: false
                                width: tblAdresses.viewport.width / 3
                            }
                            model:ListModel {
                                id: sourceModel
                                ListElement {
                                    address: "192.168.100.230"
                                    prefix: "255.255.255.0"
                                    gateway: "192.168.100.1"
                                }
                                ListElement {
                                    address: "192.168.100.231"
                                    prefix: "255.255.255.0"
                                    gateway: "192.168.100.1"
                                }
                            }
                            itemDelegate: TextField{
                                textColor: UIConstants.textColor
                                text: styleData.value
                                horizontalAlignment: Text.AlignLeft
                                verticalAlignment: Text.AlignVCenter
                                font.family: UIConstants.appFont
                                font.pixelSize: UIConstants.fontSize
                                style: TextFieldStyle {
                                    background: Rectangle {
                                        color: UIConstants.bgAppColor
                                        border.color: UIConstants.grayColor
                                        border.width: 1
                                    }
                                }
                            }
                            style: TableViewStyle{
                                backgroundColor: UIConstants.transparentColor
                            }
                        }
                    }

                    Column{
                        width: UIConstants.sRect*4
                        height: parent.height
                        spacing: 5
                        FlatButtonText{
                            width: parent.width
                            height: UIConstants.sRect * 1.5
                            text: "Add"
                            isAutoReturn: true
                            border.width: 1
                            border.color: UIConstants.grayColor
                        }
                        FlatButtonText{
                            width: parent.width
                            height: UIConstants.sRect * 1.5
                            text: "Delete"
                            isAutoReturn: true
                            border.width: 1
                            border.color: UIConstants.grayColor
                        }
                    }
                }



                Row{
                    width: parent.width
                    height: UIConstants.sRect * 1.5
                    QLabel {
                        width: UIConstants.sRect * 10
                        height: parent.height
                        horizontalAlignment: Text.AlignLeft
                        border.width: 0
                        text: "Additional DNS servers:"
                    }

                    QTextInput {
                        id: txtDNSServers
                        height: parent.height
                        width: parent.width - UIConstants.sRect * 10
                    }
                }
                Row{
                    width: parent.width
                    height: UIConstants.sRect * 1.5
                    QLabel {
                        width: UIConstants.sRect * 10
                        height: parent.height
                        horizontalAlignment: Text.AlignLeft
                        border.width: 0
                        text: "Additional search domains:"
                    }

                    QTextInput {
                        id: txtSearchDomain
                        height: parent.height
                        width: parent.width - UIConstants.sRect * 10
                    }
                }
                Row{
                    width: parent.width
                    height: UIConstants.sRect * 1.5
                    QLabel {
                        width: UIConstants.sRect * 10
                        height: parent.height
                        horizontalAlignment: Text.AlignLeft
                        border.width: 0
                        text: "IPv6 privacy extentions:"
                    }

                    QComboBox {
                        id: txtDHCPClientID
                        height: parent.height
                        width: parent.width - UIConstants.sRect * 10
                        model: [
                            "Disabled",
                            "Enabled (prefer public address)",
                            "Enabled (prefer temporary address)"
                        ]
                    }
                }
                QCheckBox{
                    id: cbkIPV6
                    width: parent.width
                    height: UIConstants.sRect * 1.5
                    text: "Require IPv6 addressing for this connection to complete"
                }
                Row{
                    width: parent.width
                    height: UIConstants.sRect * 1.5
                    layoutDirection: Qt.RightToLeft
                    FlatButtonText{
                        isAutoReturn: true
                        text: "Routes..."
                        width: UIConstants.sRect*4
                        height: parent.height
                        border.width: 1
                        border.color: UIConstants.grayColor
                    }
                }
            }
        }
    }
}
