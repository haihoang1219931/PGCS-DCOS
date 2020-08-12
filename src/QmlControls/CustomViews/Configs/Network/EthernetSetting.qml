import QtQuick 2.0
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0
Item {
    width: 630
    height: 320
    Column {
        id: column
        spacing: 5
        anchors.fill: parent
        anchors.margins: 8
        Item {
            id: row
            width: parent.width
            height: UIConstants.sRect * 1.5

            QLabel {
                id: qLabel
                width: UIConstants.sRect * 8
                height: parent.height
                horizontalAlignment: Text.AlignLeft
                anchors.verticalCenter: parent.verticalCenter
                border.width: 0
                text: "Device:"
            }

            QTextInput {
                id: qTextInput
                height: parent.height
                anchors.left: qLabel.right
                anchors.leftMargin: 8
                anchors.right: parent.right
                anchors.rightMargin: 8
                anchors.verticalCenter: parent.verticalCenter
            }
        }

        Item {
            id: row1
            width: parent.width
            height: UIConstants.sRect * 1.5
            QLabel {
                id: qLabel1
                width: UIConstants.sRect * 8
                height: parent.height
                border.width: 0
                horizontalAlignment: Text.AlignLeft
                anchors.verticalCenter: parent.verticalCenter
                text: "Cloned MAC Address:"
            }

            QTextInput {
                id: qTextInput1
                height: parent.height
                anchors.rightMargin: 8
                anchors.verticalCenter: parent.verticalCenter
                anchors.left: qLabel1.right
                anchors.right: parent.right
                anchors.leftMargin: 8
            }
        }

        Item {
            id: row2
            width: parent.width
            height: UIConstants.sRect * 1.5
            QLabel {
                id: qLabel2
                width: UIConstants.sRect * 8
                height: parent.height
                border.width: 0
                horizontalAlignment: Text.AlignLeft
                anchors.verticalCenter: parent.verticalCenter
                text: "MTU:"
            }

            QTextInput {
                id: qTextInput2
                height: parent.height
                anchors.rightMargin: 8
                anchors.verticalCenter: parent.verticalCenter
                anchors.left: qLabel2.right
                anchors.right: qLabel3.left
                anchors.leftMargin: 8
                Row{
                    anchors.right: parent.right
                    anchors.top: parent.top
                    anchors.bottom: parent.bottom
                    anchors.margins: 2
                    anchors.horizontalCenter: parent.horizontalCenter
                    layoutDirection: Qt.RightToLeft
                    FlatButtonIcon{
                        height: parent.height
                        width: parent.height
                        isAutoReturn: true
                        icon: UIConstants.iPlusText
                        isSolid: true
                    }
                    FlatButtonIcon{
                        height: parent.height
                        width: parent.height
                        isAutoReturn: true
                        icon: UIConstants.iMinusText
                        isSolid: true
                    }
                }
            }
            QLabel {
                id: qLabel3
                width: UIConstants.sRect * 2
                height: parent.height
                text: "bytes"
                border.width: 0
                anchors.verticalCenter: parent.verticalCenter
                anchors.right: parent.right
            }
        }

        Item{
            id: row3
            width: parent.width
            height: UIConstants.sRect * 2.5
            QLabel {
                id: qLabel4
                width: UIConstants.sRect * 8
                height: parent.height
                border.width: 0
                horizontalAlignment: Text.AlignLeft
                anchors.verticalCenter: parent.verticalCenter
                text: "Wake on Lan:"
            }
            GridView{
                id: gridWOL
                height: parent.height
                anchors.rightMargin: 8
                anchors.verticalCenter: parent.verticalCenter
                anchors.left: qLabel4.right
                anchors.right: parent.right
                anchors.leftMargin: 8
                layoutDirection: Qt.LeftToRight
                flow: GridView.FlowTopToBottom
                cellWidth: UIConstants.sRect * 4.7
                cellHeight: UIConstants.sRect

                model: ListModel{
                    ListElement{
                        name: "Default"
                        choosed: true
                        enable: true
                    }
                    ListElement{
                        name: "Ignore"
                        choosed: false
                        enable: false
                    }
                    ListElement{
                        name: "Phy"
                        choosed: false
                        enable: true
                    }
                    ListElement{
                        name: "Broadcast"
                        choosed: false
                        enable: true
                    }
                    ListElement{
                        name: "Unicast"
                        choosed: false
                        enable: true
                    }
                    ListElement{
                        name: "Arp"
                        choosed: false
                        enable: true
                    }
                    ListElement{
                        name: "Multicast"
                        choosed: false
                        enable: true
                    }
                    ListElement{
                        name: "Magic"
                        choosed: true
                        enable: true
                    }
                }
                delegate: QCheckBox{
                    width: gridWOL.cellWidth
                    height: gridWOL.cellHeight
                    text: name
                    enabled: enable
                    checked: choosed
                    onClicked: {
                        choosed=!choosed;
                    }
                }
            }
        }
        Item {
            id: row4
            width: parent.width
            height: UIConstants.sRect * 1.5

            QLabel {
                id: qLabel5
                width: UIConstants.sRect * 8
                height: parent.height
                horizontalAlignment: Text.AlignLeft
                anchors.verticalCenter: parent.verticalCenter
                border.width: 0
                text: "Wake on Lan password:"
            }

            QTextInput {
                id: qTextInput5
                height: parent.height
                anchors.left: qLabel5.right
                anchors.leftMargin: 8
                anchors.right: parent.right
                anchors.rightMargin: 8
                anchors.verticalCenter: parent.verticalCenter
            }
        }
    }

}
