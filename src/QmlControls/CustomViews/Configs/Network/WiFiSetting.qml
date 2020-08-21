import QtQuick 2.0
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0
Item {
    width: 630
    height: 320
    property var settingMap
    Column {
        id: column
        spacing: 5
        anchors.fill: parent
        anchors.margins: 8
        Item {
            width: parent.width
            height: UIConstants.sRect * 1.5

            QLabel {
                id: lblSSID
                width: UIConstants.sRect * 8
                height: parent.height
                horizontalAlignment: Text.AlignLeft
                anchors.verticalCenter: parent.verticalCenter
                border.width: 0
                text: "SSID:"
            }

            QTextInput {
                id: txtSSID
                height: parent.height
                anchors.left: lblSSID.right
                anchors.leftMargin: 8
                anchors.right: parent.right
                anchors.rightMargin: 8
                anchors.verticalCenter: parent.verticalCenter
                text: (settingMap === undefined ||
                       !settingMap["connection"]["type"].includes("wireless"))?"":
                    settingMap["802-11-wireless"]["ssid"]
            }
        }

        Item {
            width: parent.width
            height: UIConstants.sRect * 1.5

            QLabel {
                id: lblMode
                width: UIConstants.sRect * 8
                height: parent.height
                horizontalAlignment: Text.AlignLeft
                anchors.verticalCenter: parent.verticalCenter
                border.width: 0
                text: "Mode:"
            }

            QComboBox {
                id: cbxMode
                height: parent.height
                anchors.left: lblMode.right
                anchors.leftMargin: 8
                anchors.right: parent.right
                anchors.rightMargin: 8
                anchors.verticalCenter: parent.verticalCenter
                model: ["Client","Hotspot","Ad-hoc"]
            }
        }

        Item {
            width: parent.width
            height: UIConstants.sRect * 1.5

            QLabel {
                id: lblBSSID
                width: UIConstants.sRect * 8
                height: parent.height
                horizontalAlignment: Text.AlignLeft
                anchors.verticalCenter: parent.verticalCenter
                border.width: 0
                text: "BSSID:"
            }

            QComboBox {
                id: txtBSSID
                height: parent.height
                anchors.left: lblBSSID.right
                anchors.leftMargin: 8
                anchors.right: parent.right
                anchors.rightMargin: 8
                anchors.verticalCenter: parent.verticalCenter
                model: (settingMap === undefined ||
                       !settingMap["connection"]["type"].includes("wireless"))?[]:
                    settingMap["802-11-wireless"]["seen-bssids"]
            }
        }

        Item {
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
                text: (settingMap === undefined ||
                       !settingMap["connection"]["type"].includes("wireless"))?"":
                    settingMap["802-11-wireless"]["mac-address"]
            }
        }

        Item {
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
    }
}
