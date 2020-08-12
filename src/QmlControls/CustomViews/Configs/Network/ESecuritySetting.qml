import QtQuick 2.0
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0
Item {
    width: 630
    height: 320
    Column {
        id: clmSecurity
        spacing: 5
        anchors.fill: parent
        anchors.margins: 8
        QCheckBox {
            id: chkEnableAuthen
            width: parent.width
            height: UIConstants.sRect * 1.5
            text: "Use 802.1X security for this connection"
            checked: false
        }
        Item {
            id: itmAuthen
            width: parent.width
            height: UIConstants.sRect * 1.5
            enabled: chkEnableAuthen.checked
            opacity: enabled?1:0.5
            QLabel {
                id: lblAuthen
                width: UIConstants.sRect * 8
                height: parent.height
                horizontalAlignment: Text.AlignLeft
                anchors.verticalCenter: parent.verticalCenter
                border.width: 0
                text: "Authentication:"
            }

            QComboBox{
                id: cbxAuthen
                height: parent.height
                anchors.left: lblAuthen.right
                anchors.leftMargin: 8
                anchors.right: parent.right
                anchors.rightMargin: 8
                anchors.verticalCenter: parent.verticalCenter
                model: ["MD5","TLS","PWD","FAST","Tunneled TLS","Protected EAP (PEAP)"]
            }
        }
        Item {
            id: itmUser
            width: parent.width
            height: UIConstants.sRect * 1.5
            enabled: chkEnableAuthen.checked
            opacity: enabled?1:0.5
            QLabel {
                id: lblUser
                width: UIConstants.sRect * 8
                height: parent.height
                horizontalAlignment: Text.AlignLeft
                anchors.verticalCenter: parent.verticalCenter
                border.width: 0
                text: "User:"
            }

            QTextInput {
                id: txtUser
                height: parent.height
                anchors.left: lblUser.right
                anchors.leftMargin: 8
                anchors.right: parent.right
                anchors.rightMargin: 8
                anchors.verticalCenter: parent.verticalCenter
            }
        }
        Item {
            id: itmPass
            width: parent.width
            height: UIConstants.sRect * 1.5
            enabled: chkEnableAuthen.checked
            opacity: enabled?1:0.5
            QLabel {
                id: lblPass
                width: UIConstants.sRect * 8
                height: parent.height
                horizontalAlignment: Text.AlignLeft
                anchors.verticalCenter: parent.verticalCenter
                border.width: 0
                text: "Password:"
            }

            QTextInput {
                id: txtPass
                height: parent.height
                anchors.left: lblPass.right
                anchors.leftMargin: 8
                anchors.right: parent.right
                anchors.rightMargin: 8
                anchors.verticalCenter: parent.verticalCenter
            }
        }
        Item {
            id: itmShowPass
            width: parent.width
            height: UIConstants.sRect * 1.5
            enabled: chkEnableAuthen.checked
            opacity: enabled?1:0.5
            QCheckBox {
                id: chkShowPass
                width: UIConstants.sRect * 7
                height: UIConstants.sRect * 1.5
                enabled: chkEnableAuthen.checked
                anchors.left: parent.left
                anchors.leftMargin: UIConstants.sRect * 8 + 8
                text: "Show password"
                checked: false
            }
        }
    }
}
