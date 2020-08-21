import QtQuick 2.0
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0
Item {
    width: 630
    height: 320
    property var settingMap
    onSettingMapChanged: {
        if(settingMap !== undefined && settingMap["connection"]["type"].includes("wireless")){
            // method
            if(settingMap["802-11-wireless-security"]["key-mgmt"] === "wpa-psk"){
                cbxAuthen.currentIndex = 5;
            }
        }
    }
    Column {
        id: clmSecurity
        spacing: 5
        anchors.fill: parent
        anchors.margins: 8
        Item {
            id: itmAuthen
            width: parent.width
            height: UIConstants.sRect * 1.5
            opacity: enabled?1:0.5
            QLabel {
                id: lblAuthen
                width: UIConstants.sRect * 8
                height: parent.height
                horizontalAlignment: Text.AlignLeft
                anchors.verticalCenter: parent.verticalCenter
                border.width: 0
                text: "Security:"
            }

            QComboBox{
                id: cbxAuthen
                height: parent.height
                anchors.left: lblAuthen.right
                anchors.leftMargin: 8
                anchors.right: parent.right
                anchors.rightMargin: 8
                anchors.verticalCenter: parent.verticalCenter
                model: [
                    "None",
                    "WEP 40/128-bit key (Hex or ASCII)",
                    "WEP 128-bit Passphrase",
                    "LEAP",
                    "Dynamic WEP (802.1x)",
                    "WPA & WPA2 Personal",
                    "WPA & WPA2 Enterprise"]
            }
        }
        Item {
            id: itmPass
            width: parent.width
            height: UIConstants.sRect * 1.5
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
                echoMode: !chkShowPass.checked?
                    TextInput.Password:TextInput.Normal
                text: (settingMap === undefined ||
                       !settingMap["connection"]["type"].includes("wireless"))?"":
                    settingMap["802-11-wireless-security"]["psk"]

                onTextChanged:{
                    settingMap["802-11-wireless-security"]["psk"] = text;
                }

            }
        }
        Item {
            id: itmShowPass
            width: parent.width
            height: UIConstants.sRect * 1.5
            opacity: enabled?1:0.5
            QCheckBox {
                id: chkShowPass
                width: UIConstants.sRect * 7
                height: UIConstants.sRect * 1.5
                anchors.left: parent.left
                anchors.leftMargin: UIConstants.sRect * 8 + 8
                text: "Show password"
                checked: false
            }
        }
    }
}
