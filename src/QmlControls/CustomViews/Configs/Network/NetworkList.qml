import QtQuick 2.0
import QtQuick.Controls 2.4
import QtQuick.Layouts 1.3
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0
Rectangle{
    id: root    
    color: UIConstants.transparentColor
    border.color: UIConstants.grayColor
    border.width: 1
    clip: true
    property alias model: listInterface.model
    property bool networkEnabled: true
    property bool wifiEnabled: true
    signal connectNetwork(string bearerTypeName,string name,bool activated)
    signal enableNetwork(bool enable)
    signal enableWiFi(bool enable)
    Column{
        anchors.fill: parent
        anchors.margins: 8
        spacing: 5
        FlatButtonText{
            id: btnEnableNetwork
            isAutoReturn: true
            width: UIConstants.sRect * 16 + 5
            height: UIConstants.sRect
            text: "Enable Network"
            color: networkEnabled?UIConstants.greenColor:UIConstants.grayColor
            border.color: UIConstants.grayColor
            horizontalAlignment: Text.AlignLeft
            clip: true
            onClicked: {
                root.enableNetwork(!networkEnabled)
            }
        }
        FlatButtonText{
            id: btnEnableWifi
            isAutoReturn: true
            width: UIConstants.sRect * 16 + 5
            height: UIConstants.sRect
            text: "Enable Wi-Fi"
            color: wifiEnabled?UIConstants.greenColor:UIConstants.grayColor
            border.color: UIConstants.grayColor
            horizontalAlignment: Text.AlignLeft
            clip: true
            onClicked: {
                root.enableWiFi(!wifiEnabled)
            }
        }

        Canvas {
            id: cvsSeperateNetwork
            width: parent.width
            height: 1
            onPaint: {
                var ctx = getContext("2d");
                ctx.reset();
                ctx.strokeStyle = "white";
                ctx.lineWidth = 1;
                ctx.moveTo(0,0);
                ctx.lineTo(width,0)
                ctx.stroke();
            }
        }
        ListView{
            id: listInterface
            width: parent.width
            height: parent.height -
                    btnEnableNetwork.height -
                    btnEnableWifi.height -
                    cvsSeperateNetwork.height -
                    parent.spacing * 3
            spacing: 5
            delegate: Column{
                height: rowInterface.height +
                        (lstConnections.visible?lstConnections.height:0) +
                        cvsSeperateInterface.height +
                        spacing * 2
                width: parent.width
                spacing: 5
                Row{
                    id: rowInterface
                    height: UIConstants.sRect*2
                    spacing: 5
                    FlatIcon{
                        width: UIConstants.sRect
                        height: parent.height
                        iconSize: UIConstants.fontSize * 1.5
                        icon: bearerTypeName.includes("Ethernet")?UIConstants.iEthernet:
                                (bearerTypeName.includes("Wi-Fi")?UIConstants.iWireless: UIConstants.iUnknown)

                        isSolid: true
                        color: activated?
                            UIConstants.greenColor:UIConstants.grayColor
                    }

                    FlatButtonText{
                        isAutoReturn: true
                        width: UIConstants.sRect * 15
                        height: parent.height
                        text: bearerTypeName + " (" +name+ ")" + "\n" +"IP ["+address+"]"
                        color: UIConstants.grayColor
                        onClicked: {
                            lstConnections.visible=!lstConnections.visible;
                        }
                    }
                }
                ListView{
                    id: lstConnections
                    anchors.left: parent.left
                    anchors.leftMargin: UIConstants.sRect + rowInterface.spacing
                    width: UIConstants.sRect * 20
                    height: model.length * (UIConstants.sRect + spacing)
                    model: listNetwork
                    spacing: 5
                    clip: true
                    delegate: Row{
                        height: UIConstants.sRect
                        spacing: 5
                        FlatButtonText{
                            isAutoReturn: true
                            width: UIConstants.sRect * 15
                            height: parent.height
                            text: name
                            color: activated ?
                            UIConstants.greenColor: UIConstants.transparentColor
                            border.color: UIConstants.grayColor
                            horizontalAlignment: Text.AlignLeft
                            clip: true
                            onClicked: {
                                root.connectNetwork(bearerTypeName,name,!activated);
                            }
                        }
                        QLabel{
                            visible: bearerTypeName === "Wi-Fi"
                            width: UIConstants.sRect*2
                            text: Number(frequency/1000).toFixed(1)+"G"
                        }
                        FlatIcon{
                            visible: bearerTypeName === "Wi-Fi" && hasPass
                            width: UIConstants.sRect
                            height: parent.height
                            iconSize: UIConstants.fontSize * 1.5
                            icon: UIConstants.iLock
                            isSolid: true
                            color: UIConstants.textColor
                        }
                        Row{
                            visible: bearerTypeName === "Wi-Fi"
                            width: UIConstants.sRect*1+3*spacing
                            height: parent.height
                            spacing: 2
                            Rectangle{
                                anchors.bottom: parent.bottom
                                width: parent.width / 4
                                height: parent.height / 4
                                color: strength > 0? UIConstants.textColor: UIConstants.transparentColor
                                border.color: UIConstants.grayColor
                                border.width: 1
                            }
                            Rectangle{
                                anchors.bottom: parent.bottom
                                width: parent.width / 4
                                height: parent.height / 4 * 2
                                color: strength > 25? UIConstants.textColor: UIConstants.transparentColor
                                border.color: UIConstants.grayColor
                                border.width: 1
                            }
                            Rectangle{
                                anchors.bottom: parent.bottom
                                width: parent.width / 4
                                height: parent.height / 4 * 3
                                color: strength > 50? UIConstants.textColor: UIConstants.transparentColor
                                border.color: UIConstants.grayColor
                                border.width: 1
                            }
                            Rectangle{
                                anchors.bottom: parent.bottom
                                width: parent.width / 4
                                height: parent.height / 4 * 4
                                color: strength > 75? UIConstants.textColor: UIConstants.transparentColor
                                border.color: UIConstants.grayColor
                                border.width: 1
                            }
                        }
                    }
                }
                Canvas {
                    id: cvsSeperateInterface
                    width: parent.width
                    height: 1
                    onPaint: {
                        var ctx = getContext("2d");
                        ctx.reset();
                        ctx.strokeStyle = "white";
                        ctx.lineWidth = 1;
                        ctx.moveTo(0,0);
                        ctx.lineTo(width,0)
                        ctx.stroke();
                    }
                }
            }
        }
    }
}

