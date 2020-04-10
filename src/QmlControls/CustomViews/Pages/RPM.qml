/**
 * ==============================================================================
 * @Project: FCS-Groundcontrol-based
 * @Module: PreflightCheck page
 * @Breif:
 * @Author: Hai Nguyen Hoang
 * @Date: 14/05/2019
 * @Language: QML
 * @License: (c) Viettel Aerospace Institude - Viettel Group
 * ============================================================================
 */

//------------------ Include QT libs ------------------------------------------
import QtQuick.Window 2.2
import QtQuick 2.6
import QtQuick.Controls 2.1
import QtQuick.Layouts 1.3
import QtGraphicalEffects 1.0
import QtQuick 2.0

//------------------ Include Custom modules/plugins
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0
Rectangle{
    id: root
    width: 600
    height: 600
    color: "transparent"
    Label {
        id: lblTitle1
        height: 54
        text: "Checking UAV Propulsion motor"
        wrapMode: Text.WordWrap
        anchors.right: parent.right
        anchors.rightMargin: 8
        anchors.left: parent.left
        anchors.top: parent.top
        anchors.leftMargin: 8
        anchors.topMargin: 0
        verticalAlignment: Text.AlignVCenter
        horizontalAlignment: Text.AlignLeft
        color: UIConstants.textColor
        font.pixelSize: UIConstants.fontSize
        font.family: UIConstants.appFont
    }

    Rectangle {
        id: rectangle
        x: 120
        y: 158
        width: 520
        height: 434
        color: "#00000000"
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 8
        anchors.horizontalCenter: parent.horizontalCenter

        Image {
            id: imgLogo1
            y: -226
            height: 362
            anchors.right: parent.right
            anchors.rightMargin: 0
            anchors.left: parent.left
            anchors.leftMargin: 0
            anchors.top: parent.top
            anchors.topMargin: 0
            source: "qrc:/assets/images/Shikra_5.png"
            rotation: -2
        }

        Label {
            id: label
            y: 384
            height: 50
            text: qsTr("Lift up the UAV and check Altitude parameter")
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 0
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignHCenter
            anchors.right: parent.right
            anchors.rightMargin: 0
            anchors.left: parent.left
            anchors.leftMargin: 0
            color: UIConstants.textColor
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
        }
    }

    Rectangle {
        id: rectangle1
        x: 200
        y: 60
        width: UIConstants.sRect * 14
        height: UIConstants.sRect * 2
        color: "#00000000"
        anchors.horizontalCenterOffset: 0
        anchors.horizontalCenter: parent.horizontalCenter

        Label {
            id: label1
            y: 38
            width: 92
            height: 16
            text: qsTr("Engine speed")
            verticalAlignment: Text.AlignVCenter
            anchors.left: parent.left
            anchors.leftMargin: 8
            anchors.verticalCenter: parent.verticalCenter
            color: UIConstants.textColor
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
        }

        Rectangle {
            id: rectangle2
            x: 130
            width: UIConstants.sRect * 6
            color: "#00000000"
            radius: 20
            anchors.right: parent.right
            anchors.rightMargin: 8
            anchors.top: parent.top
            anchors.topMargin: 8
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 8
            border.width: 2
            border.color: "gray"

            Label {
                id: label2
                text: vehicle? (Number(vehicle.engineSensor_2).toFixed(0).toString() + " RPM") : qsTr("0 RPM")
                verticalAlignment: Text.AlignVCenter
                horizontalAlignment: Text.AlignHCenter
                anchors.fill: parent
                color: UIConstants.textColor
                font.pixelSize: UIConstants.fontSize
                font.family: UIConstants.appFont
            }
        }
    }
}

/*##^## Designer {
    D{i:6;anchors_x:8}
}
 ##^##*/
