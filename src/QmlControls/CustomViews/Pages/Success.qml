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

    Rectangle {
        id: rectangle
        x: 120
        y: 301
        width: 520
        height: 49
        color: "#00000000"
        anchors.horizontalCenterOffset: 0
        anchors.horizontalCenter: parent.horizontalCenter

        Label {
            id: label
            y: 384
            height: 50
            text: qsTr("Preflight check success. Ready to take off.")
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 0
            color: UIConstants.grayColor
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignHCenter
            anchors.right: parent.right
            anchors.rightMargin: 0
            anchors.left: parent.left
            anchors.leftMargin: 0
        }
    }

    Rectangle {
        id: rectangle2
        x: 224
        y: 68
        width: 100
        height: width
        color: "#00000000"
        radius: 20
        anchors.horizontalCenterOffset: 0
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.top: parent.top
        anchors.topMargin: 195
        border.width: 4
        border.color: "green"

        Label {
            id: label2
            text: UIConstants.iSuccess
            font{ pixelSize: parent.width / 2; bold: true; family: ExternalFontLoader.solidFont }
            color: rectangle2.border.color
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignHCenter
            anchors.fill: parent
        }
    }
}
