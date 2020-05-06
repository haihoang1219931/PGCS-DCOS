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

//------------------ Include QGroundControl libs

//------------------ Include Custom modules/plugins
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0
Rectangle{
    id: root
    width: 600
    height: 300
    color: "transparent"

    QLabel {
        id: lblTitle
        height: 54
        text: "Check Flight Mode"
        anchors.right: parent.right
        anchors.rightMargin: 8
        anchors.left: parent.left
        anchors.top: parent.top
        anchors.leftMargin: 8
        anchors.topMargin: 0
        border.width: 0
    }
    Column{
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottom: parent.bottom
        anchors.bottomMargin: UIConstants.sRect
        spacing: 8
        QLabel {
            id: lblModeCommandTiltle
            text: "Command Mode"
            anchors.horizontalCenter: parent.horizontalCenter
            border.width: 0
            clip: false
        }

        QTextInput{
            width: UIConstants.sRect * 8
            height: UIConstants.sRect
            anchors.horizontalCenter: parent.horizontalCenter
            enabled: false
            horizontalAlignment: TextInput.AlignHCenter
            text: vehicle.pic ? "PIC":"CIC"
        }
        QLabel {
            id: lblModeTiltle
            anchors.horizontalCenter: parent.horizontalCenter
            text: "Test Mode"
            border.width: 0
            clip: false
        }

        QTextInput{
            width: UIConstants.sRect * 8
            height: UIConstants.sRect
            anchors.horizontalCenter: parent.horizontalCenter
            text: vehicle.flightMode
            horizontalAlignment: TextInput.AlignHCenter
            enabled: false
        }
    }
}
