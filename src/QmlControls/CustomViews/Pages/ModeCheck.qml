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

    Label {
        id: lblTitle
        height: 54
        text: "Check Flight Mode"
        wrapMode: Text.WordWrap
        anchors.right: imgLogo.left
        anchors.rightMargin: 6

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
    Image {
        id: imgLogo
        x: 379
        width: 446
        height: 285
        anchors.top: parent.top
        anchors.topMargin: 8
        anchors.right: parent.right
        anchors.rightMargin: 8
        source: "qrc:/assets/images/Shikra_1.png"
    }
    Label {
        id: lblModeTiltle
        text: "Test Mode"
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 87
        anchors.horizontalCenter: parent.horizontalCenter
        verticalAlignment: Text.AlignVCenter
        horizontalAlignment: Text.AlignHCenter
        color: UIConstants.textColor
        font.pixelSize: UIConstants.fontSize
        font.family: UIConstants.appFont
    }

    Rectangle{
        x: 250
        y: 219
        color: "transparent"
        width: UIConstants.sRect * 8
        height: UIConstants.sRect
        border.color: "gray"
        border.width: 2
        radius: 10
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 51
        anchors.horizontalCenterOffset: 1
        anchors.horizontalCenter: parent.horizontalCenter

        Label {
            id: lblModeCurrent
            text: vehicle.flightMode
            anchors.fill: parent
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignHCenter
            color: UIConstants.textColor
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
        }
    }

}
