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
        text: "Wait untill UAV Location show bellow!\nCompare it with Other GPS device!"
        wrapMode: Text.WordWrap
        anchors.right: parent.right
        anchors.rightMargin: 8

        anchors.left: parent.left
        anchors.top: parent.top
        anchors.leftMargin: 8
        anchors.topMargin: 0
        verticalAlignment: Text.AlignVCenter
        horizontalAlignment: Text.AlignHCenter
        color: UIConstants.textColor
        font.pixelSize: UIConstants.fontSize
        font.family: UIConstants.appFont
    }

    Rectangle {
        id: rectangle
        x: 120
        y: 166
        width: 520
        height: 49
        color: "#00000000"
        anchors.horizontalCenterOffset: 0
        anchors.horizontalCenter: parent.horizontalCenter

        Label {
            id: label
            y: 384
            height: 50
            text: qsTr("Is UAV Location correct?")
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 0
            color: UIConstants.textColor
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
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
        width: 292
        height: 100
        color: "#00000000"
        radius: 20
        anchors.horizontalCenterOffset: 0
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.top: parent.top
        anchors.topMargin: 60
        border.width: 2
        border.color: "gray"

        Label {
            id: label2
            text: vehicle?("LATITUDE   "+Number(vehicle.coordinate.latitude).toFixed(7).toString()+
                          "\nLONGITUDE  "+Number(vehicle.coordinate.longitude).toFixed(7).toString()):
                          qsTr("0")
            color: UIConstants.textColor
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignHCenter
            anchors.fill: parent
        }
    }
}
