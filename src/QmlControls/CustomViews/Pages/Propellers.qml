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
        text: "Press Safety and continue to check propellers"
        wrapMode: Text.WordWrap
        anchors.right: imgLogo1.left
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
        id: imgLogo1
        x: 379
        width: 446
        height: 285
        anchors.top: parent.top
        anchors.topMargin: 8
        anchors.right: parent.right
        anchors.rightMargin: 8
        source: "qrc:/assets/images/Shikra_4.png"
    }

    Rectangle {
        id: rectangle
        x: 120
        y: 226
        width: UIConstants.sRect * 15
        height: UIConstants.sRect * 10
        color: "#00000000"
        anchors.horizontalCenter: parent.horizontalCenter

        FlatButtonIcon {
            id: btnP1
            icon: qsTr("Propeller 01")
            iconSize: UIConstants.fontSize
            border.color: UIConstants.greenColor
            width: UIConstants.sRect * 6
            height: UIConstants.sRect * 2
            isAutoReturn: true
            border.width: 2
            anchors.top: parent.top
            anchors.topMargin: 8
            anchors.left: parent.left
            anchors.leftMargin: 8

            onClicked: {
                vehicle.motorTest(1,8)
            }
        }

        FlatButtonIcon {
            id: btnP2
            x: 280
            icon: qsTr("Propeller 02")
            iconSize: UIConstants.fontSize
            border.color: UIConstants.greenColor
            width: UIConstants.sRect * 6
            height: UIConstants.sRect * 2
            isAutoReturn: true
            anchors.top: parent.top
            anchors.topMargin: 8
            anchors.right: parent.right
            anchors.rightMargin: 8
            onClicked: {
                vehicle.motorTest(2,8)
            }
        }

        FlatButtonIcon {
            id: btnP3
            x: 280
            y: 187
            icon: qsTr("Propeller 03")
            iconSize: UIConstants.fontSize
            border.color: UIConstants.greenColor
            width: UIConstants.sRect * 6
            height: UIConstants.sRect * 2
            isAutoReturn: true
            anchors.right: parent.right
            anchors.rightMargin: 8
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 8
            onClicked: {
                vehicle.motorTest(3,8)
            }
        }

        FlatButtonIcon {
            id: btnP4
            y: 168
            icon: qsTr("Propeller 04")
            iconSize: UIConstants.fontSize
            border.color: UIConstants.greenColor
            width: UIConstants.sRect * 6
            height: UIConstants.sRect * 2
            isAutoReturn: true
            anchors.left: parent.left
            anchors.leftMargin: 8
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 8
            onClicked: {
                vehicle.motorTest(4,8)
            }
        }
    }

}
