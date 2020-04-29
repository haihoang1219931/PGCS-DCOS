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
    QLabel {
        id: lblTitle
        height: 54
        text: "Press Safety and continue to check propellers"
        anchors.right: parent.right
        anchors.rightMargin: 8
        anchors.left: parent.left
        anchors.top: parent.top
        anchors.leftMargin: 8
        anchors.topMargin: 0
        border.width: 0
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
            icon: qsTr("Propeller A")
            iconSize: UIConstants.fontSize
            border.color: UIConstants.greenColor
            width: UIConstants.sRect * 6
            height: UIConstants.sRect * 2
            anchors.right: parent.right
            anchors.rightMargin: 8
            isAutoReturn: true
            border.width: 2
            anchors.top: parent.top
            anchors.topMargin: 8

            onClicked: {
                vehicle.motorTest(1,8)
            }
        }

        FlatButtonIcon {
            id: btnP2
            x: 280
            icon: qsTr("Propeller B")
            iconSize: UIConstants.fontSize
            border.color: UIConstants.greenColor
            width: UIConstants.sRect * 6
            height: UIConstants.sRect * 2
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 8
            isAutoReturn: true
            border.width: 2
            anchors.right: parent.right
            anchors.rightMargin: 8
            onClicked: {
                vehicle.motorTest(2,8)
            }
        }

        FlatButtonIcon {
            id: btnP3
            y: 187
            icon: qsTr("Propeller C")
            iconSize: UIConstants.fontSize
            border.color: UIConstants.greenColor
            width: UIConstants.sRect * 6
            height: UIConstants.sRect * 2
            anchors.left: parent.left
            anchors.leftMargin: 8
            isAutoReturn: true
            border.width: 2
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 8
            onClicked: {
                vehicle.motorTest(3,8)
            }
        }

        FlatButtonIcon {
            id: btnP4
            icon: qsTr("Propeller D")
            iconSize: UIConstants.fontSize
            border.color: UIConstants.greenColor
            width: UIConstants.sRect * 6
            height: UIConstants.sRect * 2
            anchors.top: parent.top
            anchors.topMargin: 8
            isAutoReturn: true
            border.width: 2
            anchors.left: parent.left
            anchors.leftMargin: 8
            onClicked: {
                vehicle.motorTest(4,8)
            }
        }
    }

}

/*##^## Designer {
    D{i:6;anchors_x:280}D{i:7;anchors_y:168}
}
 ##^##*/
