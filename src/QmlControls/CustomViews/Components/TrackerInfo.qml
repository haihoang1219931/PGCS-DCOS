/**
 * ==============================================================================
 * @Project: VCM01TargetViewer
 * @Component: Flat Button
 * @Breif:
 * @Author: Trung Nguyen
 * @Date: 18/02/2019
 * @Language: QML
 * @License: (c) Viettel Aerospace Institude - Viettel Group
 * ============================================================================
 */

//----------------------- Include QT libs -------------------------------------
import QtQuick 2.6
import QtQuick.Controls 2.1

//---------------- Include custom libs ----------------------------------------
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0

//----------------------- Component definition- ------------------------------
Rectangle {
    id: root
    width: 170
    height: 230
    //---------- properties
    color: UIConstants.transparentBlue
    border.color: UIConstants.textColor
    border.width: 1
    radius: 5
    property string time: "0000-00-00\n00:00:00"
    property real lat: 0
    property real lon: 0
    property real baro: 0
    property real heading: 0
    property real rssi: 0
    property real status: 0
    property real result1: 0
    property real result2: 0
    Label {
        id: label
        x: 9
        y: 8
        text: qsTr("Time:")
        color: UIConstants.textColor
    }

    Label {
        id: label1
        x: 79
        width: 84
        height: 34
        text: root.time
        horizontalAlignment: Text.AlignLeft
        wrapMode: Text.WordWrap
        anchors.top: parent.top
        anchors.topMargin: 8
        anchors.right: parent.right
        anchors.rightMargin: 7
        color: UIConstants.textColor
    }

    Label {
        id: label2
        x: 9
        y: 48
        text: qsTr("Lat:")
        color: UIConstants.textColor
    }

    Label {
        id: label3
        x: 79
        y: -7
        width: 84
        height: 16
        text: Number(root.lat).toFixed(7).toString()
        anchors.right: parent.right
        anchors.rightMargin: 7
        anchors.top: parent.top
        anchors.topMargin: 48
        color: UIConstants.textColor
    }

    Label {
        id: label4
        x: 79
        y: -13
        width: 84
        height: 16
        text: Number(root.lon).toFixed(7).toString()
        anchors.right: parent.right
        anchors.rightMargin: 7
        anchors.top: parent.top
        anchors.topMargin: 70
        color: UIConstants.textColor
    }

    Label {
        id: label5
        x: 9
        y: 70
        text: qsTr("Lon:")
        color: UIConstants.textColor
    }

    Label {
        id: label6
        x: 79
        y: -12
        width: 84
        height: 16
        text: Number(root.baro).toFixed(0).toString()
        anchors.right: parent.right
        anchors.rightMargin: 7
        anchors.top: parent.top
        anchors.topMargin: 92
        color: UIConstants.textColor
    }

    Label {
        id: label7
        x: 79
        y: -12
        width: 84
        height: 16
        text: Number(root.heading).toFixed(2).toString()
        anchors.right: parent.right
        anchors.rightMargin: 7
        anchors.top: parent.top
        anchors.topMargin: 114
        color: UIConstants.textColor
    }

    Label {
        id: label8
        x: 79
        y: -9
        width: 84
        height: 16
        text: Number(root.rssi).toFixed(2).toString()
        anchors.right: parent.right
        anchors.rightMargin: 7
        anchors.top: parent.top
        anchors.topMargin: 136
        color: UIConstants.textColor
    }

    Label {
        id: label9
        x: 79
        y: -9
        width: 84
        height: 16
        text: Number(root.status).toFixed(0).toString()
        anchors.right: parent.right
        anchors.rightMargin: 7
        anchors.top: parent.top
        anchors.topMargin: 158
        color: UIConstants.textColor
    }

    Label {
        id: label10
        x: 9
        y: 92
        text: qsTr("Baro:")
        color: UIConstants.textColor
    }

    Label {
        id: label11
        x: 9
        y: 114
        text: qsTr("Heading:")
        color: UIConstants.textColor
    }

    Label {
        id: label12
        x: 9
        y: 136
        text: qsTr("RSSI:")
        color: UIConstants.textColor
    }

    Label {
        id: label13
        x: 9
        y: 158
        text: qsTr("Status:")
        color: UIConstants.textColor
    }

    Label {
        id: label14
        x: 79
        y: -14
        width: 84
        height: 16
        text: Number(root.result1).toFixed(0).toString()
        anchors.right: parent.right
        anchors.rightMargin: 7
        anchors.top: parent.top
        anchors.topMargin: 180
        color: UIConstants.textColor
    }

    Label {
        id: label15
        x: 79
        y: -1
        width: 84
        height: 16
        text: Number(root.result2).toFixed(0).toString()
        anchors.right: parent.right
        anchors.rightMargin: 7
        anchors.top: parent.top
        anchors.topMargin: 202
        color: UIConstants.textColor
    }

    Label {
        id: label16
        x: 9
        y: 180
        text: qsTr("Result1:")
        color: UIConstants.textColor
    }

    Label {
        id: label17
        x: 9
        y: 202
        text: qsTr("Result2:")
        color: UIConstants.textColor
    }

}

/*##^## Designer {
    D{i:2;anchors_y:8}D{i:4;anchors_y:8}D{i:5;anchors_y:8}D{i:7;anchors_y:8}D{i:8;anchors_y:8}
D{i:9;anchors_y:8}D{i:10;anchors_y:8}D{i:16;anchors_y:8}D{i:17;anchors_y:8}
}
 ##^##*/
