/****************************************************************************
 *
 *   (c) 2009-2016 QGROUNDCONTROL PROJECT <http://www.qgroundcontrol.org>
 *
 * QGroundControl is licensed according to the terms in the file
 * COPYING.md in the root of the source code directory.
 *
 ****************************************************************************/


import QtQuick          2.3
import QtQuick.Controls 1.2
import QtQuick.Layouts  1.2
import QtQuick.Window 2.2
import QtQuick 2.6
import QtQuick.Controls 2.1
import QtQuick.Layouts 1.3
import QtGraphicalEffects 1.0
import QtQuick 2.0

import CustomViews.Components   1.0
import CustomViews.UIConstants  1.0

import io.qdt.dev               1.0
//-------------------------------------------------------------------------
//-- GPS Indicator
Item {
    id:             rootItem
    width:  640
    height: 480
//    anchors.top:    parent.top
//    anchors.bottom: parent.bottom
    property bool showIndicator: false
    property int iconSize: 30
    signal clicked();
    Item {
        id: gpsInfo
        width: 280
        height: 150
        anchors.top: parent.top
        anchors.topMargin: 55 + UIConstants.defaultFontPixelHeight
        visible: showIndicator
        anchors.horizontalCenter: parent.horizontalCenter
        Rectangle {
            anchors.fill: parent
            radius: UIConstants.rectRadius
            color:  UIConstants.transparentBlue
            border.color:   UIConstants.grayColor

            Column {
                id:                 gpsCol
                spacing:            UIConstants.defaultFontPixelHeight * 0.5
                width:              Math.max(gpsGrid.width, gpsLabel.width)
                anchors.margins:    UIConstants.defaultFontPixelHeight
                anchors.centerIn:   parent

                Label {
                    id:             gpsLabel
                    text:           (vehicle && vehicle.countGPS >= 0) ? qsTr("GPS Status") : qsTr("GPS Data Unavailable")
                    color: UIConstants.textColor; font.family: UIConstants.appFont; font.pixelSize: UIConstants.fontSize
                    anchors.horizontalCenter: parent.horizontalCenter
                }

                GridLayout {
                    id:                 gpsGrid
//                    visible:            (vehicle && vehicle.countGPS >= 0)
                    columnSpacing:      UIConstants.defaultFontPixelWidth
                    anchors.left: parent.left
                    columns: 1

                    Label { text: qsTr("GPS Count:") + (vehicle.countGPS > 0 ? vehicle.countGPS : qsTr("N/A", "No data to display"))
                        color: UIConstants.textColor; font.family: UIConstants.appFont; font.pixelSize: UIConstants.fontSize }
                    Label { text: qsTr("GPS Lock:") + (vehicle.countGPS > 0 ? vehicle.lockGPS : qsTr("N/A", "No data to display"))
                        color: UIConstants.textColor; font.family: UIConstants.appFont; font.pixelSize: UIConstants.fontSize }
                    Label { text: qsTr("HDOP:") + (vehicle.countGPS > 0 ? Number(vehicle.hdopGPS).toFixed(2).toString() : qsTr("--.--", "No data to display"))
                        color: UIConstants.textColor; font.family: UIConstants.appFont; font.pixelSize: UIConstants.fontSize }
                    Label { text: qsTr("VDOP:")+ (vehicle.countGPS > 0 ? Number(vehicle.vdopGPS).toFixed(2).toString() : qsTr("--.--", "No data to display"))
                        color: UIConstants.textColor; font.family: UIConstants.appFont; font.pixelSize: UIConstants.fontSize }
                    Label { text: qsTr("Course Over Ground:") + (vehicle.countGPS > 0 ? Number(vehicle.courseOverGroundGPS).toFixed(2).toString() : qsTr("--.--", "No data to display"))
                        color: UIConstants.textColor; font.family: UIConstants.appFont; font.pixelSize: UIConstants.fontSize }
                }
            }
        }
    }

    IconSVG {
        id:                 gpsIcon
        anchors.top:        parent.top
        source:             "qrc:/qmlimages/ToolBar/Images/Gps.svg"
        color:              vehicle && vehicle.countGPS > 0 ? UIConstants.greenColor : UIConstants.textColor
        anchors.horizontalCenter: parent.horizontalCenter
        width:              iconSize
        height:             iconSize
        opacity: 0.6
    }

    Column {
        id: gpsValuesColumn
        spacing: 2
        anchors.left: gpsIcon.right
        anchors.top: parent.top
        anchors.topMargin: 2
        anchors.leftMargin: 2
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 2
        opacity: 0.6
        Label {
            anchors.horizontalCenter:   hdopValue.horizontalCenter
            visible:                    vehicle && !isNaN(vehicle.hdopGPS)
            color:                      vehicle && vehicle.countGPS > 0 ? UIConstants.greenColor : UIConstants.textColor
            text:                       vehicle ? Number(vehicle.countGPS).toString() : ""
            font.family: UIConstants.appFont
            font.pixelSize: UIConstants.fontSize
        }

        Label {
            id:         hdopValue
            visible:    vehicle && !isNaN(vehicle.hdopGPS)
            color:      vehicle && vehicle.countGPS > 0 ? UIConstants.greenColor : UIConstants.textColor
            text:       vehicle ? vehicle.hdopGPS.toFixed(1) : ""
            font.family: UIConstants.appFont
            font.pixelSize: UIConstants.fontSize
        }
    }

    MouseArea {
        anchors.fill:   parent
        onClicked: {
            rootItem.clicked();
        }
    }
}
