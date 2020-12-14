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
    property bool showIndicator: false
    property int iconSize: 30
    signal clicked();
    Item {
        id: gpsInfo
        width: UIConstants.sRect * 12
        height: UIConstants.sRect * 6
        anchors.top: parent.top
        anchors.topMargin: UIConstants.sRect * 3
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
                    text:           (vehicle && FlightVehicle.countGPS >= 0) ? qsTr("GPS Status") : qsTr("GPS Data Unavailable")
                    color: UIConstants.textColor; font.family: UIConstants.appFont; font.pixelSize: UIConstants.fontSize
                    anchors.horizontalCenter: parent.horizontalCenter
                }

                GridLayout {
                    id:                 gpsGrid
//                    visible:            (vehicle && FlightVehicle.countGPS >= 0)
                    columnSpacing:      UIConstants.defaultFontPixelWidth
                    anchors.left: parent.left
                    columns: 1

                    Label { text: qsTr("GPS Count:") + (FlightVehicle.countGPS > 0 ? FlightVehicle.countGPS : qsTr("N/A", "No data to display"))
                        color: UIConstants.textColor; font.family: UIConstants.appFont; font.pixelSize: UIConstants.fontSize }
                    Label { text: qsTr("GPS Lock:") + (FlightVehicle.countGPS > 0 ? FlightVehicle.lockGPS : qsTr("N/A", "No data to display"))
                        color: UIConstants.textColor; font.family: UIConstants.appFont; font.pixelSize: UIConstants.fontSize }
                    Label { text: qsTr("HDOP:") + (FlightVehicle.countGPS > 0 ? Number(FlightVehicle.hdopGPS).toFixed(2).toString() : qsTr("--.--", "No data to display"))
                        color: UIConstants.textColor; font.family: UIConstants.appFont; font.pixelSize: UIConstants.fontSize }
                    Label { text: qsTr("VDOP:")+ (FlightVehicle.countGPS > 0 ? Number(FlightVehicle.vdopGPS).toFixed(2).toString() : qsTr("--.--", "No data to display"))
                        color: UIConstants.textColor; font.family: UIConstants.appFont; font.pixelSize: UIConstants.fontSize }
                    Label { text: qsTr("Course Over Ground:") + (FlightVehicle.countGPS > 0 ? Number(FlightVehicle.courseOverGroundGPS).toFixed(2).toString() : qsTr("--.--", "No data to display"))
                        color: UIConstants.textColor; font.family: UIConstants.appFont; font.pixelSize: UIConstants.fontSize }
                }
            }
        }
    }

    IconSVG {
        id:                 gpsIcon
        anchors.top:        parent.top
        source:             "qrc:/qmlimages/ToolBar/Images/Gps.svg"
        color:              vehicle && FlightVehicle.countGPS > 0 ? UIConstants.navIconColor : UIConstants.textColor
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
            visible:                    vehicle && !isNaN(FlightVehicle.hdopGPS)
            color:                      vehicle && FlightVehicle.countGPS > 0 ? UIConstants.navIconColor : UIConstants.textColor
            text:                       vehicle ? Number(FlightVehicle.countGPS).toString() : ""
            font.family: UIConstants.appFont
            font.pixelSize: UIConstants.fontSize
        }

        Label {
            id:         hdopValue
            visible:    vehicle && !isNaN(FlightVehicle.hdopGPS)
            color:      vehicle && FlightVehicle.countGPS > 0 ? UIConstants.navIconColor : UIConstants.textColor
            text:       vehicle ? FlightVehicle.hdopGPS.toFixed(1) : ""
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
