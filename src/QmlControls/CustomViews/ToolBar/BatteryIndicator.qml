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
        id: batteryInfo
        width: UIConstants.sRect * 11
        height: UIConstants.sRect * 7.5
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
                id:                 batteryCol
                spacing:            UIConstants.defaultFontPixelHeight * 0.5
                width:              parent.width
                height:             parent.height
                anchors.centerIn: parent
                Label {
                    id:             batteryLabel
                    text:           vehicle ? qsTr("Battery Status") : qsTr("Battery Data Unavailable")
                    color: UIConstants.textColor;
                    font.family: UIConstants.appFont;
                    font.pixelSize: UIConstants.fontSize
                    anchors.horizontalCenter: parent.horizontalCenter
                    anchors.margins:    UIConstants.defaultFontPixelHeight
                }

                ListView{
                    id:listBatteryData
                    anchors.left: parent.left
                    model: vehicle.propertiesModel
                    height: UIConstants.sRect * 5 + 10
                    width: parent.width
                    anchors.margins:    UIConstants.defaultFontPixelHeight
                    delegate: Item{
                        height: visible?UIConstants.sRect:0
                        visible: name.includes("V_BattA") || name.includes("V_BattB") ||
                                 name.includes("I_BattA") || name.includes("I_BattB") ||
                                 name.includes("PMU_Temp")|| name.includes("GenStatus")
                        Label {
                            id: lblName
                            height: UIConstants.sRect
                            text: name + ": " + parseFloat(value).toFixed(2) + unit
                            anchors.verticalCenter: parent.verticalCenter
                            verticalAlignment: Text.AlignVCenter
                            horizontalAlignment: Text.AlignLeft
                            color: {var _color = !isNaN(parseFloat(value)) ? ((parseFloat(value) < lowerValue) ?
                                                                    lowerColor :((parseFloat(value) > upperValue) ? upperColor : middleColor)): "transparent"
                                if(_color === "transparent")
                                    _color = UIConstants.textColor
                                if(!(name.includes("V_BattA") || name.includes("V_BattB") ||
                                   name.includes("I_BattA") || name.includes("I_BattB") ||
                                   name.includes("PMU_Temp")|| name.includes("GenStatus")))
                                    _color = UIConstants.textColor
                                return _color;
                            }
                            font.pixelSize: UIConstants.fontSize
                            font.family: UIConstants.appFont
                        }
                    }

                    Timer{
                        id: updateBattIcon
                        interval: 500
                        repeat: true
                        running: true
                        onTriggered: {
                            var error = false
                            for(var i=0;i<listBatteryData.contentItem.children.length ; i++)
                            {
                                var qitem = listBatteryData.contentItem.children[i]
                                var lbText = qitem.children[0]
                                if(lbText)
                                    if(lbText.color === UIConstants.redColor)
                                        error = true
                            }
                            if(error)
                                batteryIcon.color =  vehicle.link ? UIConstants.redColor : UIConstants.textColor
                            else
                                batteryIcon.color =  vehicle.link ? UIConstants.navIconColor  : UIConstants.textColor
                        }
                    }
                }

//                GridLayout {
//                    id:                 batteryGrid
////                    visible:            (vehicle && vehicle.countGPS >= 0)
//                    columnSpacing:      UIConstants.defaultFontPixelWidth
//                    anchors.left: parent.left
//                    columns: 1

//                    Label { text: qsTr("V_batt12S: ") + (vehicle ? vehicle.vBatt12S.toFixed(2).toString() + "V" : "N/A")
//                        color: UIConstants.textColor; font.family: UIConstants.appFont; font.pixelSize: UIConstants.fontSize }
//                    Label { text: qsTr("V_battA: ") + (vehicle ? vehicle.vBattA.toFixed(2).toString() + "V" : "N/A")
//                        color: UIConstants.textColor; font.family: UIConstants.appFont; font.pixelSize: UIConstants.fontSize }
//                    Label { text: qsTr("I_battA: ") + (vehicle ? vehicle.iBattA.toFixed(2).toString() + "A" : "N/A")
//                        color: UIConstants.textColor; font.family: UIConstants.appFont; font.pixelSize: UIConstants.fontSize }
//                    Label { text: qsTr("V_battB: ")+  (vehicle ? vehicle.vBattB.toFixed(2).toString() + "V" : "N/A")
//                        color: UIConstants.textColor; font.family: UIConstants.appFont; font.pixelSize: UIConstants.fontSize }
//                    Label { text: qsTr("I_battB: ") + (vehicle ? vehicle.iBattA.toFixed(2).toString() + "A" : "N/A")
//                        color: UIConstants.textColor; font.family: UIConstants.appFont; font.pixelSize: UIConstants.fontSize }
//                    Label { text: qsTr("GenStatus: ") + (vehicle ? vehicle.genStatus : "N/A")
//                        color: UIConstants.textColor; font.family: UIConstants.appFont; font.pixelSize: UIConstants.fontSize }
//                }
            }
        }
    }

    IconSVG {
        id:                 batteryIcon
        anchors.top:        parent.top
        source:             "qrc:/qmlimages/ToolBar/Images/battery.svg"
        color:              UIConstants.textColor
        anchors.horizontalCenter: parent.horizontalCenter
        width:              iconSize
        height:             iconSize
        opacity: 0.6
    }


    MouseArea {
        anchors.fill:   parent
        onClicked: {
            rootItem.clicked();
        }
    }
}
