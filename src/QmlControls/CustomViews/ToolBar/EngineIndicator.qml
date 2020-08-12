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
//-- Engine Indicator
Item {
    id:             rootItem
    width:  500
    height: 350
    property var validatorFloat: /^([0-9]|[1-9][0-9])(\.)([0-9])/
    property real flowRate: 0.7
    property real availbleFuel: 0
    property bool showIndicator: false
    property int iconSize: 30
    signal clicked();
    Item {
        id: engineInfo
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
                id:                 engineCol
                spacing:            UIConstants.defaultFontPixelHeight * 0.5
                width:              Math.max(engineGrid.width, engineLabel.width)
                anchors.margins:    UIConstants.defaultFontPixelHeight
                anchors.centerIn:   parent

                Label {
                    id:             engineLabel
                    text:           (vehicle) ? qsTr("Engine Status") : qsTr("Engine Data Unavailable")
                    color: UIConstants.textColor; font.family: UIConstants.appFont; font.pixelSize: UIConstants.fontSize
                    anchors.horizontalCenter: parent.horizontalCenter
                }

                GridLayout {
                    id:                 engineGrid
//                    visible:            (vehicle && vehicle.countGPS >= 0)
                    columnSpacing:      UIConstants.defaultFontPixelWidth
                    anchors.left: parent.left
                    columns: 1
                    Label { text: qsTr("Original Fuel(l):       ");
                        color: UIConstants.textColor; font.family: UIConstants.appFont; font.pixelSize: UIConstants.fontSize
                        Rectangle{
                            id:rectTxt
                            border.color: "white"
                            color: "transparent"
                            border.width: 1
                            radius: 3
                            width: 60;
                            height: 20;
                            x:160
                            TextInput{
                                clip: true
                                id: txtFuelOrigin
                                anchors.left: parent.left
                                anchors.right: parent.right
                                anchors.bottom: parent.bottom
                                anchors.top: parent.top
                                anchors.leftMargin: 3
                                anchors.rightMargin: 3
                                //x:160

                                validator: RegExpValidator { regExp: rootItem.validatorFloat }
                                color: UIConstants.textColor; font.family: UIConstants.appFont; font.pixelSize: UIConstants.fontSize
                            }
                        }


                    }



                    Label { text: qsTr("Used Fuel(l): ") + vehicle.fuelUsed.toFixed(2);
                        color: UIConstants.textColor; font.family: UIConstants.appFont; font.pixelSize: UIConstants.fontSize
                    }

                    Label { text: qsTr("Available Fuel(l): ") + (engineGrid.getOriginalFuel() - vehicle.fuelUsed).toFixed(2).toString() ;
                        id:lbAvailbleFuel
                        color: UIConstants.textColor; font.family: UIConstants.appFont; font.pixelSize: UIConstants.fontSize
                        onTextChanged:{
                            var timeRemain = engineGrid.getRemainingTime();
                            var engineColor = engineGrid.getRemainingTime() > 0.5 ? UIConstants.navIconColor : UIConstants.redColor;
                            lbRemainingTime.text = qsTr("Remaining Time(h): ") + timeRemain.toFixed(2).toString();
                            engineIcon.color = engineColor;
                            lbTime.color = engineColor;
                            lbTime.text = timeRemain.toFixed(1).toString()+"h";
                            //console.log("original: "+engineGrid.getOriginalFuel())
                        }
                    }

                    Label { text: qsTr("Remaining Time(s): NaN") ;
                        id:lbRemainingTime
                        color: UIConstants.textColor; font.family: UIConstants.appFont; font.pixelSize: UIConstants.fontSize }

                    function getOriginalFuel(){
                        var valueText = txtFuelOrigin.text;
                        if(valueText!== "" && !isNaN(valueText)){
                            var originalFuel =  parseFloat(valueText);
                            return originalFuel;
                        }else return 0;
                    }

                    function getAvailbleFuel(){
                        if(vehicle.fuelUsed != null && vehicle.fuelUsed != undefined){
                            var fuel = getOriginalFuel() - vehicle.fuelUsed;
                            return fuel;
                        }else return 0;
                    }

                    function getRemainingTime(){
                        var time = getAvailbleFuel() / flowRate;
                        if(time < 0)
                            return 0;
                        return time;
                    }

                 }
            }
        }
    }

    IconSVG {
        id:                 engineIcon
        anchors.top:        parent.top
        source:             "qrc:/qmlimages/ToolBar/Images/engine.svg"
        color:              UIConstants.textColor
        anchors.horizontalCenter: parent.horizontalCenter
        width:              iconSize
        height:             iconSize
        opacity: 0.6
    }

//    Column {
//        id: engineValuesColumn
//        spacing: 2
//        anchors.left: engineIcon.right
//        anchors.top: parent.top
//        anchors.topMargin: 2
//        anchors.leftMargin: 2
//        anchors.bottom: parent.bottom
//        anchors.bottomMargin: 2
//        opacity: 0.6
        Label {
            id: lbTime
            anchors.top:   engineIcon.top
            anchors.left: engineIcon.right
            visible:                    vehicle && !isNaN(vehicle.fuelUsed)
            color:                      UIConstants.textColor
            //text:                       vehicle ? Number(vehicle.countGPS).toString() : ""
            font.family: UIConstants.appFont
            font.pixelSize: UIConstants.fontSize
            opacity: 0.6
        }

//        Label {
//            id:         hdopValue
//            visible:    vehicle && !isNaN(vehicle.hdopGPS)
//            color:      vehicle && vehicle.countGPS > 0 ? UIConstants.navIconColor : UIConstants.textColor
//            text:       vehicle ? vehicle.hdopGPS.toFixed(1) : ""
//            font.family: UIConstants.appFont
//            font.pixelSize: UIConstants.fontSize
//        }
   // }

    MouseArea {
        anchors.fill:   parent
        onClicked: {
            rootItem.clicked();
        }
    }
}
