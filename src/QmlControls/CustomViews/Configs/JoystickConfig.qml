/****************************************************************************
 *
 *   (c) 2009-2016 QGROUNDCONTROL PROJECT <http://www.qgroundcontrol.org>
 *
 * QGroundControl is licensed according to the terms in the file
 * COPYING.md in the root of the source code directory.
 *
 ****************************************************************************/


import QtQuick          2.3
import QtQuick.Controls 1.2 as OldCtrl
import QtQuick.Controls 2.4
import QtQuick.Controls.Styles 1.4
import QtQuick.Dialogs  1.2

import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0
/// Joystick Config
Item {
    id: rootItem
    width: 1280
    height: 768
    clip: true
    property var itemListName:
        UIConstants.itemTextMultilanguages["CONFIGURATION"]["JOYSTICK"]
    property var mapAxisKeys: ["Unused","Roll","Pitch","Yaw","Throttle"]
    property var mapAxis: {"Unused":-1,"Roll":0,"Pitch":1,"Yaw":2,"Throttle":3}
    property var mapAxisCamKeys: ["Unused","Pan","Tilt","Zoom"]
    property var mapAxisCam: {"Unused":-1,"Pan":0,"Tilt":1,"Zoom":2}
    property var mapButtonKeys: ["Unused","PIC/CIC","CIC/PIC","Guided","Loiter","Auto","RTL",
        "EO/IR",
        "SNAPSHOT","VISUAL","FREE","PRESET","DIGITAL_STAB","RECORD"]
    property var mapButton: {"Unused":-1,"PIC/CIC":0,"CIC/PIC":1,"Guided":2,"Loiter":3,"Auto":4,"RTL":5,
        "EO/IR":6,
        "SNAPSHOT":7,"VISUAL":8,"FREE":9,"PRESET":10,"DIGITAL_STAB":11,"RECORD":12}
    Row{
        id: row
        anchors.rightMargin: UIConstants.sRect
        anchors.bottomMargin: UIConstants.sRect
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.top: parent.top
        anchors.topMargin: UIConstants.sRect
        anchors.left: parent.left
        anchors.leftMargin: UIConstants.sRect
        spacing: UIConstants.sRect
        Column{
            id: clmJoystick
            width: UIConstants.sRect* 15
            spacing:    UIConstants.sRect/2
            Row{
                Label {
                    text: itemListName["SELECTION"]
                          [UIConstants.language[UIConstants.languageID]]
                    color: UIConstants.textColor
                    font.pixelSize: UIConstants.fontSize
                    font.family: UIConstants.appFont
                }
            }
            Row{
                spacing: UIConstants.sRect/2
                QComboBox{
                    id: cbxListJoystick
                    width: clmJoystick.width - btnSelectJoystick.width - parent.spacing
                    height: UIConstants.sRect * 1.5
                }
                OldCtrl.Button{
                    id: btnSelectJoystick
                    width: UIConstants.sRect * 3
                    height: UIConstants.sRect * 1.5
                    text: itemListName["BTN_SELECT"]
                          [UIConstants.language[UIConstants.languageID]]
                    style: ButtonStyle{
                        background: Rectangle{
                            color: UIConstants.info
                        }
                        label: Label{
                            color: UIConstants.textColor
                            font.pixelSize: UIConstants.fontSize
                            font.family: UIConstants.appFont
                            verticalAlignment: Label.AlignVCenter
                            horizontalAlignment: Label.AlignHCenter
                            text: btnSelectJoystick.text
                        }
                    }

                    onClicked: {
                        joystick.setJoyID(cbxListJoystick.currentText);
                    }
                }
            }
            Row{
                Label{
                    text: itemListName["NAME"]
                          [UIConstants.language[UIConstants.languageID]]+":"
                    width: UIConstants.sRect*3
                    height: UIConstants.sRect * 1.5
                    color: UIConstants.textColor
                    font.pixelSize: UIConstants.fontSize
                    font.family: UIConstants.appFont
                    horizontalAlignment: Label.AlignLeft
                    verticalAlignment: Label.AlignVCenter
                }
                Label{
                    id: lblJSName
                    width: clmJoystick.width - clmJoystick.spacing - UIConstants.sRect*3
                    color: UIConstants.textColor
                    font.pixelSize: UIConstants.fontSize
                    font.family: UIConstants.appFont
                    horizontalAlignment: Label.AlignLeft
                    verticalAlignment: Label.AlignVCenter
                    wrapMode: Label.WordWrap
                }
            }
            Row{
                Label{
                    text: itemListName["VERSION"]
                          [UIConstants.language[UIConstants.languageID]]+":"
                    width: UIConstants.sRect*3
                    height: UIConstants.sRect * 1.5
                    color: UIConstants.textColor
                    font.pixelSize: UIConstants.fontSize
                    font.family: UIConstants.appFont
                    horizontalAlignment: Label.AlignLeft
                    verticalAlignment: Label.AlignVCenter
                }
                Label{
                    id: lblJSVersion
                    height: UIConstants.sRect * 1.5
                    color: UIConstants.textColor
                    font.pixelSize: UIConstants.fontSize
                    font.family: UIConstants.appFont
                    horizontalAlignment: Label.AlignLeft
                    verticalAlignment: Label.AlignVCenter
                }
            }
            Row{
                Label{
                    text: itemListName["AXES"]
                          [UIConstants.language[UIConstants.languageID]]+":"
                    width: UIConstants.sRect*3
                    height: UIConstants.sRect * 1.5
                    color: UIConstants.textColor
                    font.pixelSize: UIConstants.fontSize
                    font.family: UIConstants.appFont
                    horizontalAlignment: Label.AlignLeft
                    verticalAlignment: Label.AlignVCenter
                }
                Label{
                    id: lblJSAxes
                    height: UIConstants.sRect * 1.5
                    color: UIConstants.textColor
                    font.pixelSize: UIConstants.fontSize
                    font.family: UIConstants.appFont
                    horizontalAlignment: Label.AlignLeft
                    verticalAlignment: Label.AlignVCenter
                }
            }
            Row{
                Label{
                    text: itemListName["BUTTONS"]
                          [UIConstants.language[UIConstants.languageID]]+":"
                    width: UIConstants.sRect*3
                    height: UIConstants.sRect * 1.5
                    color: UIConstants.textColor
                    font.pixelSize: UIConstants.fontSize
                    font.family: UIConstants.appFont
                    horizontalAlignment: Label.AlignLeft
                    verticalAlignment: Label.AlignVCenter
                }
                Label{
                    id: lblJSButtons
                    height: UIConstants.sRect * 1.5
                    color: UIConstants.textColor
                    font.pixelSize: UIConstants.fontSize
                    font.family: UIConstants.appFont
                    horizontalAlignment: Label.AlignLeft
                    verticalAlignment: Label.AlignVCenter
                }
            }
            Row {
                id: chbUseJoystick
                width: parent.width
                height: UIConstants.sRect * 1.5
                spacing: 5
                Rectangle{
                    width: parent.height
                    height: parent.height
                    radius: 3
                    Rectangle{
                        visible: joystick.useJoystick
                        color: "#555"
                        border.color: "#333"
                        radius: 1
                        anchors.margins: parent.height / 4
                        anchors.fill: parent
                    }
                    MouseArea{
                        anchors.fill: parent
                        onClicked: {
                            joystick.useJoystick = !joystick.useJoystick;
                        }
                    }
                }
                Label{
                    height: parent.height
                    verticalAlignment: Label.AlignVCenter
                    horizontalAlignment: Label.AlignLeft
                    font.pixelSize: UIConstants.fontSize
                    font.family: UIConstants.appFont
                    color: UIConstants.textColor
                    text: itemListName["ENABLE"]
                          [UIConstants.language[UIConstants.languageID]]
                }
            }

            Row{
                spacing: parent.width - btnSaveConfig.width - btnResetConfig.width
                OldCtrl.Button{
                    id: btnSaveConfig
                    width: UIConstants.sRect*6
                    height: UIConstants.sRect*1.5
                    scale: pressed? 0.9:1
                    style: ButtonStyle{
                        background: Rectangle{
                            color: btnSaveConfig.pressed?UIConstants.textColor:UIConstants.greenColor
                        }
                        label: Label{
                            color: !btnSaveConfig.pressed?UIConstants.textColor:UIConstants.greenColor
                            font.pixelSize: UIConstants.fontSize
                            font.family: UIConstants.appFont
                            text: btnSaveConfig.text
                            verticalAlignment: Label.AlignVCenter
                            horizontalAlignment: Label.AlignHCenter
                        }
                    }
                    text: itemListName["BTN_SAVE"]
                          [UIConstants.language[UIConstants.languageID]]
                    onClicked: {
                        joystick.saveConfig();
                    }
                }
                OldCtrl.Button{
                    id: btnResetConfig
                    width: UIConstants.sRect*6
                    height: UIConstants.sRect*1.5
                    scale: pressed? 0.9:1
                    style: ButtonStyle{
                        background: Rectangle{
                            color: btnResetConfig.pressed?UIConstants.textColor:UIConstants.redColor
                        }
                        label: Label{
                            color: !btnResetConfig.pressed?UIConstants.textColor:UIConstants.redColor
                            font.pixelSize: UIConstants.fontSize
                            font.family: UIConstants.appFont
                            text: btnResetConfig.text
                            verticalAlignment: Label.AlignVCenter
                            horizontalAlignment: Label.AlignHCenter
                        }
                    }
                    text: itemListName["BTN_RESET"]
                          [UIConstants.language[UIConstants.languageID]]
                    onClicked: {
                        joystick.resetConfig();
                    }
                }
            }
        }

        // Button monitor
        Column {
            id: clmButton
            width: UIConstants.sRect * 9
            height: parent.height - parent.spacing
            spacing:    UIConstants.sRect/2
            Label {
                id: lblButtonMonitor
                text: itemListName["BUTTONS_MONITOR"]
                      [UIConstants.language[UIConstants.languageID]]
                color: UIConstants.textColor
                font.pixelSize: UIConstants.fontSize
                font.family: UIConstants.appFont
            }

            ListView {
                id:     buttonMonitorRepeater
                width: parent.width
                height: parent.height - lblButtonMonitor.height - parent.spacing
                model: joystick.buttons
                spacing:    UIConstants.sRect/2
                clip: true
                delegate: Row{
                    id: rowButton
                    spacing: UIConstants.sRect/2
                    height: UIConstants.sRect*1.5
                    Rectangle {
                        width:          parent.height
                        height:         width
                        border.width:   1
                        border.color:   UIConstants.grayColor
                        color:          pressed ? UIConstants.greenColor : UIConstants.transparentColor
                        Label {
                            anchors.fill:           parent
                            horizontalAlignment:    Text.AlignHCenter
                            verticalAlignment:      Text.AlignVCenter
                            text:                   Number(id).toString()
                            color: UIConstants.textColor
                            font.pixelSize: UIConstants.fontSize
                            font.family: UIConstants.appFont
                        }
                    }
                    QComboBox{
                        id: cbxButton
                        model: mapButtonKeys
                        currentIndex: mapButton[mapFuncConfig]+1
                        onCurrentTextChanged: {
                            if(currentText !== mapFuncConfig)
                                joystick.mapButton(id,cbxButton.currentText,false)
                        }
                    }
                }
            } // Repeater
        } // Column - Button Monitor

        Column {
            id: column
            width: parent.width - clmJoystick.width -  clmButton.width - parent.spacing*2
            spacing:    UIConstants.sRect/2
            height: parent.height
            Column {
                id: clmAxis
                spacing:    UIConstants.sRect/2
                width: parent.width
                Label {
                    text: itemListName["AXES_FLIGHT_MONITOR"]
                          [UIConstants.language[UIConstants.languageID]]
                    color: UIConstants.textColor
                    font.pixelSize: UIConstants.fontSize
                    font.family: UIConstants.appFont
                }

                Repeater {
                    id:     axisMonitorRepeater
                    width:  parent.width
                    model: joystick.axes

                    delegate:Row {
                        id: rowAxis
                        spacing: UIConstants.sRect/2
                        height: UIConstants.sRect*1.5
                        width: parent.width
                        Label {
                            id:     axisLabel
                            width:          parent.height
                            height:         width
                            text:   Number(id).toString()
                            horizontalAlignment: Label.AlignHCenter
                            verticalAlignment: Label.AlignVCenter
                            anchors.verticalCenter: parent.verticalCenter
                            color: UIConstants.textColor
                            font.pixelSize: UIConstants.fontSize
                            font.family: UIConstants.appFont
                        }

                        Item{
                            id: axisItem
                            width: clmAxis.width - axisLabel.width - cbxAxis.width - chbInvert.width -
    //                               btnSaveAxis.width -
                                   rowAxis.spacing * (rowAxis.children.length - 1)
                            height: parent.height
                            anchors.verticalCenter: parent.verticalCenter
                            Rectangle {
                                id:                     bar
                                anchors.verticalCenter: parent.verticalCenter
                                width:                  parent.width
                                height:                 UIConstants.sRect / 4
                                color:                  UIConstants.blackColor
                            }
                            Rectangle {
                                id: center
                                anchors.horizontalCenter:   parent.horizontalCenter
                                anchors.verticalCenter: parent.verticalCenter
                                width:                      2
                                height:                     bar.height
                                color:                      UIConstants.textColor
                            }
                            Rectangle{
                                id: indicator
                                anchors.verticalCenter: parent.verticalCenter
                                width:                  parent.height/2
                                height:                 width
                                radius:                 width / 2
                                color:                  UIConstants.textColor
                                x:                      (value*(inverted?-1:1)+32768)/65535 * parent.width - width/2
                            }
                        }
                        QComboBox{
                            id: cbxAxis
                            width: parent.height*4
                            height: parent.height
                            model: mapAxisKeys
                            currentIndex: mapAxis[mapFuncConfig]+1
                            onCurrentTextChanged: {
                                if(currentText !== mapFuncConfig){
                                    joystick.mapAxis(id,cbxAxis.currentText,!invertedConfig,false);
                                }
                            }
                        }
                        Row {
                            id: chbInvert
                            width: parent.height*2.5
                            height: parent.height
                            spacing: 5
                            Rectangle{
                                width: parent.height
                                height: parent.height
                                radius: 3
                                Rectangle{
                                    id: rectCheck
                                    visible: invertedConfig
                                    color: "#555"
                                    border.color: "#333"
                                    radius: 1
                                    anchors.margins: parent.height / 4
                                    anchors.fill: parent
                                }
                                MouseArea{
                                    anchors.fill: parent
                                    onClicked: {
                                        joystick.mapAxis(id,cbxAxis.currentText,!invertedConfig,false);
                                    }
                                }
                            }
                            Label{
                                height: parent.height
                                verticalAlignment: Label.AlignVCenter
                                horizontalAlignment: Label.AlignLeft
                                font.pixelSize: UIConstants.fontSize
                                font.family: UIConstants.appFont
                                color: UIConstants.textColor
                                text: itemListName["INVERT"]
                                      [UIConstants.language[UIConstants.languageID]]
                            }
                        }
                    }
                }
            } // Column - Axis Monitor
            Column {
                id: clmAxisCam
                spacing:    UIConstants.sRect/2
                width: parent.width
                Label {
                    text: itemListName["AXES_PAYLOAD_MONITOR"]
                          [UIConstants.language[UIConstants.languageID]]
                    color: UIConstants.textColor
                    font.pixelSize: UIConstants.fontSize
                    font.family: UIConstants.appFont
                }

                Repeater {
                    id:     axisCamMonitorRepeater
                    width:  parent.width
                    model: joystick.axesCam

                    delegate:Row {
                        id: rowAxisCam
                        spacing: UIConstants.sRect/2
                        height: UIConstants.sRect*1.5
                        width: parent.width
                        Label {
                            id:     axisCamLabel
                            width:          parent.height
                            height:         width
                            text:   Number(id).toString()
                            horizontalAlignment: Label.AlignHCenter
                            verticalAlignment: Label.AlignVCenter
                            anchors.verticalCenter: parent.verticalCenter
                            color: UIConstants.textColor
                            font.pixelSize: UIConstants.fontSize
                            font.family: UIConstants.appFont
                        }

                        Item{
                            id: axisCamItem
                            width: clmAxisCam.width - axisCamLabel.width - cbxAxisCam.width - chbInvertCam.width -
    //                               btnSaveAxis.width -
                                   rowAxisCam.spacing * (rowAxisCam.children.length - 1)
                            height: parent.height
                            anchors.verticalCenter: parent.verticalCenter
                            Rectangle {
                                id:                     barCam
                                anchors.verticalCenter: parent.verticalCenter
                                width:                  parent.width
                                height:                 UIConstants.sRect / 4
                                color:                  UIConstants.blackColor
                            }
                            Rectangle {
                                id: centerCam
                                anchors.horizontalCenter:   parent.horizontalCenter
                                anchors.verticalCenter: parent.verticalCenter
                                width:                      2
                                height:                     barCam.height
                                color:                      UIConstants.textColor
                            }
                            Rectangle{
                                id: indicatorCam
                                anchors.verticalCenter: parent.verticalCenter
                                width:                  parent.height/2
                                height:                 width
                                radius:                 width / 2
                                color:                  UIConstants.textColor
                                x:                      (value*(inverted?-1:1)+32768)/65535 * parent.width - width/2
                            }
                        }
                        QComboBox{
                            id: cbxAxisCam
                            width: parent.height*4
                            height: parent.height
                            model: mapAxisCamKeys
                            currentIndex: mapAxisCam[mapFuncConfig]+1
                            onCurrentTextChanged: {
                                if(currentText !== mapFuncConfig){
                                    joystick.mapAxis(id,cbxAxisCam.currentText,!invertedConfig,false,true);
                                }
                            }
                        }
                        Row {
                            id: chbInvertCam
                            width: parent.height*2.5
                            height: parent.height
                            spacing: 5
                            Rectangle{
                                width: parent.height
                                height: parent.height
                                radius: 3
                                Rectangle{
                                    id: rectCheckCam
                                    visible: invertedConfig
                                    color: "#555"
                                    border.color: "#333"
                                    radius: 1
                                    anchors.margins: parent.height / 4
                                    anchors.fill: parent
                                }
                                MouseArea{
                                    anchors.fill: parent
                                    onClicked: {
                                        joystick.mapAxis(id,cbxAxisCam.currentText,!invertedConfig,false,true);
                                    }
                                }
                            }
                            Label{
                                height: parent.height
                                verticalAlignment: Label.AlignVCenter
                                horizontalAlignment: Label.AlignLeft
                                font.pixelSize: UIConstants.fontSize
                                font.family: UIConstants.appFont
                                color: UIConstants.textColor
                                text: itemListName["INVERT"]
                                      [UIConstants.language[UIConstants.languageID]]
                            }
                        }
                    }
                }
            } // Column - Axis Monitor
        }
    }
    Connections{
        target: joystick
        onJoystickConnected:{
            if(state){
                var joystickInfo = joystick.task.getJoystickInfo(
                            joystick.task.joyID);
                lblJSName.text = joystickInfo["NAME"];
                lblJSVersion.text = joystickInfo["VERSION"];
                lblJSAxes.text = joystickInfo["AXES"];
                lblJSButtons.text = joystickInfo["BUTTONS"];
            }
        }
    }

    Component.onCompleted: {
        cbxListJoystick.model = joystick.task.getListJoystick();
    }
} // SetupPage
