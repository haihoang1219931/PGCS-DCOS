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
Flickable {
    id: rootItem
    property string pageName:           qsTr("Joystick")
    property string pageDescription:    qsTr("Joystick Setup is used to configure a calibrate joysticks.")
    width: 1280
    height: 768
    clip: true
    contentWidth: width
    contentHeight: 1000
    property var mapAxisKeys: ["Unused","Yaw","Throttle","Roll","Pitch"]
    property var mapAxis: {"Unused":-1,"Roll":2,"Pitch":3,"Yaw":0,"Throttle":1}
    property var mapButtonKeys: ["Unused","Stabilize","Loiter","Auto","PIC/CIC","CIC/PIC"]
    property var mapButton: {"Unused":-1,"Stabilize":0,"Loiter":1,"Auto":2,"PIC/CIC":3,"CIC/PIC":4}
    Row{
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
                    text: qsTr("Joystick selection")
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
                    text: "Select"
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
                    text: "Name:"
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
                    text: "Version:"
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
                    text: "Axes:"
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
                    text: "Buttons:"
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
                    text: "Save joystick"
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
                    text: "Reset default"
                    onClicked: {
                        joystick.resetConfig();
                    }
                }
            }
        }

        Column {
            id: clmAxis
            width: UIConstants.sRect* 25
            spacing:    UIConstants.sRect/2
            Label {
                text: qsTr("Axis Monitor")
                color: UIConstants.textColor
                font.pixelSize: UIConstants.fontSize
                font.family: UIConstants.appFont
            }

            Repeater {
                id:     axisMonitorRepeater
                width:  parent.width
                model: joystick.axesConfig

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
                        currentIndex: mapAxis[mapFunc]+1
                        onCurrentTextChanged: {
                            if(currentText !== mapFunc){
                                joystick.mapAxisConfig(id,cbxAxis.currentText,!inverted);
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
                                visible: inverted
                                color: "#555"
                                border.color: "#333"
                                radius: 1
                                anchors.margins: parent.height / 4
                                anchors.fill: parent
                            }
                            MouseArea{
                                anchors.fill: parent
                                onClicked: {
                                    joystick.mapAxisConfig(id,cbxAxis.currentText,!inverted);
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
                            text: "Invert"
                        }
                    }
                }
            }
        } // Column - Axis Monitor

        // Button monitor
        Column {
            id: clmButton
            width: UIConstants.sRect* 20
            spacing:    UIConstants.sRect/2

            Label {
                text: qsTr("Button Monitor")
                color: UIConstants.textColor
                font.pixelSize: UIConstants.fontSize
                font.family: UIConstants.appFont
            }

            Repeater {
                id:     buttonMonitorRepeater
                width:  parent.width
                model: joystick.buttonsConfig
                delegate: Row{
                    spacing: UIConstants.sRect/2
                    height: UIConstants.sRect*1.5
                    Rectangle {
                        width:          parent.height
                        height:         width
                        border.width:   1
                        border.color:   UIConstants.grayColor
                        color:          pressed ? "green" : UIConstants.transparentColor
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
                        currentIndex: mapButton[mapFunc]+1
                        onCurrentTextChanged: {
                            if(currentText !== mapFunc)
                                joystick.mapButtonConfig(id,cbxButton.currentText)
                        }
                    }
                }
            } // Repeater
        } // Column - Axis Monitor
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
        joystick.mapFile = "conf/joystick.conf"
        joystick.start();
    }
} // SetupPage



