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

import CustomViews.Components 1.0 as CustomViews
import CustomViews.UIConstants 1.0
/// Joystick Config
Flickable {
    id: rootItem
    property string pageName:           qsTr("Joystick")
    property string pageDescription:    qsTr("Joystick Setup is used to configure a calibrate joysticks.")
    width: 1280
    height: 768
    contentWidth: width
    contentHeight: 1000
    readonly property real _maxButtons: 16

    property int smallFontPointSize: UIConstants.fontSize

    property int axisRoll: 2
    property int axisPitch: 1
    property int axisYaw: 0
    property int axisThrottle: 3
    property bool thrushNegative: false

    property color color: UIConstants.transparentColor
    clip: true


    Item {
        anchors.fill: parent
        anchors.topMargin: 10
        anchors.leftMargin: 10
        anchors.rightMargin: 10
        property bool controllerCompleted:      false
        property bool controllerAndViewReady:   false

        readonly property real labelToMonitorMargin: UIConstants.defaultTextWidth * 3

        // Live axis monitor control component
        Component {
            id: axisMonitorDisplayComponent

            Item {
                property int axisValue: 0
                property int deadbandValue: 0
                property bool narrowIndicator: true
                property color deadbandColor: "#8c161a"

                property color  __barColor: UIConstants.grayColor

                // Bar
                Rectangle {
                    id:                     bar
                    anchors.verticalCenter: parent.verticalCenter
                    width:                  parent.width
                    height:                 parent.height / 3
                    color:                  parent.__barColor
                }

                // Deadband
                Rectangle {
                    id:                     deadbandBar
                    anchors.verticalCenter: parent.verticalCenter
                    x:                      _deadbandPosition
                    width:                  _deadbandWidth
                    height:                 parent.height / 2
                    color:                  UIConstants.grayColor
                    visible:                false

                    property real _percentDeadband:    ((2 * deadbandValue) / (32768.0 * 2))
                    property real _deadbandWidth:   parent.width * _percentDeadband
                    property real _deadbandPosition:   (parent.width - _deadbandWidth) / 2
                }

                // Center point
                Rectangle {
                    anchors.horizontalCenter:   parent.horizontalCenter
                    width:                      UIConstants.defaultTextWidth / 2
                    height:                     parent.height
                    color:                      rootItem.color
                }

                // Indicator
                Rectangle {
                    anchors.verticalCenter: parent.verticalCenter
                    width:                  10
                    height:                 10
                    x:                      (reversed ? (parent.width - _indicatorPosition) : _indicatorPosition) - (width / 2)
                    radius:                 width / 2
                    color:                  UIConstants.textColor

                    property real _percentAxisValue:    ((axisValue + 32768.0) / (32768.0 * 2))
                    property real _indicatorPosition:   parent.width * _percentAxisValue
                }

                Label {
                    anchors.fill:           parent
                    horizontalAlignment:    Text.AlignHCenter
                    verticalAlignment:      Text.AlignVCenter
                    text:                   qsTr("Not Mapped")
                    visible: false
                    color: UIConstants.textColor
                    font.pixelSize: UIConstants.fontSize
                    font.family: UIConstants.appFont
                }

                ColorAnimation {
                    id:         barAnimation
                    target:     bar
                    property:   "color"
                    from:       "yellow"
                    to:         __barColor
                    duration:   1500
                }


                // Axis value debugger
                /*
                Label {
                    anchors.fill: parent
                    text: axisValue
                }
                */

            }
        } // Component - axisMonitorDisplayComponent

        // Main view Qml starts here

        // Left side column
        Column {
            id:                     leftColumn
            anchors.rightMargin:    UIConstants.defaultFontPixelWidth
            anchors.left:           parent.left
            anchors.right:          rightColumn.left
            spacing:                10
            // Attitude Controls
            Column {
                width:      parent.width
                spacing:    5

                Label {
                    text: qsTr("Attitude Controls")
                    color: UIConstants.textColor
                    font.pixelSize: UIConstants.fontSize
                    font.family: UIConstants.appFont
                }

                Item {
                    width:  parent.width
                    height: UIConstants.defaultTextHeight * 2

                    Label {
                        id:     rollLabel
                        width:  UIConstants.defaultTextWidth * 10
                        text:   qsTr("Roll")
                        color: UIConstants.textColor
                        font.pixelSize: UIConstants.fontSize
                        font.family: UIConstants.appFont
                    }

                    Loader {
                        id:                 rollLoader
                        anchors.left:       rollLabel.right
                        anchors.right:      parent.right
                        height:             UIConstants.defaultFontPixelHeight
                        width:              100
                        sourceComponent:    axisMonitorDisplayComponent

                        property bool mapped:           true
                        property bool reversed:         true
                    }
                    Connections{
                        target: joystick
                        onAxisValueChanged:{
                            if (axisID === axisRoll) {
                                rollLoader.item.axisValue = -value;
                            }
                        }
                    }
                }

                Item {
                    width:  parent.width
                    height: UIConstants.defaultTextHeight * 2

                    Label {
                        id:     pitchLabel
                        width:  UIConstants.defaultTextWidth * 10
                        text:   qsTr("Pitch")
                        color: UIConstants.textColor
                        font.pixelSize: UIConstants.fontSize
                        font.family: UIConstants.appFont
                    }

                    Loader {
                        id:                 pitchLoader
                        anchors.left:       pitchLabel.right
                        anchors.right:      parent.right
                        height:             UIConstants.defaultFontPixelHeight
                        width:              100
                        sourceComponent:    axisMonitorDisplayComponent

                        property bool mapped:           true
                        property bool reversed:         true
                    }
                    Connections{
                        target: joystick
                        onAxisValueChanged:{
                            if (axisID === axisPitch) {
                                pitchLoader.item.axisValue = -value;
                            }
                        }
                    }
                }

                Item {
                    width:  parent.width
                    height: UIConstants.defaultTextHeight * 2

                    Label {
                        id:     yawLabel
                        width:  UIConstants.defaultTextWidth * 10
                        text:   qsTr("Yaw")
                        color: UIConstants.textColor
                        font.pixelSize: UIConstants.fontSize
                        font.family: UIConstants.appFont
                    }

                    Loader {
                        id:                 yawLoader
                        anchors.left:       yawLabel.right
                        anchors.right:      parent.right
                        height:             UIConstants.defaultFontPixelHeight
                        width:              100
                        sourceComponent:    axisMonitorDisplayComponent
                        property bool mapped:           true
                        property bool reversed:         true
                    }
                    Connections{
                        target: joystick
                        onAxisValueChanged:{
                            if (axisID === axisYaw) {
                                yawLoader.item.axisValue = -value;
                            }
                        }
                    }
                }

                Item {
                    width:  parent.width
                    height: UIConstants.defaultTextHeight * 2

                    Label {
                        id:     throttleLabel
                        width:  UIConstants.defaultTextWidth * 10
                        text:   qsTr("Throttle")
                        color: UIConstants.textColor
                        font.pixelSize: UIConstants.fontSize
                        font.family: UIConstants.appFont
                    }

                    Loader {
                        id:                 throttleLoader
                        anchors.left:       throttleLabel.right
                        anchors.right:      parent.right
                        height:             UIConstants.defaultFontPixelHeight
                        width:              100
                        sourceComponent:    axisMonitorDisplayComponent

                        property bool mapped:           true
                        property bool reversed:         true
                    }
                    Connections{
                        target: joystick
                        onAxisValueChanged:{
                            if (axisID === axisThrottle) {
                                var throttle = throttleLoader.item.axisValue;
                                console.log("throttle with value =" + value);
                                var tmpThrottle = (throttle + value/60);
                                if(tmpThrottle < -32768) tmpThrottle = -32768;
                                else if(tmpThrottle > 32768) tmpThrottle = 32768;
                                throttleLoader.item.axisValue = tmpThrottle;
                            }
                        }
                    }
                }
            } // Column - Attitude Control labels

            // Command Buttons
            Row {
                spacing: 10

                Button {
                    id:     skipButton
                    text:   qsTr("Skip")

                }

                Button {
                    id:     cancelButton
                    text:   qsTr("Cancel")

                }

                Button {
                    id:         nextButton
                    text:       qsTr("Calibrate")
                }
            } // Row - Buttons

            // Status Text
            Label {
                id:         statusText
                width:      parent.width
                wrapMode:   Text.WordWrap
            }

            Rectangle {
                width:          parent.width
                height:         1
                border.color:   UIConstants.textColor
                border.width:   1
            }

            // Settings

            Row {
                width:      parent.width
                spacing:    UIConstants.defaultFontPixelWidth

                // Left column settings
                Column {
                    width:      parent.width / 2
                    spacing:    UIConstants.defaultFontPixelHeight

                    Label {
                        text: qsTr("Additional Joystick settings:")
                        color: UIConstants.textColor
                        font.pixelSize: UIConstants.fontSize
                        font.family: UIConstants.appFont
                    }

                    Column {
                        width:      parent.width
                        spacing:    UIConstants.defaultFontPixelHeight


                        CheckBox {
                            id:         enabledCheckBox
                            font.family: UIConstants.appFont
                            font.pixelSize: UIConstants.fontSize
                            text:       "<font font-family=\""+UIConstants.appFont+"\" color=\""+UIConstants.textColor+"\">"+"Enable joystick input"+"</font>"
                        }

                        Row {
                            width:      parent.width
                            spacing:    UIConstants.defaultFontPixelWidth

                            Label {
                                id:                 activeJoystickLabel
                                anchors.baseline:   joystickCombo.baseline
                                text:               qsTr("Active joystick:")
                                color: UIConstants.textColor
                                font.pixelSize: UIConstants.fontSize
                                font.family: UIConstants.appFont
                            }

                            ComboBox {
                                id:                 joystickCombo
                                width:              parent.width - activeJoystickLabel.width - parent.spacing

                            }
                        }

                        Column {
                            spacing: UIConstants.defaultFontPixelHeight / 3

                            OldCtrl.ExclusiveGroup { id: throttleModeExclusiveGroup }

                            OldCtrl.RadioButton {
                                id: rbtnCenterJoystick
                                exclusiveGroup: throttleModeExclusiveGroup
                                style: RadioButtonStyle{
                                    label: Label{
                                        text: rbtnCenterJoystick.text
                                        font.pixelSize: UIConstants.fontSize
                                        font.family: UIConstants.appFont
                                        anchors.margins: 0
                                        horizontalAlignment: Label.left
                                    }
                                }
                                text:           "<font font-family=\""+UIConstants.appFont+"\" color=\""+UIConstants.textColor+"\">Center stick is zero throttle</font>"
                            }

                            Row {
                                x:          20
                                width:      parent.width
                                spacing:    UIConstants.defaultFontPixelWidth
                                CheckBox {
                                    id:         accumulator
                                    font.family: UIConstants.appFont
                                    font.pixelSize: UIConstants.fontSize
                                    text:       "<font font-family=\""+UIConstants.appFont+"\" color=\""+UIConstants.textColor+"\">Spring loaded throttle smoothing</font>"

                                }
                            }

                            OldCtrl.RadioButton {
                                id: rbtnFullThrottle
                                exclusiveGroup: throttleModeExclusiveGroup
                                style: RadioButtonStyle{
                                    label: Label{
                                        text: rbtnFullThrottle.text
                                        font.pixelSize: UIConstants.fontSize
                                        font.family: UIConstants.appFont
                                        anchors.margins: 0
                                        horizontalAlignment: Label.left
                                    }
                                }
                                text:           "<font font-family=\""+UIConstants.appFont+"\" color=\""+UIConstants.textColor+"\">Full down stick is zero throttle</font>"
                            }

                            CheckBox {
                                id:             negativeThrust
                                font.family: UIConstants.appFont
                                font.pixelSize: UIConstants.fontSize
                                text:           "<font font-family=\""+UIConstants.appFont+"\" color=\""+UIConstants.textColor+"\">Allow negative Thrust</font>"
                            }
                        }

                        Column {
                            spacing: UIConstants.defaultFontPixelHeight / 3

                            Label {
                                id:                 expoSliderLabel
                                text:               qsTr("Exponential:")
                                color: UIConstants.textColor
                                font.pixelSize: UIConstants.fontSize
                                font.family: UIConstants.appFont
                            }

                            Row {
                                OldCtrl.Slider {
                                    id: expoSlider
                                    minimumValue: 0
                                    maximumValue: 0.75
                                 }

                                Label {
                                    id:     expoSliderIndicator
                                    text:   expoSlider.value.toFixed(2)
                                    color: UIConstants.textColor
                                    font.pixelSize: UIConstants.fontSize
                                    font.family: UIConstants.appFont
                                }
                            }
                        }

                        CheckBox {
                            id:         advancedSettings
                            checked:    false
                            font.family: UIConstants.appFont
                            font.pixelSize: UIConstants.fontSize
                            text:       "<font font-family=\""+UIConstants.appFont+"\" color=\""+UIConstants.textColor+"\">"+"Advanced settings (careful!)"+"</font>"
//                            property bool checked: true
                        }

                        Row {
                            width:      parent.width
                            spacing:    UIConstants.defaultFontPixelWidth
                            visible:    advancedSettings.checked

                            Label {
                                id:                 joystickModeLabel
                                anchors.baseline:   joystickModeCombo.baseline
                                text:               qsTr("Joystick mode:")
                                color: UIConstants.textColor
                                font.pixelSize: UIConstants.fontSize
                                font.family: UIConstants.appFont
                            }

                            ComboBox {
                                id:             joystickModeCombo
                            }
                        }

                        Row {
                            width:      parent.width
                            spacing:    UIConstants.defaultFontPixelWidth
                            visible:    advancedSettings.checked
                            Label {
                                text:       qsTr("Message frequency (Hz):")
                                anchors.verticalCenter: parent.verticalCenter
                                color: UIConstants.textColor
                                font.pixelSize: UIConstants.fontSize
                                font.family: UIConstants.appFont
                            }
                            TextField {
                                validator:  DoubleValidator { bottom: 0.25; top: 100.0; }
                                inputMethodHints: Qt.ImhFormattedNumbersOnly
                            }
                        }

                        Row {
                            width:      parent.width
                            spacing:    UIConstants.defaultFontPixelWidth
                            visible:    advancedSettings.checked
                            CheckBox {
                                id:         joystickCircleCorrection
                                font.family: UIConstants.appFont
                                font.pixelSize: UIConstants.fontSize
                                text:       "<font color=\""+UIConstants.textColor+"\">"+qsTr("Enable circle correction")+"</font>"

                            }
                        }

                        Row {
                            width:      parent.width
                            spacing:    UIConstants.defaultFontPixelWidth
                            visible:    advancedSettings.checked
                            CheckBox {
                                id:         deadband
                                font.family: UIConstants.appFont
                                font.pixelSize: UIConstants.fontSize
                                text:       "<font color=\""+UIConstants.textColor+"\">"+qsTr("Deadbands")+"</font>"
                            }
                        }
                        Row{
                            width: parent.width
                            spacing: UIConstants.defaultFontPixelWidth
                            visible: advancedSettings.checked
                            Label{
                                width:       parent.width * 0.85
                                wrapMode:           Text.WordWrap
                                text:   qsTr("Deadband can be set during the first ") +
                                        qsTr("step of calibration by gently wiggling each axis. ") +
                                        qsTr("Deadband can also be adjusted by clicking and ") +
                                        qsTr("dragging vertically on the corresponding axis monitor.")
                                color: UIConstants.textColor
                                font.pixelSize: UIConstants.fontSize
                                font.family: UIConstants.appFont
                            }
                        }
                    }
                } // Column - left column

                // Right column settings
                Column {
                    width:      parent.width / 2
                    spacing:    UIConstants.defaultFontPixelHeight

                    Label {
                        text: qsTr("Button actions:")
                        color: UIConstants.textColor
                        font.pixelSize: UIConstants.fontSize
                        font.family: UIConstants.appFont
                    }

                    Column {
                        width:      parent.width
                        spacing:    UIConstants.defaultFontPixelHeight / 3

                        Repeater {
                            id:     buttonActionRepeater
                            model: joystick.buttons
                            Row {
                                spacing: UIConstants.defaultFontPixelWidth
                                CheckBox {
                                    anchors.verticalCenter:     parent.verticalCenter
                                }

                                Rectangle {
                                    anchors.verticalCenter:     parent.verticalCenter
                                    width:                      UIConstants.defaultFontPixelHeight * 1.5
                                    height:                     width
                                    border.width:               1
                                    border.color:               UIConstants.grayColor
                                    color:                      pressed ? "green" : rootItem.color

                                    Label {
                                        anchors.fill:           parent
                                        color:                  UIConstants.textColor
                                        horizontalAlignment:    Text.AlignHCenter
                                        verticalAlignment:      Text.AlignVCenter
                                        text:                   Number(id).toString()
                                        font.pixelSize: UIConstants.fontSize
                                        font.family: UIConstants.appFont
                                    }
                                }

                                ComboBox {
                                    id:             buttonActionCombo
                                    width:          UIConstants.defaultFontPixelWidth * 20
                                    model: vehicle.flightModes
                                }
                            }
                        } // Repeater
                    } // Column
                } // Column - right setting column
            } // Row - Settings
        } // Column - Left Main Column

        // Right side column
        Column {
            id:             rightColumn
            x: 455
            width: 185
            height: 67
            anchors.top:    parent.top
            anchors.right:  parent.right
            anchors.rightMargin: 30
            spacing:        UIConstants.defaultFontPixelHeight / 2

            Row {
                spacing: UIConstants.defaultFontPixelWidth

                OldCtrl.ExclusiveGroup { id: modeGroup }

                Label {
                    text: "TX Mode:"
                    color: UIConstants.textColor
                }

                OldCtrl.RadioButton {
                    id: rbtnTXMode1
                    exclusiveGroup: modeGroup
                    style: RadioButtonStyle{
                        label: Label{
                            text: rbtnTXMode1.text
                            font.pixelSize: UIConstants.fontSize
                            font.family: UIConstants.appFont
                            anchors.margins: 0
                            horizontalAlignment: Label.left
                        }
                    }
                    text:           "<font color=\""+UIConstants.textColor+"\">1</font>"
                }

                OldCtrl.RadioButton {
                    id: rbtnTXMode2
                    exclusiveGroup: modeGroup
                    style: RadioButtonStyle{
                        label: Label{
                            text: rbtnTXMode2.text
                            font.pixelSize: UIConstants.fontSize
                            font.family: UIConstants.appFont
                            anchors.margins: 0
                            horizontalAlignment: Label.left
                        }
                    }
                    text:           "<font color=\""+UIConstants.textColor+"\">2</font>"
                }

                OldCtrl.RadioButton {
                    id: rbtnTXMode3
                    exclusiveGroup: modeGroup
                    style: RadioButtonStyle{
                        label: Label{
                            text: rbtnTXMode3.text
                            font.pixelSize: UIConstants.fontSize
                            font.family: UIConstants.appFont
                            anchors.margins: 0
                            horizontalAlignment: Label.left
                        }
                    }
                    text:           "<font color=\""+UIConstants.textColor+"\">3</font>"
                }

                OldCtrl.RadioButton {
                    id: rbtnTXMode4
                    exclusiveGroup: modeGroup
                    style: RadioButtonStyle{
                        label: Label{
                            text: rbtnTXMode4.text
                            font.pixelSize: UIConstants.fontSize
                            font.family: UIConstants.appFont
                            anchors.margins: 0
                            horizontalAlignment: Label.left
                        }
                    }
                    text:           "<font color=\""+UIConstants.textColor+"\">4</font>"
                }
            }

            Image {
                width:      parent.width
                fillMode:   Image.PreserveAspectFit
                smooth:     true
            }

            // Axis monitor
            Column {
                width:      parent.width
                spacing:    5

                Label {
                    text: qsTr("Axis Monitor")
                    color: UIConstants.textColor
                }
                Connections{
                    target: joystick
                    onAxisValueChanged:{
                        if (axisMonitorRepeater.itemAt(axisID)) {
                            console.log("axis["+axisID+"]"+value);
                            axisMonitorRepeater.itemAt(axisID).loader.item.axisValue = value;
                        }
                    }
                }

                Repeater {
                    id:     axisMonitorRepeater
                    width:  parent.width
                    model: joystick.axes
                    Row {
                        spacing:    5

                        // Need this to get to loader from Connections above
                        property Item loader: theLoader

                        Label {
                            id:     axisLabel
                            text:   Number(id).toString()
                            color: UIConstants.textColor
                        }

                        Loader {
                            id:                     theLoader
                            anchors.verticalCenter: axisLabel.verticalCenter
                            height:                 UIConstants.defaultFontPixelHeight
                            width:                  200
                            sourceComponent:        axisMonitorDisplayComponent
                            Component.onCompleted:  item.narrowIndicator = true

                            property bool mapped:               true
                            readonly property bool reversed:    false

                            MouseArea {
                                id:             deadbandMouseArea
                                anchors.fill:   parent.item

                                property real startY

                                onPressed: {
                                    startY = mouseY
                                    parent.item.deadbandColor = "#3C6315"
                                }
                                onPositionChanged: {
                                    var newValue = parent.item.deadbandValue + (startY - mouseY)*15
                                    if ((newValue > 0) && (newValue <32768)){parent.item.deadbandValue=newValue;}
                                    startY = mouseY
                                }
                                onReleased: {
                                    parent.item.deadbandColor = "#8c161a"
                                }
                            }
                        }

                    }
                }
            } // Column - Axis Monitor

            // Button monitor
            Column {
                width:      parent.width
                spacing:    UIConstants.defaultFontPixelHeight

                Label {
                    text: qsTr("Button Monitor")
                    color: UIConstants.textColor
                }

                Flow {
                    width:      parent.width
                    spacing:    -1

                    Repeater {
                        id:     buttonMonitorRepeater
                        model: joystick.buttons
                        Rectangle {
                            width:          UIConstants.defaultFontPixelHeight * 1.5
                            height:         width
                            border.width:   1
                            border.color:   UIConstants.grayColor
                            color:          pressed ? "green" : rootItem.color
                            Label {
                                anchors.fill:           parent
                                color:                  UIConstants.textColor
                                horizontalAlignment:    Text.AlignHCenter
                                verticalAlignment:      Text.AlignVCenter
                                text:                   Number(id).toString()
                            }
                        }
                    } // Repeater
                } // Row
            } // Column - Axis Monitor
        } // Column - Right Column
    }


    Component.onCompleted: {
//        if (controllerCompleted) {
//            controllerAndViewReady = true
//        }
        var listJs = joystick.getListJoystick();
        joystickCombo.model = listJs;
        joystick.setJoyID("/dev/input/js0");
        joystick.start();
    }
} // SetupPage




/*##^## Designer {
    D{i:102;anchors_height:756}
}
 ##^##*/
