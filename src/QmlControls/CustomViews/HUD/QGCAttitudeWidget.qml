/****************************************************************************
 *
 *   (c) 2009-2016 QGROUNDCONTROL PROJECT <http://www.qgroundcontrol.org>
 *
 * QGroundControl is licensed according to the terms in the file
 * COPYING.md in the root of the source code directory.
 *
 ****************************************************************************/


/**
 * @file
 *   @brief QGC Attitude Instrument
 *   @author Gus Grubba <mavlink@grubba.com>
 */

import QtQuick              2.3
import QtGraphicalEffects   1.0

import CustomViews.UIConstants 1.0
import CustomViews.Components 1.0
Item {
    id: root

    property bool showPitch:    true
//    property var  vehicle:      null
    property real size: UIConstants.sRect * 5
    property bool showHeading:  false

    property real _rollAngle:   vehicle ? vehicle.roll  : 0
    property real _pitchAngle:  vehicle ? vehicle.pitch : 0

    width:  size
    height: size

//    QGCPalette { id: qgcPal; colorGroupEnabled: enabled }

    Item {
        id:             instrument
        anchors.fill:   parent

        //----------------------------------------------------
        //-- Artificial Horizon
        QGCArtificialHorizon {
            rollAngle:          _rollAngle
            pitchAngle:         _pitchAngle
            anchors.fill:       parent
        }
        //----------------------------------------------------
        //-- Pointer
        Image {
            id:                 pointer
            source:             "/qmlimages/attitudePointer.svg"
            mipmap:             true
            fillMode:           Image.PreserveAspectFit
            anchors.fill:       parent
            sourceSize.height:  parent.height
        }
        //----------------------------------------------------
        //-- Instrument Dial
        Image {
            id:                 instrumentDial
            source:             "/qmlimages/attitudeDial.svg"
            mipmap:             true
            fillMode:           Image.PreserveAspectFit
            anchors.fill:       parent
            sourceSize.height:  parent.height
            transform: Rotation {
                origin.x:       root.width  / 2
                origin.y:       root.height / 2
                angle:          -_rollAngle
            }
        }
        //----------------------------------------------------
        //-- Pitch
        QGCPitchIndicator {
            id:                 pitchWidget
            visible:            root.showPitch
            size:               root.size * 0.5
            anchors.verticalCenter: parent.verticalCenter
            pitchAngle:         _pitchAngle
            rollAngle:          _rollAngle
            color:              Qt.rgba(0,0,0,0)
        }
        //----------------------------------------------------
        //-- Cross Hair
        Image {
            id:                 crossHair
            anchors.centerIn:   parent
            source:             "/qmlimages/crossHair.svg"
            mipmap:             true
            width:              size * 0.75
            sourceSize.width:   width
            fillMode:           Image.PreserveAspectFit
        }
        property bool rounded: true
        property bool adapt: true
        layer.enabled: rounded
        layer.effect: OpacityMask {
            maskSource: Item {
                width: instrument.width
                height: instrument.height
                Rectangle {
                    anchors.centerIn: parent
                    width: instrument.adapt ? instrument.width : Math.min(instrument.width, instrument.height)
                    height: instrument.adapt ? instrument.height : width
                    radius: Math.min(width, height)
                }
            }
        }
    }

    Rectangle {
        id:             borderRect
        anchors.fill:   parent
        radius:         width / 2
        color:          Qt.rgba(0,0,0,0)
        border.color:   UIConstants.sidebarBorderColor
        border.width:   1
    }

    Text {
        anchors.bottomMargin:       10
        anchors.bottom:             parent.bottom
        anchors.horizontalCenter:   parent.horizontalCenter
        text:                       _headingString3
        color:                      "white"
        visible:                    showHeading

        property string _headingString: vehicle ? Number(vehicle.heading).toFixed(0).toString() : "OFF"
        property string _headingString2: _headingString.length === 1 ? "0" + _headingString : _headingString
        property string _headingString3: _headingString2.length === 2 ? "0" + _headingString2 : _headingString2
    }
}
