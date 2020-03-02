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
 *   @brief QGC Compass Widget
 *   @author Gus Grubba <mavlink@grubba.com>
 */

import QtQuick              2.3
import QtGraphicalEffects   1.0

import CustomViews.UIConstants 1.0
import CustomViews.Components 1.0

Item {
    id:     root
    width:  size
    height: size

    property real size:     _defaultSize
//    property var  vehicle:  null

    property real _defaultSize: 10* (10)
    property real _sizeRatio:   1.2
    property int  _fontSize:    10 * _sizeRatio
    property real _heading:     vehicle ? vehicle.heading : 0

    Rectangle {
        id:             borderRect
        anchors.fill:   parent
        radius:         width / 2
        color:          UIConstants.sidebarActiveBg
        border.color:   UIConstants.sidebarBorderColor
        border.width:   1
    }

    Item {
        id:             instrument
        anchors.fill:   parent

        Image {
            id:                 pointer
            width:              size * 0.65
            source:             vehicle ? "/qmlimages/compassInstrumentArrow.svg" : ""
            mipmap:             true
            sourceSize.width:   width
            fillMode:           Image.PreserveAspectFit
            anchors.centerIn:   parent
            transform: Rotation {
                origin.x:       pointer.width  / 2
                origin.y:       pointer.height / 2
                angle:          _heading
            }
        }

        Image {
            id:                 compassDial
            source:             "/qmlimages/compassInstrumentDial.svg"
            mipmap:             true
            fillMode:           Image.PreserveAspectFit
            anchors.fill:       parent
            sourceSize.height:  parent.height
        }

        Rectangle {
            anchors.centerIn:   parent
            width:              size * 0.35
            height:             size * 0.2
            border.color:       UIConstants.textColor
            opacity:            0.65
            color: UIConstants.transparentBlue
            Text {
                text:               _headingString3
                font.pointSize:     _fontSize < 8 ? 8 : _fontSize;
                color:              UIConstants.textColor
                anchors.centerIn:   parent

                property string _headingString: vehicle ? _heading.toFixed(0) : "OFF"
                property string _headingString2: _headingString.length === 1 ? "0" + _headingString : _headingString
                property string _headingString3: _headingString2.length === 2 ? "0" + _headingString2 : _headingString2
            }
        }
    }
}
