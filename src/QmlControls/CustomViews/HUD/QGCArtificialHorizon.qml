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
 *   @brief QGC Artificial Horizon
 *   @author Gus Grubba <mavlink@grubba.com>
 */

import QtQuick 2.3

Item {
    id: root
    property real rollAngle :   0
    property real pitchAngle:   0
    clip:           true
    anchors.fill:   parent

    property real angularScale: pitchAngle * root.height / 45

    Item {
        id: artificialHorizon
        width:  root.width  * 4
        height: root.height * 8
        anchors.centerIn: parent
        Rectangle {
            id: sky
            anchors.fill: parent
            smooth: true
            antialiasing: true
            gradient: Gradient {
                GradientStop { position: 0.25; color: Qt.hsla(0.6, 1.0, 0.25) }
                GradientStop { position: 0.5;  color: Qt.hsla(0.6, 0.5, 0.55) }
            }
        }
        Rectangle {
            id: ground
            height: sky.height / 2
            anchors {
                left:   sky.left;
                right:  sky.right;
                bottom: sky.bottom
            }
            smooth: true
            antialiasing: true
            gradient: Gradient {
                GradientStop { position: 0.0;  color: Qt.rgba(0.76, 0.54, 0.19,1) }
                GradientStop { position: 0.5; color: Qt.rbga(0.64, 0.42, 0.11,1) }
            }
        }
        transform: [
            Translate {
                y:  angularScale
            },
            Rotation {
                origin.x: artificialHorizon.width  / 2
                origin.y: artificialHorizon.height / 2
                angle:    -rollAngle
            }]
    }
}
