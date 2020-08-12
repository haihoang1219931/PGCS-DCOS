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
import QtQuick.Dialogs  1.2
import QtQuick.Layouts 1.3
import QtQuick.Window 2.11
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0
/// ConfigPage
Rectangle {
    id: rootItem
    color: UIConstants.transparentColor
    width: 1376
    height: 768
    ColumnLayout{
        anchors.top: parent.top
        anchors.topMargin: UIConstants.sRect
        anchors.horizontalCenter: parent.horizontalCenter
        width: UIConstants.sRect* 20
        QLabel{
            Layout.fillWidth: true
            text: "desktopAvailableWidth: " + Screen.desktopAvailableWidth
        }
        QLabel{
            Layout.fillWidth: true
            text: "desktopAvailableHeight: " + Screen.desktopAvailableHeight
        }
        QLabel{
            Layout.fillWidth: true
            text: "devicePixelRatio: " + Screen.devicePixelRatio
        }
        QLabel{
            Layout.fillWidth: true
            text: "width: " + Screen.width
        }
        QLabel{
            Layout.fillWidth: true
            text: "height: " + Screen.height
        }

        QLabel{
            Layout.fillWidth: true
            text: "manufacturer: " + Screen.manufacturer
        }
        QLabel{
            Layout.fillWidth: true
            text: "model: " + Screen.model
        }
        QLabel{
            Layout.fillWidth: true
            text: "name: " + Screen.name
        }
        QLabel{
            Layout.fillWidth: true
            text: "orientation: " + Screen.orientation
        }
        QLabel{
            Layout.fillWidth: true
            text: "serialNumber: " + Screen.serialNumber
        }
        QLabel{
            Layout.fillWidth: true
            text: "virtualX: " + Screen.virtualX
        }
        QLabel{
            Layout.fillWidth: true
            text: "virtualY: " + Screen.virtualY
        }
        QLabel{
            Layout.fillWidth: true
            text: "pixelDensity: " + Screen.pixelDensity
        }
    }

} // ConfigPage



