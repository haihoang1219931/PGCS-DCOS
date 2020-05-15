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
//-- Message Indicator
Item {
    id: rootItem
    width:          height
    anchors.top:    parent.top
    anchors.bottom: parent.bottom
    property int iconSize: 30
    IconSVG {
        id: messageIcon
        source: "qrc:/qmlimages/ToolBar/Images/Signal.svg"
        color:  !vehicle.link ? UIConstants.textColor: UIConstants.greenColor

        anchors.horizontalCenter: parent.horizontalCenter
        width:              iconSize
        height:             iconSize
        opacity: 0.6
    }
    Label {
        color: messageIcon.color
        text: vehicle ? ( (!isNaN(vehicle.mavlinkLossPercent)?
                               Number(100.0 - vehicle.mavlinkLossPercent).toFixed(0).toString():"100")
                         + "%") : "100%"
        anchors.top: parent.top
        anchors.topMargin: 2
        anchors.left: messageIcon.right
        anchors.leftMargin: 5
        font.family: UIConstants.appFont
        font.pixelSize: UIConstants.fontSize
        opacity: 0.6
    }
    Label {
        id: lblSNRRemote
        color: messageIcon.color
        text: ""
        horizontalAlignment: Text.AlignRight
        anchors.right: messageIcon.left
        anchors.rightMargin: 5
        anchors.top: parent.top
        anchors.topMargin: 2
        font.family: UIConstants.appFont
        font.pixelSize: UIConstants.fontSize
        opacity: 0.6
    }
    Label {
        id: lblSNRLocal
        color: messageIcon.color
        text: ""
        anchors.bottom: messageIcon.bottom
        anchors.right: messageIcon.left
        anchors.rightMargin: 5
        verticalAlignment: Text.AlignBottom
        horizontalAlignment: Text.AlignRight
        font.family: UIConstants.appFont
        font.pixelSize: UIConstants.fontSize
        opacity: 0.6
    }
    Connections{
        target: comVehicle
        onTeleDataReceived:{
            if(srcAddr === "LOCAL"){
                lblSNRLocal.text = dataType;
            }else if(srcAddr === "REMOTE"){
                lblSNRRemote.text = dataType;
            }
        }
    }
}
