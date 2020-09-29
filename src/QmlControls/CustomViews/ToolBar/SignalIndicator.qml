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
        color:  !vehicle.link ? UIConstants.textColor: UIConstants.navIconColor

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
        anchors.topMargin: UIConstants.sRect/4
        anchors.top: lblSNRRemote.bottom
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
            var value = parseInt(dataType)
//            console.log("value:"+value)
            if(value !== undefined && value !== null){
                if(srcAddr === "LOCAL"){
                    lblSNRLocal.text = "DH-SNR:" + dataType;
                    if(vehicle.link){
                        if(value < 10) lblSNRLocal.color = UIConstants.redColor;
                        else if(value >=15)lblSNRLocal.color = messageIcon.color;
                        else lblSNRLocal.color = UIConstants.orangeColor
                    }
                    else lblSNRLocal.color = messageIcon.color
                }else if(srcAddr === "REMOTE"){
                    lblSNRRemote.text = "VH-SNR:" + dataType;
                    if(vehicle.link){
                        if(value < 10) lblSNRRemote.color = UIConstants.redColor;
                        else if(value >=15)lblSNRRemote.color = messageIcon.color;
                        else lblSNRRemote.color = UIConstants.orangeColor
                    }
                    else lblSNRRemote.color = messageIcon.color
                }
            }
        }
    }
}
