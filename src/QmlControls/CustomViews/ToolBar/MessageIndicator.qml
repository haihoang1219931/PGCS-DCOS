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
    property bool showIndicator: false
    property bool   _isMessageImportant:  vehicle.messageSecurity !== "MSG_INFO"
    signal clicked()
    IconSVG {
        id: messageIcon
        source: "qrc:/qmlimages/ToolBar/Images/Megaphone.svg"
        color:  vehicle.uas.messages.length > 0? (!_isMessageImportant?
                                                                  UIConstants.greenColor:UIConstants.redColor) :
                                                             UIConstants.textColor
        anchors.horizontalCenter: parent.horizontalCenter
        width:              iconSize
        height:             iconSize
        opacity: 0.6
    }
    Label {
        color: messageIcon.color
        text: vehicle ? Number(vehicle.uas.messages.length).toString() : ""
        anchors.top: parent.top
        anchors.topMargin: 2
        anchors.left: messageIcon.right
        anchors.leftMargin: 2
        font.family: UIConstants.appFont
        font.pixelSize: UIConstants.fontSize
        opacity: 0.6
    }
    //-- System Message Area
    Rectangle {
        id:                         messageArea
        width:                      UIConstants.sRect * 20
        height:                     UIConstants.sRect * 10
        anchors.horizontalCenter:   parent.horizontalCenter
        anchors.top:                parent.top
        anchors.topMargin:          UIConstants.sRect * 2 + UIConstants.defaultFontPixelHeight
        radius:                     UIConstants.rectRadius
        color:                      UIConstants.transparentBlue
        border.color:               UIConstants.grayColor
        visible:                    showIndicator

        MouseArea {
            // This MouseArea prevents the Map below it from getting Mouse events. Without this
            // things like mousewheel will scroll the Flickable and then scroll the map as well.
            anchors.fill:       parent
            preventStealing:    true
            onWheel:            wheel.accepted = true
        }
        ListView {
            id:                 messageFlick
            anchors.margins:    UIConstants.defaultFontPixelHeight
            anchors.fill:       parent
            pixelAligned:       true
            clip:               true
            model: vehicle.uas.messages
            delegate: TextEdit {
                id:             messageText
                readOnly:       true
                textFormat:     TextEdit.RichText
                color:          formatedColor
                text:           formatedText
                font.pixelSize: UIConstants.fontSize
            }
            onCountChanged:{
                console.log("On count changed")
                positionViewAtEnd();
            }
        }
        //-- Dismiss System Message
        FlatButtonIcon {
            anchors.margins:    UIConstants.defaultFontPixelHeight * 0.5
            anchors.top:        parent.top
            anchors.right:      parent.right
            width:              UIConstants.isMobile ? UIConstants.defaultFontPixelHeight * 1.5 : UIConstants.defaultFontPixelHeight
            height:             width
            icon: UIConstants.iChatClose
            isSolid:            true
            isShowRect:         false
            smooth:             true
            iconSize: UIConstants.fontSize
            MouseArea {
                anchors.fill:       parent
                anchors.margins:    UIConstants.isMobile ? -UIConstants.defaultFontPixelHeight : 0
                onClicked: {
                    rootItem.clicked();
                }
            }
        }
        //-- Clear Messages
        FlatButtonIcon {
            anchors.bottom:     parent.bottom
            anchors.right:      parent.right
            anchors.margins:    UIConstants.defaultFontPixelHeight * 0.5
            height:             UIConstants.isMobile ? UIConstants.defaultFontPixelHeight * 1.5 : UIConstants.defaultFontPixelHeight
            width:              height
            icon: UIConstants.iTrash
            isSolid:            true
            isShowRect:         false
            smooth:             true
            iconSize: UIConstants.fontSize
            MouseArea {
                anchors.fill:   parent
                onClicked: {
                    if(true) {
                        vehicle.uas.messages = [];
                        rootItem.clicked();
                    }
                }
            }
        }
    }    

    MouseArea {
        anchors.fill:   parent
        onClicked: {
            rootItem.clicked();
        }
    }
}
