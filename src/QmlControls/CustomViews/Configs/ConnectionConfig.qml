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
import QtQuick.Layouts 1.3

import CustomViews.Components   1.0
import CustomViews.UIConstants  1.0
import CustomViews.Dialogs      1.0
/// ConfigPage
Rectangle {
    id: root
    color: UIConstants.transparentColor
    width: 1376
    height: 768
    property var vehicle
    property var itemListName:
        UIConstants.itemTextMultilanguages["CONFIGURATION"]["CONNECTION"]
    property var config
    Row{
        id: rectSelect
        anchors.topMargin: 8
        anchors.left: parent.left
        anchors.leftMargin: 8
        anchors.top: parent.top
        spacing: UIConstants.sRect/2
        QComboBox{
            id: cbxListConfig
            width: UIConstants.sRect * 6
            height: UIConstants.sRect * 1.5
            onCurrentIndexChanged: {
                if(cbxListConfig.currentIndex === 0){
                    root.config = FCSConfig;
                }else if(cbxListConfig.currentIndex === 1){
                    root.config = TRKConfig;
                }else if(cbxListConfig.currentIndex === 2){
                    root.config = PCSConfig
                }else if(cbxListConfig.currentIndex === 3){
                    root.config = UcApiConfig
                }
            }
        }
    }

    ListView {
        id: listView
        clip: true
        anchors.top: rectSelect.bottom
        anchors.topMargin: 8
        anchors.bottom: parent.bottom
        anchors.bottomMargin: UIConstants.sRect
        anchors.right: parent.right
        anchors.rightMargin: 8
        anchors.left: parent.left
        anchors.leftMargin: 8
        model: root.config.paramsModel
        delegate: Rectangle {
            id: rectItem
            height: visible?UIConstants.sRect:0
            width: listView.width
            color: UIConstants.transparentColor
            MouseArea{
                anchors.fill: parent
                hoverEnabled: true
                onEntered: {
                    rectItem.color = UIConstants.transparentBlue;
                }
                onExited: {
                    rectItem.color = UIConstants.transparentColor;
                }
            }
            Label {
                id: lblName
                width: UIConstants.sRect*10
                height: UIConstants.sRect
                text: name
                anchors.verticalCenter: parent.verticalCenter
                verticalAlignment: Text.AlignVCenter
                horizontalAlignment: Text.AlignLeft
                color: UIConstants.textColor
                font.pixelSize: UIConstants.fontSize
                font.family: UIConstants.appFont
            }

            Rectangle {
                id: rectValue
                color: UIConstants.bgAppColor
                anchors.bottomMargin: 2
                anchors.topMargin: 2
                anchors.rightMargin: 2
                anchors.left: lblName.right
                width: UIConstants.sRect*7
                anchors.bottom: parent.bottom
                anchors.top: parent.top
                anchors.leftMargin: 2

                TextInput {
                    id: lblValue
                    text: focus?text:value
                    verticalAlignment: Text.AlignVCenter
                    horizontalAlignment: Text.AlignLeft
                    anchors.fill: parent
                    anchors.leftMargin: 8
                    color: UIConstants.textColor
                    font.pixelSize: UIConstants.fontSize
                    font.family: UIConstants.appFont
                    clip: true
                }
            }

            Label {
                id: lblUnit
                width: UIConstants.sRect * 2
                height: UIConstants.sRect - 4
                color: UIConstants.textColor
                font.pixelSize: UIConstants.fontSize
                font.family: UIConstants.appFont
                anchors.left: rectValue.right
                anchors.leftMargin: 8
                anchors.verticalCenter: parent.verticalCenter
                horizontalAlignment: Text.AlignLeft
            }
            FlatButtonText{
                isAutoReturn: true
                width: UIConstants.sRect * 3
                height: UIConstants.sRect - 4
                anchors.left: lblUnit.left
                anchors.leftMargin: 8
                anchors.verticalCenter: parent.verticalCenter
                color: isEnable?UIConstants.greenColor:UIConstants.grayColor
                text: "Save"
                isEnable: lblValue.text != value
                onClicked: {
                    root.config.setPropertyValue(lblName.text,lblValue.text);
                    lblValue.focus = false;
                }
            }

        }
    }
    Component.onCompleted: {
        if(CAMERA_CONTROL){
            if(UC_API){
                cbxListConfig.model = ["Flight","Tracker","Camera","UC"];
            }else{
                cbxListConfig.model = ["Flight","Tracker","Camera"];
            }
        }else{
            if(UC_API){
                cbxListConfig.model = ["Flight","Tracker","UC"];
            }else{
                cbxListConfig.model = ["Flight","Tracker"];
            }
        }
    }
} // ConfigPage



