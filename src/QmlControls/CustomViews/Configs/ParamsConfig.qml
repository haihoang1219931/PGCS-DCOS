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
import QtQuick.Controls.Styles 1.4
import QtQuick.Dialogs  1.2
import QtQuick.Layouts 1.3

import CustomViews.Components   1.0
import CustomViews.UIConstants  1.0
import CustomViews.Dialogs      1.0
/// ConfigPage
Rectangle {
    id: rootItem
    color: UIConstants.transparentColor
    width: 1376
    height: 768
    property var vehicle
    Rectangle{
        id: rectSearch
        width: UIConstants.sRect*10
        height: UIConstants.sRect*1
        color: UIConstants.transparentColor
        border.color: UIConstants.grayColor
        radius: 1
        TextField{
            id: txtSearch
            anchors.fill: parent
            clip: true
            placeholderText: "Param name filter"
            inputMethodHints: Qt.ImhPreferUppercase
            font.family: UIConstants.appFont
            font.pixelSize: UIConstants.fontSize
            font.capitalization: Font.AllUppercase
            style:TextFieldStyle{
                background:Rectangle{
                    color: UIConstants.transparentColor
                    border.width: 1
                    border.color: UIConstants.textColor
                }
                textColor: UIConstants.textColor
                placeholderTextColor: UIConstants.grayColor
            }

        }
    }
    ListView {
        id: listView
        clip: true
        anchors.top: rectSearch.bottom
        anchors.topMargin: 8
        anchors.bottom: parent.bottom
        anchors.bottomMargin: UIConstants.sRect
        anchors.right: parent.right
        anchors.rightMargin: 8
        anchors.left: parent.left
        anchors.leftMargin: 8
        model: vehicle.paramsModel
//            model: ListModel {
//                ListElement {selected: false;name: "Param A"; unit:"m/s"; value:"12"}
//                ListElement {selected: true;name: "Param B"; unit:"m/s"; value:"12"}
//                ListElement {selected: true;name: "Param C"; unit:"m/s"; value:"12"}
//                ListElement {selected: false;name: "Param D"; unit:"m/s"; value:"12"}
//            }
        delegate: Rectangle {
            id: rectItem
            height: visible?UIConstants.sRect:0
            visible: name === "" || name.includes(txtSearch.text.toUpperCase())
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
//                anchors.right: lblUnit.left
                width: UIConstants.sRect*5
                anchors.bottom: parent.bottom
                anchors.top: parent.top
                anchors.leftMargin: 2

                TextInput {
                    id: lblValue
                    text: value
                    verticalAlignment: Text.AlignVCenter
                    horizontalAlignment: Text.AlignLeft
                    anchors.fill: parent
                    anchors.leftMargin: 8
                    color: UIConstants.textColor
                    font.pixelSize: UIConstants.fontSize
                    font.family: UIConstants.appFont
                    Keys.onReturnPressed: {
                        console.log("Keys.onReturnPressed")
                        if(!footerBar.isShowConfirm){
                            footerBar.isShowConfirm = true;
                            var compo = Qt.createComponent("qrc:/CustomViews/Dialogs/ConfirmDialog.qml");
                            var confirmDialogObj = compo.createObject(rootItem,{
                                "title":"Are you sure to want to change \n  value of param ["+lblName.text+"] to "+lblValue.text,
                                "type": "CONFIRM",
                                "x":rootItem.width / 2 - UIConstants.sRect * 13 / 2,
                                "y":rootItem.height / 2 - UIConstants.sRect * 6 / 2,
                                "z":200});
                            confirmDialogObj.clicked.connect(function (type,func){
                                footerBar.isShowConfirm = false;
                                if(func === "DIALOG_OK"){
                                    vehicle.paramsController._writeParameterRaw(lblName.text,lblValue.text);
                                }else if(func === "DIALOG_CANCEL"){

                                }
                                confirmDialogObj.destroy();
                                compo.destroy();

                            });
                        }
                    }
                }
            }

            Label {
                id: lblUnit
                width: 20
                height: 17
                text: unit
                color: UIConstants.textColor
                font.pixelSize: UIConstants.fontSize
                font.family: UIConstants.appFont
                anchors.right: parent.right
                anchors.rightMargin: 8
                anchors.verticalCenter: parent.verticalCenter
                horizontalAlignment: Text.AlignLeft
            }


        }
    }
} // ConfigPage



