/**
 * ===============================================================
 * @Project: %{ProjectName}
 * @Module: QComboBox
   @Breif:
 * @Author:
 * @Date: 4/21/20 12:28 PM
 * Language: QML
 * @License: (C) Viettel Aerospace Institude - Viettel Group
 * ================================================================
 */


//--- Include QML core components
import QtQuick          2.3
import QtQuick.Controls 1.2
import QtQuick.Controls.Styles 1.4
import QtQuick.Controls.Private 1.0

import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0
ComboBox{
    width: UIConstants.sRect*6
    height: UIConstants.sRect*1.5
    style: ComboBoxStyle {
        id: comboBox
        background: Rectangle {
            id: rectCategory
            color: UIConstants.textColor
            Rectangle{
                anchors.right: parent.right
                anchors.top: parent.top
                anchors.bottom: parent.bottom
                width: height
                color: UIConstants.transparentColor
                Canvas{
                    anchors.fill: parent
                    anchors.margins: UIConstants.sRect/3
                    onPaint: {
                        var ctx = getContext("2d");
                        var drawColor = UIConstants.blackColor
                        ctx.lineWidth = 2;
                        ctx.strokeStyle = UIConstants.grayColor;
                        ctx.beginPath();
                        ctx.moveTo(0,0);
                        ctx.lineTo(width/2,height);
                        ctx.lineTo(width,0);
                        ctx.closePath();
                        ctx.stroke();
                    }
                }
            }
        }
        label: Text {
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignLeft
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
//                                font.capitalization: Font.AllUppercase
            color: "black"
            text: control.currentText
        }

        // drop-down customization here
        property Component __dropDownStyle: MenuStyle {
            __maxPopupHeight: UIConstants.sRect*20
            __menuItemType: "comboboxitem"

            frame: Rectangle {              // background
                color: "#fff"
            }

            itemDelegate.label:             // an item text
                Text {
                verticalAlignment: Text.AlignVCenter
                horizontalAlignment: Text.AlignHCenter
                font.pixelSize: UIConstants.fontSize
                font.family: UIConstants.appFont
//                                    font.capitalization: Font.AllUppercase
                color: styleData.selected ? "white" : "black"
                text: styleData.text
            }

            itemDelegate.background: Rectangle {  // selection of an item
                radius: 2
                color: styleData.selected ? "darkGray" : "transparent"
            }

            __scrollerStyle: ScrollViewStyle { }
        }

        property Component __popupStyle: Style {
            property int __maxPopupHeight: 400
            property int submenuOverlap: 0

            property Component frame: Rectangle {
                width: (parent ? parent.contentWidth : 0)
                height: (parent ? parent.contentHeight : 0) + 2
                border.color: "black"
                property real maxHeight: 500
                property int margin: 1
            }

            property Component menuItemPanel: Text {
                text: "NOT IMPLEMENTED"
                color: "red"
                font {
                    pixelSize: 14
                    bold: true
                }
            }

            property Component __scrollerStyle: null
        }
    }
}

