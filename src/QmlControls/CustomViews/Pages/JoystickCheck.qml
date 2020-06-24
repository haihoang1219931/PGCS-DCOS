/**
 * ==============================================================================
 * @Project: FCS-Groundcontrol-based
 * @Module: PreflightCheck page
 * @Breif:
 * @Author: Hai Nguyen Hoang
 * @Date: 14/05/2019
 * @Language: QML
 * @License: (c) Viettel Aerospace Institude - Viettel Group
 * ============================================================================
 */

//------------------ Include QT libs ------------------------------------------
import QtQuick.Window 2.2
import QtQuick 2.6
import QtQuick.Controls 2.1
import QtQuick.Layouts 1.3
import QtGraphicalEffects 1.0
import QtQuick 2.0

//------------------ Include Custom modules/plugins
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0
Rectangle{
    id: root
    width: 600
    height: 600
    color: "transparent"
    property var itemListName:
        UIConstants.itemTextMultilanguages["PRECHECK"]["JOYSTICK"]
    QLabel {
        id: lblTitle
        height: 54
        text: itemListName["TITTLE"]
              [UIConstants.language[UIConstants.languageID]]
        anchors.right: parent.right
        anchors.rightMargin: 8
        anchors.left: parent.left
        anchors.top: parent.top
        anchors.leftMargin: 8
        anchors.topMargin: 0
        border.width: 0
    }

    Column {
        id: clmAxis
        anchors.top: lblTitle.bottom
        anchors.topMargin: 8
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 8
        spacing:    UIConstants.sRect/2
        anchors.right: clmRCIn.left
        anchors.rightMargin: 8
        anchors.left: parent.left
        anchors.leftMargin: 8
        Label {
            text: itemListName["AXIS_MONITOR"]
                  [UIConstants.language[UIConstants.languageID]]
            color: UIConstants.textColor
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
        }
        Repeater {
            id:     axisMonitorRepeater
            width:  parent.width
            model: joystick.axesConfig

            delegate:Row {
                id: rowAxis
                spacing: UIConstants.sRect/2
                height: UIConstants.sRect*1.5
                width: parent.width
                Label {
                    id:     axisLabel
                    width:          parent.height
                    height:         width
                    text:   Number(id).toString()
                    horizontalAlignment: Label.AlignHCenter
                    verticalAlignment: Label.AlignVCenter
                    anchors.verticalCenter: parent.verticalCenter
                    color: UIConstants.textColor
                    font.pixelSize: UIConstants.fontSize
                    font.family: UIConstants.appFont
                }

                Item{
                    id: axisItem
                    width: clmAxis.width - axisLabel.width - cbxAxis.width -
                           //                           chbInvert.width -
                           //                               btnSaveAxis.width -
                           rowAxis.spacing * (rowAxis.children.length - 1)
                    height: parent.height
                    anchors.verticalCenter: parent.verticalCenter
                    Rectangle {
                        id:                     bar
                        anchors.verticalCenter: parent.verticalCenter
                        width:                  parent.width
                        height:                 UIConstants.sRect / 4
                        color:                  UIConstants.blackColor
                    }
                    Rectangle {
                        id: center
                        anchors.horizontalCenter:   parent.horizontalCenter
                        anchors.verticalCenter: parent.verticalCenter
                        width:                      2
                        height:                     bar.height
                        color:                      UIConstants.textColor
                    }
                    Rectangle{
                        id: indicator
                        anchors.verticalCenter: parent.verticalCenter
                        width:                  parent.height/2
                        height:                 width
                        radius:                 width / 2
                        color:                  UIConstants.textColor
                        x:                      (value*(inverted?-1:1)+32768)/65535 * parent.width - width/2
                    }
                }
                Label {
                    id: cbxAxis
                    width: parent.height*4
                    height: parent.height
                    text: mapFunc
                    horizontalAlignment: Label.AlignHCenter
                    verticalAlignment: Label.AlignVCenter
                    anchors.verticalCenter: parent.verticalCenter
                    color: UIConstants.textColor
                    font.pixelSize: UIConstants.fontSize
                    font.family: UIConstants.appFont
                }
            }
        }
    }

    Column {
        id: clmRCIn
        width: UIConstants.sRect * 10
        anchors.rightMargin: 8
        anchors.bottom: parent.bottom
        anchors.right: parent.right
        spacing: UIConstants.sRect/2
        anchors.topMargin: 8
        anchors.top: lblTitle.bottom
        anchors.bottomMargin: 8
        Label {
            text: itemListName["RC_MONITOR"]
                  [UIConstants.language[UIConstants.languageID]]
            color: UIConstants.textColor
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
        }
        Column{
            spacing:    UIConstants.sRect/2
            Label {
                height: UIConstants.sRect*1.5
                text: qsTr("RCIN_chan1: ") + vehicle.rcinChan1 +" us"
                verticalAlignment: Label.AlignVCenter
                color: UIConstants.textColor
                font.pixelSize: UIConstants.fontSize
                font.family: UIConstants.appFont
            }
            Label {
                height: UIConstants.sRect*1.5
                text: qsTr("RCIN_chan2: ") + vehicle.rcinChan2 +" us"
                verticalAlignment: Label.AlignVCenter
                color: UIConstants.textColor
                font.pixelSize: UIConstants.fontSize
                font.family: UIConstants.appFont
            }
            Label {
                height: UIConstants.sRect*1.5
                text: qsTr("RCIN_chan4: ") + vehicle.rcinChan4 +" us"
                verticalAlignment: Label.AlignVCenter
                color: UIConstants.textColor
                font.pixelSize: UIConstants.fontSize
                font.family: UIConstants.appFont
            }
            Label {
                height: UIConstants.sRect*1.5
                text: qsTr("RCIN_chan3: ") + vehicle.rcinChan3  +" us"
                verticalAlignment: Label.AlignVCenter
                color: UIConstants.textColor
                font.pixelSize: UIConstants.fontSize
                font.family: UIConstants.appFont
            }
        }


    } // Column - Axis Monitor
}
