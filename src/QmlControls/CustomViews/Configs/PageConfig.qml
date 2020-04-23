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

import CustomViews.Components 1.0 as CustomViews
import CustomViews.UIConstants 1.0

import io.qdt.dev   1.0
/// ConfigPage
Rectangle {
    id: rootItem
    color: UIConstants.bgColorOverlay
    width: 1376
    height: 768
    property var vehicle
    signal clicked(string type, string func)
    CustomViews.SidebarTitle {
        id: sidebarTitle
        width: UIConstants.sRect * 9
        anchors { top: parent.top; left: parent.left; }
        height: UIConstants.sRect * 2
        visible: true
        title: "Menu settings"
        CustomViews.RectBorder {
            type: "right"
        }
    }

    Column {
        id: lstItem
        property int currentIndex: 0
        width: sidebarTitle.width
        anchors.topMargin: 0
        anchors.top: sidebarTitle.bottom
        anchors.left: parent.left
        anchors.leftMargin: 0
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 0

        Repeater{
            model: ListModel{
    //            ListElement{name: "Theme"; icon: UIConstants.iWindowStore}
                ListElement{id_: 0; btnText_: "Joystick"; icon_: "\uf11b"}
                ListElement{id_: 1; btnText_: "Screen monitor"; icon_: "\uf2d2"}
                ListElement{id_: 2; btnText_: "Parametes"; icon_: "\uf03c"}
                ListElement{id_: 3; btnText_: "Application"; icon_: "\uf109"}
            }
            delegate: Item {
                width: lstItem.width
                height: 60
                Rectangle {
                    anchors.fill: parent
                    color: lstItem.currentIndex === id_?
                        UIConstants.sidebarActiveBg:UIConstants.bgColorOverlay
                    Label {
                        id: textSide
                        text: btnText_
                        font.pixelSize: UIConstants.fontSize
                        font.family: UIConstants.appFont
                        color: lstItem.currentIndex === id_?
                                   UIConstants.textColor:UIConstants.textFooterColor
                        anchors.left: parent.left
                        anchors.leftMargin: parent.height
                        anchors.verticalCenter: parent.verticalCenter
                    }
                    Item {
                        width: parent.height
                        height: parent.height
                        Label {
                            id: iconSide
                            text: icon_
                            font { pixelSize: UIConstants.fontSize * 3 / 2; bold: true;family: ExternalFontLoader.solidFont }
                            color: textSide.color
                            anchors.centerIn: parent
                        }
                    }
                }

                CustomViews.RectBorder {
                    type: "bottom"
                }

                MouseArea{
                    anchors.fill: parent
                    onClicked: {
                        lstItem.currentIndex = id_;
                        if(id_ === 2){
                            rootItem.vehicle.paramsController._updateParamTimeout();
                        }
                    }
                }
            }
        }

    }
    StackLayout {
        id: swipeView
        anchors.top: parent.top
        anchors.left: lstItem.right
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        currentIndex: lstItem.currentIndex
        Item{
            CustomViews.SidebarTitle {
                id: sdbJoystick
                height: UIConstants.sRect * 2
                anchors.left: parent.left
                anchors.right: parent.right
                title: "Joystick Configuration"
                iconType: "\uf197"
                xPosition: 20
            }
            JoystickConfig{
                id: cfgJoystick
                anchors.top: sdbJoystick.bottom
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.bottom: parent.bottom
            }

        }
        Item{
            CustomViews.SidebarTitle {
                id: sdbTheme
                height: UIConstants.sRect * 2
                anchors.left: parent.left
                anchors.right: parent.right
                title: "Screen Monitor Information"
                iconType: "\uf197"
                xPosition: 20
            }
            ThemeConfig{
                anchors.top: sdbTheme.bottom
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.bottom: parent.bottom
            }
        }
        Item{
            CustomViews.SidebarTitle {
                id: sdbParameters
                height: UIConstants.sRect * 2
                anchors.left: parent.left
                anchors.right: parent.right
                title: "Parameter List"
                iconType: "\uf197"
                xPosition: 20
            }
            ParamsConfig{
                id: paramsConfig
                anchors.top: sdbParameters.bottom
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.bottom: parent.bottom
                vehicle: rootItem.vehicle
            }
        }
        Item{
            CustomViews.SidebarTitle {
                id: sdbApplication
                height: UIConstants.sRect * 2
                anchors.left: parent.left
                anchors.right: parent.right
                title: "Application config"
                iconType: "\uf197"
                xPosition: 20
            }
            AppConfig{
                id: appConfig
                anchors.top: sdbApplication.bottom
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.bottom: parent.bottom
                onClicked: {
                    rootItem.clicked(type,action);
                }
            }
        }
    }
    Rectangle{
        anchors.left: parent.left
        anchors.top: parent.top
        anchors.bottom: parent.bottom
        width: sidebarTitle.width
        color: UIConstants.transparentColor
        CustomViews.RectBorder {
            type: "right"
        }
    }
    Rectangle{
        anchors.fill: parent
        color: UIConstants.transparentColor
        border.width: 1
        border.color: UIConstants.grayColor
    }
} // ConfigPage



