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
    property var itemListName:
        UIConstants.itemTextMultilanguages["CONFIGURATION"]
    signal clicked(string type, string func)
    function showAdvancedConfig(enable){
        for(var i= 2; i< modelConfig.count ; i++){
            modelConfig.get(i).visible_ = enable;
        }

        if(!enable && lstItem.currentIndex > 1){
            lstItem.currentIndex = 0;
        }
    }

    CustomViews.SidebarTitle {
        id: sidebarTitle
        width: UIConstants.sRect * 9
        anchors { top: parent.top; left: parent.left; }
        height: UIConstants.sRect * 2
        visible: true
        title: itemListName["TITTLE"]
               [UIConstants.language[UIConstants.languageID]]
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
        Connections{
            target: UIConstants
            onLanguageIDChanged:{
                modelConfig.clear();
                modelConfig.append({btnText_: itemListName["APPLICATION"]["SIDEBAR"]
                           [UIConstants.language[UIConstants.languageID]], icon_: "\uf109", visible_: true});
                modelConfig.append({btnText_: itemListName["SCREEN"]["SIDEBAR"]
                           [UIConstants.language[UIConstants.languageID]], icon_: "\uf2d2", visible_: true});
                modelConfig.append({btnText_: itemListName["PARAMETERS"]["SIDEBAR"]
                           [UIConstants.language[UIConstants.languageID]], icon_: "\uf03c", visible_: false});
                modelConfig.append({btnText_: itemListName["JOYSTICK"]["SIDEBAR"]
                           [UIConstants.language[UIConstants.languageID]], icon_: "\uf11b", visible_: false});
                modelConfig.append({btnText_: itemListName["CONNECTION"]["SIDEBAR"]
                           [UIConstants.language[UIConstants.languageID]], icon_: "\uf0c1", visible_: false});
            }
        }

        Repeater{
            id: listConfig
            model: ListModel{
                id: modelConfig
            }
            delegate: Item {
                width: lstItem.width
                height: 60
                visible: visible_
                Rectangle {
                    anchors.fill: parent
                    color: lstItem.currentIndex === index?
                        UIConstants.sidebarActiveBg:UIConstants.bgColorOverlay
                    Label {
                        id: textSide
                        text: btnText_
                        font.pixelSize: UIConstants.fontSize
                        font.family: UIConstants.appFont
                        color: lstItem.currentIndex === index?
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
                        lstItem.currentIndex = index;
                        if(index === 2){
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
                id: sdbApplication
                height: UIConstants.sRect * 2
                anchors.left: parent.left
                anchors.right: parent.right
                title: itemListName["APPLICATION"]["TITTLE"]
                       [UIConstants.language[UIConstants.languageID]]
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
        Item{
            CustomViews.SidebarTitle {
                id: sdbTheme
                height: UIConstants.sRect * 2
                anchors.left: parent.left
                anchors.right: parent.right
                title: itemListName["SCREEN"]["TITTLE"]
                       [UIConstants.language[UIConstants.languageID]]
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
                title: itemListName["PARAMETERS"]["TITTLE"]
                       [UIConstants.language[UIConstants.languageID]]
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
                id: sdbJoystick
                height: UIConstants.sRect * 2
                anchors.left: parent.left
                anchors.right: parent.right
                title: itemListName["JOYSTICK"]["TITTLE"]
                       [UIConstants.language[UIConstants.languageID]]
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
                id: sdbConnection
                height: UIConstants.sRect * 2
                anchors.left: parent.left
                anchors.right: parent.right
                title: itemListName["CONNECTION"]["TITTLE"]
                       [UIConstants.language[UIConstants.languageID]]
                iconType: "\uf197"
                xPosition: 20
            }
            ConnectionConfig{
                id: cfgConnection
                anchors.top: sdbConnection.bottom
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.bottom: parent.bottom
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
    Component.onCompleted: {
        modelConfig.clear();
        modelConfig.append({btnText_: itemListName["APPLICATION"]["SIDEBAR"]
                   [UIConstants.language[UIConstants.languageID]], icon_: "\uf109", visible_: true});
        modelConfig.append({btnText_: itemListName["SCREEN"]["SIDEBAR"]
                   [UIConstants.language[UIConstants.languageID]], icon_: "\uf2d2", visible_: true});
        modelConfig.append({btnText_: itemListName["PARAMETERS"]["SIDEBAR"]
                   [UIConstants.language[UIConstants.languageID]], icon_: "\uf03c", visible_: false});
        modelConfig.append({btnText_: itemListName["JOYSTICK"]["SIDEBAR"]
                   [UIConstants.language[UIConstants.languageID]], icon_: "\uf11b", visible_: false});
        modelConfig.append({btnText_: itemListName["CONNECTION"]["SIDEBAR"]
                   [UIConstants.language[UIConstants.languageID]], icon_: "\uf0c1", visible_: false});
    }
} // ConfigPage



