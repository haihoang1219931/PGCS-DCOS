/**
 * ==============================================================================
 * @Project: FCS-Groundcontrol-based
 * @Module: PreflightCheck page
 * @Breif:
 * @Author: Trung Nguyen
 * @Date: 22/03/2019
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
import QtQml.Models 2.11

//------------------ Include Custom modules/plugins
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0
import CustomViews.Pages 1.0

//------------------ Item definition
Rectangle {
    id: rootItem
    color: UIConstants.bgColorOverlay
    border.color: UIConstants.grayColor
    border.width: 1
    property var itemListName:
        UIConstants.itemTextMultilanguages["PRECHECK"]
    property int vehicleType: 0
    function changeVehicleType(vehicleType){
        rootItem.vehicleType = vehicleType;
        if(vehicleType === 2 ||
                vehicleType === 14){
            sidebarGeneralConfigs.model.setProperty(2,"showed_",false);
            sidebarGeneralConfigs.model.setProperty(3,"showed_",false);
            sidebarGeneralConfigs.model.setProperty(7,"showed_",false);
        }else{
            sidebarGeneralConfigs.model.setProperty(2,"showed_",true);
            sidebarGeneralConfigs.model.setProperty(3,"showed_",true);
            sidebarGeneralConfigs.model.setProperty(7,"showed_",true);
        }
        loadCheckList();
    }
    function reload(){
        for(var i=0; i< sidebarGeneralConfigs.model.count; i++){
            sidebarGeneralConfigs.model.setProperty(i,"state_","uncheck");
        }
        sidebarGeneralConfigs.currentIndex = 0;
    }

    function loadCheckList(){
        sidebarGeneralConfigs.model.clear();
        sidebarGeneralConfigs.model.append({state_: "uncheck",showed_: true,
                   text_: itemListName["MODECHECK"]["MENU_TITTLE"]
                   [UIConstants.language[UIConstants.languageID]] });
        sidebarGeneralConfigs.model.append({state_: "uncheck",showed_: true,
                   text_: itemListName["PROPELLERS"]["MENU_TITTLE"]
                   [UIConstants.language[UIConstants.languageID]] });
        sidebarGeneralConfigs.model.append({state_: "uncheck",showed_: rootItem.vehicleType != 2 &&
                                                                       rootItem.vehicleType != 14,
                   text_: itemListName["STEERING"]["MENU_TITTLE"]
                   [UIConstants.language[UIConstants.languageID]] });
        sidebarGeneralConfigs.model.append({state_: "uncheck",showed_: rootItem.vehicleType != 2 &&
                                                                       rootItem.vehicleType != 14,
                   text_: itemListName["PITOT"]["MENU_TITTLE"]
                   [UIConstants.language[UIConstants.languageID]] });
        sidebarGeneralConfigs.model.append({state_: "uncheck",showed_: true,
                   text_: itemListName["LASER"]["MENU_TITTLE"]
                   [UIConstants.language[UIConstants.languageID]] });
        sidebarGeneralConfigs.model.append({state_: "uncheck",showed_: true,
                   text_: itemListName["GPS"]["MENU_TITTLE"]
                   [UIConstants.language[UIConstants.languageID]] });
        sidebarGeneralConfigs.model.append({state_: "uncheck",showed_: true,
                   text_: itemListName["JOYSTICK"]["MENU_TITTLE"]
                   [UIConstants.language[UIConstants.languageID]] });
        sidebarGeneralConfigs.model.append({state_: "uncheck",showed_: rootItem.vehicleType != 2 &&
                                                                       rootItem.vehicleType != 14,
                   text_: itemListName["RPM"]["MENU_TITTLE"]
                   [UIConstants.language[UIConstants.languageID]] });
        sidebarGeneralConfigs.model.append({state_: "uncheck",showed_: true,
                   text_: itemListName["RESULT"]["MENU_TITTLE"]
                   [UIConstants.language[UIConstants.languageID]] });
    }

    RowLayout {
        anchors.fill: parent
        //---------- Sidebar
        CheckList {
            id: sidebarGeneralConfigs
            Layout.preferredWidth: UIConstants.sRect * 9
            Layout.preferredHeight: parent.height
            RectBorder {
                type: "right"
            }
            model: ListModel {                
            }
        }

        Connections{
            target: UIConstants
            onLanguageIDChanged:{
                loadCheckList()
                changeVehicleType(rootItem.vehicleType);
            }
        }

        StackLayout {
            id: checkingContentStack
            Layout.preferredWidth: parent.width - sidebarGeneralConfigs.width
            Layout.preferredHeight: parent.height
            Layout.leftMargin: -5
            currentIndex: sidebarGeneralConfigs.currentIndex
            ColumnLayout {
                ModeCheck{
                    Layout.alignment: Qt.AlignBottom
                    Layout.preferredWidth: checkingContentStack.width
                    Layout.preferredHeight: UIConstants.sRect * 2 * 9
                    Layout.fillHeight: true
                }
            }

            ColumnLayout {
                Propellers{
                    Layout.alignment: Qt.AlignBottom
                    Layout.preferredWidth: checkingContentStack.width
                    Layout.preferredHeight: UIConstants.sRect * 2 * 9
                    Layout.fillHeight: true
                }
            }

            ColumnLayout {
                Steering{
                    Layout.alignment: Qt.AlignBottom
                    Layout.preferredWidth: checkingContentStack.width
                    Layout.preferredHeight: UIConstants.sRect * 2 * 9
                    Layout.fillHeight: true
                }
            }

            ColumnLayout {
                Pitot{
                    Layout.alignment: Qt.AlignBottom
                    Layout.preferredWidth: checkingContentStack.width
                    Layout.preferredHeight: UIConstants.sRect * 2 * 9
                    Layout.fillHeight: true
                }
            }

            ColumnLayout {
                Laser{
                    Layout.alignment: Qt.AlignBottom
                    Layout.preferredWidth: checkingContentStack.width
                    Layout.preferredHeight: UIConstants.sRect * 2 * 9
                    Layout.fillHeight: true
                }
            }

            ColumnLayout {
                GPS{
                    Layout.alignment: Qt.AlignBottom
                    Layout.preferredWidth: checkingContentStack.width
                    Layout.preferredHeight: UIConstants.sRect * 2 * 9
                    Layout.fillHeight: true
                }
            }

            ColumnLayout {
                JoystickCheck{
                    Layout.alignment: Qt.AlignBottom
                    Layout.preferredWidth: checkingContentStack.width
                    Layout.preferredHeight: UIConstants.sRect * 2 * 9
                    Layout.fillHeight: true
                }
            }

            ColumnLayout {
                RPM{
                    Layout.alignment: Qt.AlignBottom
                    Layout.preferredWidth: checkingContentStack.width
                    Layout.preferredHeight: UIConstants.sRect * 2 * 9
                    Layout.fillHeight: true
                }
            }

            ColumnLayout {
                Success{
                    id: precheckResult
                    Layout.alignment: Qt.AlignBottom
                    Layout.preferredWidth: checkingContentStack.width
                    Layout.preferredHeight: UIConstants.sRect * 2 * 9
                    Layout.fillHeight: true
                }
            }
            onCurrentIndexChanged: {
                if(currentIndex === sidebarGeneralConfigs.model.count - 1){
                    precheckResult.showResult(sidebarGeneralConfigs.model);
                }
            }
        }
    }
    Rectangle{
        anchors.fill: parent
        color: UIConstants.transparentColor
        border.width: 1
        border.color: UIConstants.grayColor
    }

    ProcessingOverlay {
        id: processingOverlay
        width: parent.width * 4 / 5
        height: parent.height * 9 / 10
        anchors{ right: parent.right; top: parent.top }
        z: 10
        opacity: 0
        Behavior on opacity {
            NumberAnimation {
                duration: 300
                easing.type: Easing.InCubic
            }
        }
    }

    //------------------Js supported func
    function next()
    {
        if(sidebarGeneralConfigs.model.get(sidebarGeneralConfigs.currentIndex).state_ === "uncheck"
                && sidebarGeneralConfigs.currentIndex>=0
                && sidebarGeneralConfigs.currentIndex < sidebarGeneralConfigs.model.count - 1){
            sidebarGeneralConfigs.model.get(sidebarGeneralConfigs.currentIndex).state_ = "failed";
        }
        sidebarGeneralConfigs.next();
    }

    function prev()
    {
        sidebarGeneralConfigs.prev();
    }

    function doCheck()
    {
        if(sidebarGeneralConfigs.currentIndex !==
                sidebarGeneralConfigs.model.count - 1){
            sidebarGeneralConfigs.doCheck();
            sidebarGeneralConfigs.next();
        }

    }
    Component.onCompleted: {
        loadCheckList();
    }
}
