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
    RowLayout {
        anchors.fill: parent
        //---------- Sidebar
        CheckList {
            id: sidebarGeneralConfigs
            Layout.preferredWidth: UIConstants.sRect * 9
            Layout.preferredHeight: parent.height
            onDisplayActiveConfigBoard: {
                checkingContentStack.currentIndex = boardId_;
            }
            RectBorder {
                type: "right"
            }
            model: ListModel {
                Component.onCompleted: {
                    append({state_: "uncheck", text_: "ModeCheck" });
                    append({state_: "uncheck", text_: "Propellers" });
//                    append({state_: "uncheck", text_: "Steering" });
//                    append({state_: "uncheck", text_: "Pitot" });
                    append({state_: "uncheck", text_: "Laser" });
                    append({state_: "uncheck", text_: "GPS" });
                    append({state_: "uncheck", text_: "Joystick" });
//                    append({state_: "uncheck", text_: "RPM" });
//                    append({state_: "uncheck", text_: "Payload" });
                }
            }
        }

        StackLayout {
            id: checkingContentStack
            Layout.preferredWidth: parent.width - sidebarGeneralConfigs.width
            Layout.preferredHeight: parent.height
            Layout.leftMargin: -5
            currentIndex: 0
            ColumnLayout {
                SidebarTitle {
                    Layout.alignment: Qt.AlignTop
                    Layout.preferredWidth: parent.width
                    Layout.preferredHeight: UIConstants.sRect * 2
                    title: "Flight Mode check"
                    iconType: "\uf197"
                    xPosition: 20
                }
                ModeCheck{
                    Layout.alignment: Qt.AlignBottom
                    Layout.preferredWidth: checkingContentStack.width
                    Layout.preferredHeight: UIConstants.sRect * 2 * 9
                    Layout.fillHeight: true
                }
            }

            ColumnLayout {
                SidebarTitle {
                    Layout.alignment: Qt.AlignTop
                    Layout.preferredWidth: parent.width
                    Layout.preferredHeight: UIConstants.sRect * 2
                    visible: true
                    title: "Propeller check"
                    iconType: "\uf197"
                    xPosition: 20
                }
                Propellers{
                    Layout.alignment: Qt.AlignBottom
                    Layout.preferredWidth: checkingContentStack.width
                    Layout.preferredHeight: UIConstants.sRect * 2 * 9
                    Layout.fillHeight: true
                }
            }

//            ColumnLayout {
//                SidebarTitle {
//                    Layout.alignment: Qt.AlignTop
//                    Layout.preferredWidth: parent.width
//                    Layout.preferredHeight: UIConstants.sRect * 2
//                    visible: true
//                    title: "Steering check"
//                    iconType: "\uf197"
//                    xPosition: 20
//                }
//                Steering{
//                    Layout.alignment: Qt.AlignBottom
//                    Layout.preferredWidth: checkingContentStack.width
//                    Layout.preferredHeight: UIConstants.sRect * 2 * 9
//                    Layout.fillHeight: true
//                }
//            }

//            ColumnLayout {
//                SidebarTitle {
//                    Layout.alignment: Qt.AlignTop
//                    Layout.preferredWidth: parent.width
//                    Layout.preferredHeight: UIConstants.sRect * 2
//                    title: "Pitot check"
//                    iconType: "\uf197"
//                    xPosition: 20
//                }
//                Pitot{
//                    Layout.alignment: Qt.AlignBottom
//                    Layout.preferredWidth: checkingContentStack.width
//                    Layout.preferredHeight: UIConstants.sRect * 2 * 9
//                    Layout.fillHeight: true
//                }
//            }

            ColumnLayout {
                SidebarTitle {
                    Layout.alignment: Qt.AlignTop
                    Layout.preferredWidth: parent.width
                    Layout.preferredHeight: UIConstants.sRect * 2
                    title: "Altitude measurement Laser check"
                    iconType: "\uf197"
                    xPosition: 20
                }
                Laser{
                    Layout.alignment: Qt.AlignBottom
                    Layout.preferredWidth: checkingContentStack.width
                    Layout.preferredHeight: UIConstants.sRect * 2 * 9
                    Layout.fillHeight: true
                }
            }

            ColumnLayout {
                SidebarTitle {
                    Layout.alignment: Qt.AlignTop
                    Layout.preferredWidth: parent.width
                    Layout.preferredHeight: UIConstants.sRect * 2
                    visible: true
                    title: "UAV GPS Location checking"
                    iconType: "\uf197"
                    xPosition: 20
                }
                GPS{
                    Layout.alignment: Qt.AlignBottom
                    Layout.preferredWidth: checkingContentStack.width
                    Layout.preferredHeight: UIConstants.sRect * 2 * 9
                    Layout.fillHeight: true
                }
            }
            ColumnLayout {
                SidebarTitle {
                    Layout.alignment: Qt.AlignTop
                    Layout.preferredWidth: parent.width
                    Layout.preferredHeight: UIConstants.sRect * 2
                    visible: true
                    title: "Joystick action"
                    iconType: "\uf197"
                    xPosition: 20
                }
                JoystickCheck{
                    Layout.alignment: Qt.AlignBottom
                    Layout.preferredWidth: checkingContentStack.width
                    Layout.preferredHeight: UIConstants.sRect * 2 * 9
                    Layout.fillHeight: true
                }
            }
//            ColumnLayout {
//                SidebarTitle {
//                    Layout.alignment: Qt.AlignTop
//                    Layout.preferredWidth: parent.width
//                    Layout.preferredHeight: UIConstants.sRect * 2
//                    title: "UAV Propulsion engine check"
//                    iconType: "\uf197"
//                    xPosition: 20
//                }
//                RPM{
//                    Layout.alignment: Qt.AlignBottom
//                    Layout.preferredWidth: checkingContentStack.width
//                    Layout.preferredHeight: UIConstants.sRect * 2 * 9
//                    Layout.fillHeight: true
//                }
//            }

//            ColumnLayout {
//                SidebarTitle {
//                    Layout.alignment: Qt.AlignTop
//                    Layout.preferredWidth: parent.width
//                    Layout.preferredHeight: UIConstants.sRect * 2
//                    title: "Camera functions checking"
//                    iconType: "\uf197"
//                    xPosition: 20
//                }
//                Payload{
//                    Layout.alignment: Qt.AlignBottom
//                    Layout.preferredWidth: checkingContentStack.width
//                    Layout.preferredHeight: UIConstants.sRect * 2 * 9
//                    Layout.fillHeight: true
//                }
//            }

            ColumnLayout {
                SidebarTitle {
                    Layout.alignment: Qt.AlignTop
                    Layout.preferredWidth: parent.width
                    Layout.preferredHeight: UIConstants.sRect * 2
                    title: "Preflight check process"
                    iconType: "\uf197"
                    xPosition: 20
                }
                Success{
                    Layout.alignment: Qt.AlignBottom
                    Layout.preferredWidth: checkingContentStack.width
                    Layout.preferredHeight: UIConstants.sRect * 2 * 9
                    Layout.fillHeight: true
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
        sidebarGeneralConfigs.next();
    }

    function prev()
    {
        sidebarGeneralConfigs.prev();
    }

    function doCheck()
    {
//        processingOverlay.opacity = 1;
//        processingOverlay.textOverlay = "Checking. Please wait !"
        sidebarGeneralConfigs.doCheck();
        next();
    }
}
