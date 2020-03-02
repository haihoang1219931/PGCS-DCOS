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
            Layout.preferredWidth: 270
            Layout.preferredHeight: parent.height
            onDisplayActiveConfigBoard: {
                checkingContentStack.currentIndex = boardId_;
            }
            RectBorder {
                type: "right"
            }
        }

        StackLayout {
            id: checkingContentStack
            Layout.preferredWidth: parent.width - 270
            Layout.preferredHeight: parent.height
            Layout.leftMargin: -5
            currentIndex: 0
            Item {
                ColumnLayout {
                    anchors.fill: parent
                    //--- Title
                    Rectangle {
                        color: UIConstants.transparentColor
                        Layout.alignment: Qt.AlignTop
                        Layout.preferredWidth: parent.width
                        Layout.preferredHeight: 70
                        SidebarTitle {
                            id: title
                            anchors.fill: parent
                            visible: true
                            title: "Flight Mode check"
                            iconType: "\uf197"
                            xPosition: 20
                        }
                    }

                    Rectangle {
                        id: rectangle
                        color: UIConstants.transparentColor
                        Layout.alignment: Qt.AlignBottom
                        Layout.preferredWidth: parent.width
                        Layout.preferredHeight: 70 * 9
                        Layout.fillHeight: true
                        ModeCheck{
                            anchors.fill: parent
                        }
                    }
                }
            }

            Item {
                ColumnLayout {
                    anchors.fill: parent
                    //--- Title
                    Rectangle {
                        color: UIConstants.transparentColor
                        Layout.alignment: Qt.AlignTop
                        Layout.preferredWidth: parent.width
                        Layout.preferredHeight: 70
                        SidebarTitle {
                            id: title1
                            anchors.fill: parent
                            visible: true
                            title: "Propeller check"
                            iconType: "\uf197"
                            xPosition: 20
                        }
                    }

                    Rectangle {
                        color: UIConstants.transparentColor
                        Layout.alignment: Qt.AlignBottom
                        Layout.preferredWidth: parent.width
                        Layout.preferredHeight: 70 * 9
                        Layout.fillHeight: true
                        Propellers{
                            anchors.fill: parent
                        }
                    }
                }
            }

            Item {
                ColumnLayout {
                    anchors.fill: parent
                    //--- Title
                    Rectangle {
                        color: UIConstants.transparentColor
                        Layout.alignment: Qt.AlignTop
                        Layout.preferredWidth: parent.width
                        Layout.preferredHeight: 70
                        SidebarTitle {
                            id: title2
                            anchors.fill: parent
                            visible: true
                            title: "Steering check"
                            iconType: "\uf197"
                            xPosition: 20
                        }
                    }

                    Rectangle {
                        color: UIConstants.transparentColor
                        Layout.alignment: Qt.AlignBottom
                        Layout.preferredWidth: parent.width
                        Layout.preferredHeight: 70 * 9
                        Layout.fillHeight: true
                        Steering{
                            anchors.fill: parent
                        }
                    }
                }
            }

            Item {
                ColumnLayout {
                    anchors.fill: parent
                    //--- Title
                    Rectangle {
                        color: UIConstants.transparentColor
                        Layout.alignment: Qt.AlignTop
                        Layout.preferredWidth: parent.width
                        Layout.preferredHeight: 70
                        SidebarTitle {
                            id: title4
                            anchors.fill: parent
                            visible: true
                            title: "Pitot check"
                            iconType: "\uf197"
                            xPosition: 20
                        }
                    }

                    Rectangle {
                        color: UIConstants.transparentColor
                        Layout.alignment: Qt.AlignBottom
                        Layout.preferredWidth: parent.width
                        Layout.preferredHeight: 70 * 9
                        Layout.fillHeight: true
                        Pitot{
                            anchors.fill: parent
                        }
                    }
                }
            }

            Item {
                ColumnLayout {
                    anchors.fill: parent
                    //--- Title
                    Rectangle {
                        color: UIConstants.transparentColor
                        Layout.alignment: Qt.AlignTop
                        Layout.preferredWidth: parent.width
                        Layout.preferredHeight: 70
                        SidebarTitle {
                            id: title5
                            anchors.fill: parent
                            visible: true
                            title: "Altitude measurement Laser check"
                            iconType: "\uf197"
                            xPosition: 20
                        }
                    }

                    Rectangle {
                        color: UIConstants.transparentColor
                        Layout.alignment: Qt.AlignBottom
                        Layout.preferredWidth: parent.width
                        Layout.preferredHeight: 70 * 9
                        Layout.fillHeight: true
                        Laser{
                            anchors.fill: parent
                        }
                    }
                }
            }

            Item {
                ColumnLayout {
                    anchors.fill: parent
                    //--- Title
                    Rectangle {
                        color: UIConstants.transparentColor
                        Layout.alignment: Qt.AlignTop
                        Layout.preferredWidth: parent.width
                        Layout.preferredHeight: 70
                        SidebarTitle {
                            id: title7
                            anchors.fill: parent
                            visible: true
                            title: "UAV GPS Location checking"
                            iconType: "\uf197"
                            xPosition: 20
                        }
                    }

                    Rectangle {
                        color: UIConstants.transparentColor
                        Layout.alignment: Qt.AlignBottom
                        Layout.preferredWidth: parent.width
                        Layout.preferredHeight: 70 * 9
                        Layout.fillHeight: true
                        GPS{
                            anchors.fill: parent
                        }
                    }
                }
            }

            Item {
                ColumnLayout {
                    anchors.fill: parent
                    //--- Title
                    Rectangle {
                        color: UIConstants.transparentColor
                        Layout.alignment: Qt.AlignTop
                        Layout.preferredWidth: parent.width
                        Layout.preferredHeight: 70
                        SidebarTitle {
                            anchors.fill: parent
                            visible: true
                            title: "UAV Propulsion engine check"
                            iconType: "\uf197"
                            xPosition: 20
                        }
                    }

                    Rectangle {
                        color: UIConstants.transparentColor
                        Layout.alignment: Qt.AlignBottom
                        Layout.preferredWidth: parent.width
                        Layout.preferredHeight: 70 * 9
                        Layout.fillHeight: true
                        RPM{
                            anchors.fill: parent
                        }
                    }
                }
            }

            Item {
                ColumnLayout {
                    anchors.fill: parent
                    //--- Title
                    Rectangle {
                        color: UIConstants.transparentColor
                        Layout.alignment: Qt.AlignTop
                        Layout.preferredWidth: parent.width
                        Layout.preferredHeight: 70
                        SidebarTitle {
                            anchors.fill: parent
                            visible: true
                            title: "Camera functions checking"
                            iconType: "\uf197"
                            xPosition: 20
                        }
                    }

                    Rectangle {
                        color: UIConstants.transparentColor
                        Layout.alignment: Qt.AlignBottom
                        Layout.preferredWidth: parent.width
                        Layout.preferredHeight: 70 * 9
                        Layout.fillHeight: true
                        Payload{
                            anchors.fill: parent
                        }
                    }
                }
            }

            Item {
                ColumnLayout {
                    anchors.fill: parent
                    //--- Title
                    Rectangle {
                        color: UIConstants.transparentColor
                        Layout.alignment: Qt.AlignTop
                        Layout.preferredWidth: parent.width
                        Layout.preferredHeight: 70
                        SidebarTitle {
                            anchors.fill: parent
                            visible: true
                            title: "Preflight check process"
                            iconType: "\uf197"
                            xPosition: 20
                        }
                    }

                    Rectangle {
                        color: UIConstants.transparentColor
                        Layout.alignment: Qt.AlignBottom
                        Layout.preferredWidth: parent.width
                        Layout.preferredHeight: 70 * 9
                        Layout.fillHeight: true
                        Success{
                            anchors.fill: parent
                        }
                    }
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
