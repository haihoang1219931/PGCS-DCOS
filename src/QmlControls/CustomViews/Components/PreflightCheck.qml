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
//import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0

//------------------ Item definition
Item {
    id: rootItem
    RowLayout {
        anchors.fill: parent
        //---------- Sidebar
        SidebarConfigs {
            id: sidebarGeneralConfigs
            Layout.preferredWidth: parent.width / 5
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
            Layout.preferredWidth: parent.width * 4 / 5
            Layout.preferredHeight: parent.height
            anchors.left: sidebarGeneralConfigs.right
            currentIndex: 0
            Rectangle {
                anchors.fill: parent
                color: UIConstants.cfProcessingOverlayBg
                RowLayout {
                    anchors.fill: parent
                    //--- Title
                    SidebarTitle {
                        id: title
                        anchors { top: parent.top; left: parent.left; right: parent.right }
                        height: parent.height / 10
                        visible: true
                        title: "Tracker information verification"
                        iconType: "\uf197"
                        xPosition: 20
                    }

                    Text {
                        text: "Tracker"
                    }
                }
            }
            Rectangle {
                anchors.fill: parent
                color: UIConstants.cfProcessingOverlayBg
                RowLayout {
                    anchors.fill: parent
                    //--- Title
                    SidebarTitle {
                        id: title1
                        anchors { top: parent.top; left: parent.left; right: parent.right }
                        height: parent.height / 10
                        visible: true
                        title: "SmonCheck information verification"
                        iconType: "\uf197"
                        xPosition: 20
                    }

                    Text {
                        text: "SmonCheck"
                    }
                }
            }
            Rectangle {
                anchors.fill: parent
                color: UIConstants.cfProcessingOverlayBg
                RowLayout {
                    anchors.fill: parent
                    //--- Title
                    SidebarTitle {
                        id: title2
                        anchors { top: parent.top; left: parent.left; right: parent.right }
                        height: parent.height / 10
                        visible: true
                        title: "GPS information verification"
                        iconType: "\uf197"
                        xPosition: 20
                    }

                    Text {
                        text: "GPS"
                    }
                }
            }
            Rectangle {
                anchors.fill: parent
                color: UIConstants.cfProcessingOverlayBg
                RowLayout {
                    anchors.fill: parent
                    //--- Title
                    SidebarTitle {
                        id: title3
                        anchors { top: parent.top; left: parent.left; right: parent.right }
                        height: parent.height / 10
                        visible: true
                        title: "IncDecSync information verification"
                        iconType: "\uf197"
                        xPosition: 20
                    }

                    Text {
                        text: "IncDecSync"
                    }
                }
            }
            Rectangle {
                anchors.fill: parent
                color: UIConstants.cfProcessingOverlayBg
                RowLayout {
                    anchors.fill: parent
                    //--- Title
                    SidebarTitle {
                        id: title4
                        anchors { top: parent.top; left: parent.left; right: parent.right }
                        height: parent.height / 10
                        visible: true
                        title: "Heading information verification"
                        iconType: "\uf197"
                        xPosition: 20
                    }

                    Text {
                        text: "Heading"
                    }
                }
            }
            Rectangle {
                anchors.fill: parent
                color: UIConstants.cfProcessingOverlayBg
                RowLayout {
                    anchors.fill: parent
                    //--- Title
                    SidebarTitle {
                        id: title5
                        anchors { top: parent.top; left: parent.left; right: parent.right }
                        height: parent.height / 10
                        visible: true
                        title: "Stearing information verification"
                        iconType: "\uf197"
                        xPosition: 20
                    }

                    Text {
                        text: "Stearing"
                    }
                }
            }
            Rectangle {
                anchors.fill: parent
                color: UIConstants.cfProcessingOverlayBg
                RowLayout {
                    anchors.fill: parent
                    //--- Title
                    SidebarTitle {
                        id: title6
                        anchors { top: parent.top; left: parent.left; right: parent.right }
                        height: parent.height / 10
                        visible: true
                        title: "Pilot information verification"
                        iconType: "\uf197"
                        xPosition: 20
                    }

                    Text {
                        text: "Pilot"
                    }
                }
            }
            Rectangle {
                anchors.fill: parent
                color: UIConstants.cfProcessingOverlayBg
                RowLayout {
                    anchors.fill: parent
                    //--- Title
                    SidebarTitle {
                        id: title7
                        anchors { top: parent.top; left: parent.left; right: parent.right }
                        height: parent.height / 10
                        visible: true
                        title: "AHRS information verification"
                        iconType: "\uf197"
                        xPosition: 20
                    }

                    Text {
                        text: "AHRS"
                    }
                }
            }
            Rectangle {
                anchors.fill: parent
                color: UIConstants.cfProcessingOverlayBg
                RowLayout {
                    anchors.fill: parent
                    //--- Title
                    SidebarTitle {
                        id: title8
                        anchors { top: parent.top; left: parent.left; right: parent.right }
                        height: parent.height / 10
                        visible: true
                        title: "Payload information verification"
                        iconType: "\uf197"
                        xPosition: 20
                    }

                    Text {
                        text: "Payload"
                    }
                }
            }
        }
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
        processingOverlay.opacity = 1;
        processingOverlay.textOverlay = "Checking. Please wait !"
        sidebarGeneralConfigs.doCheck();
    }
}
