/**
 * ==============================================================================
 * @Project: FCS-Groundcontrol-based
 * @Module: Main FooterBar Custom
 * @Breif:
 * @Author: Trung Nguyen
 * @Date: 26/03/2019
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
import CustomViews.Dialogs 1.0

import io.qdt.dev 1.0
Item {
    id: rootItem
    width: 1366
    height: 40
    //-------------------- Custom Properties
    property int waypointIDSetProperty: -1
    property bool addWPSelected: btnAddWP.isPressed
    property alias loiterSelected: btnLoiter.isOn
    property alias footerBarCurrent: footerBarStack.currentIndex
    property alias topFooterBarVisible: topFooterBar.visible
    property var footerBarCorresFlyingBtnsEnable: [true,false]
    property string flightView: "MAP" // "WP"
    property bool   _armed: vehicle.armed

    //-------------------- Signals
    signal preflightCheckNext()
    signal preflightCheckPrev()
    signal doPreflightItemCheck()
    signal doFlyAction(real actionIndex) //1: Auto, 2: Guided, 3: takeoff, 4: altitudeChange, 5: speedChange, 6: loiter radius
    signal doFlyExecuteAction(real actionIndex)
    signal doLoadMap()
    signal doNewMission()
    signal doLoadMission()
    signal doSaveMission()
    signal doEnableAddingWP(var en)
    signal doAddingSurvey()
    signal doAddingCorridorScan()
    signal doAddingFixedWingLanding()
    signal doDownloadPlan()
    signal doUploadPlan()
    signal doCircle(var closeWise)
    signal doWaypoint()
    signal doManual()
    signal doSwitchPlaneMode(var _currentMode, var _interestedMode)
    signal doDeleteWP(var _index)
    signal doGoWP()
    signal doGoPosition()
    signal doArm(var arm)
    signal addWP()
    signal deleteWP()
    signal doAddMarker()
    signal doDeleteMarker()
    signal doNextWP()


    function getFlightAltitudeTarget(){
        return Math.round(btnFlightAltitude.topValue);
    }

    function getFlightSpeedTarget(value){
        return Math.round(btnFlightSpeed.topValue);
    }

    function setFlightAltitudeTarget(waypointID,value){
        waypointIDSetProperty = waypointID;
        btnFlightAltitude.topValue = Math.round(value);
    }

    function setFlightSpeedTarget(value){
        btnFlightSpeed.topValue = Math.round(value);
    }

    //-------------------- Childrems
    Rectangle {
        id: topFooterBar
        anchors.fill: parent
        color: UIConstants.bgColorOverlay
        StackLayout {
            id: footerBarStack
            anchors.fill: parent
            currentIndex: 2
            //--- Index = 0
            Rectangle {
                id: corresMissionPlanner
                color: UIConstants.transparentColor
                Layout.alignment: Qt.AlignCenter
                RowLayout {
                    id: group1
                    x: 0
                    spacing: 5
                    height: parent.height
                    width: parent.height
                    FooterButton {
                        id: btnLoadMap
                        Layout.preferredWidth: parent.height
                        Layout.preferredHeight: parent.height
                        icon: UIConstants.iOpenFolder
                        btnText: "Load\nMap"
                        onClicked: {
                            rootItem.doLoadMap();
                        }
                    }
                    FooterButton {
                        id: btnLoadMission
                        Layout.preferredWidth: parent.height
                        Layout.preferredHeight: parent.height
                        icon: UIConstants.iOpenFolder
                        btnText: "Load\nMission"
                        onClicked: {
                            rootItem.doLoadMission();
                        }
                    }
                    FooterButton {
                        id: btnSaveMission
                        Layout.preferredWidth: parent.height
                        Layout.preferredHeight: parent.height
                        icon: UIConstants.iSaveFolder
                        btnText: "Save\nMission"
                        onClicked: {
                            rootItem.doSaveMission();
                        }
                    }

                    FooterButton {
                        id: btnAddNewMission
                        Layout.preferredWidth: parent.height
                        Layout.preferredHeight: parent.height
                        btnText: "New\nMission"
                        icon: UIConstants.iNewFolder
                        onClicked: {
                            rootItem.doNewMission();
                        }
                    }
                }

                RowLayout {
                    id: group2
                    spacing: 5
                    height: parent.height
                    anchors.horizontalCenter: parent.horizontalCenter
                    width: (spacing*4+height*5)
                    FooterButton {
                        id: btnNextWP
                        Layout.preferredWidth: parent.height
                        Layout.preferredHeight: parent.height
                        btnText: "Next\nWP"
                        icon: UIConstants.iNextWP
                        onClicked: {
                            rootItem.doNextWP();
                        }
                    }

                    FooterButton {
                        id: btnAddWP
                        Layout.preferredWidth: parent.height
                        Layout.preferredHeight: parent.height
                        btnText: "Add WP"
                        isEnable: mapPane.mousePressed && UIConstants.mouseOnMapMode === UIConstants.mouseOnMapModeWaypoint
                        icon: UIConstants.iAddWaypoint
                        isAutoReturn: false
                        onClicked: {
                            if(btnAddWP02.isPressed != btnAddWP.isPressed){
                                btnAddWP02.isPressed = btnAddWP.isPressed
                            }
                        }
                    }

                    FooterButton {
                        id: btnDeleteWP
                        Layout.preferredWidth: parent.height
                        Layout.preferredHeight: parent.height
                        btnText: "Delete\nWP"
                        isEnable: mapPane.selectedWP !== undefined
                                  && (mapPane.selectedWP.attributes.attributeValue("id") > 0)
                                  && (UIConstants.mouseOnMapMode === UIConstants.mouseOnMapModeWaypoint)
                        isSolid: false
                        icon: UIConstants.iDeleteWP
                        onClicked: {
                            rootItem.deleteWP();
                        }
                    }

                    FooterButton {
                        id: btnAddMarker
                        Layout.preferredWidth: parent.height
                        Layout.preferredHeight: parent.height
                        btnText: "Add\nMarker"
                        isEnable: mapPane.mousePressed && UIConstants.mouseOnMapMode === UIConstants.mouseOnMapModeWaypoint
                        icon: UIConstants.iAddMarker
                        onClicked: {
                            rootItem.doAddMarker();
                        }
                    }

                    FooterButton {
                        id: btnDeleteMarker
                        Layout.preferredWidth: parent.height
                        Layout.preferredHeight: parent.height
                        btnText: "Delete\nMarker"
                        isEnable: mapPane.selectedMarker !== undefined && UIConstants.mouseOnMapMode === UIConstants.mouseOnMapModeWaypoint
                        icon: UIConstants.iRemoveMarker
                        onClicked: {
                            rootItem.doDeleteMarker();
                        }
                    }
                }

                RowLayout {
                    id: group3
                    anchors{ right: parent.right }
                    layoutDirection: Qt.RightToLeft
                    height: parent.height
                    width: parent.height
                    spacing: 5
                    FooterButton {
                        id: btnUploadPlan
                        Layout.preferredWidth: parent.height
                        Layout.preferredHeight: parent.height
                        icon: UIConstants.iUpload
                        btnText: qsTr("Upload\nPlan")
                        isTimerActive: true
                        timeout: 1000
                        isEnable: vehicle.link
                        onClicked: {
                            rootItem.doUploadPlan()
                        }
                    }
                    FooterButton {
                        id: btnDownloadPlan
                        Layout.preferredWidth: parent.height
                        Layout.preferredHeight: parent.height
                        icon: UIConstants.iDownload
                        btnText: "Down\nPlan"
                        isTimerActive: true
                        timeout: 1000
                        isEnable: vehicle.link
                        onClicked: {
                                rootItem.doDownloadPlan()
                                isEnable = false;
                                timerDownloadPlan.start();
                        }
                        Timer{
                            id: timerDownloadPlan
                            interval: 1000
                            running: false
                            repeat: false
                            onTriggered: {
                                parent.isEnable = true;
                            }
                        }
                    }
                    FooterButton {
                        id: btnArm
                        Layout.preferredWidth: parent.height
                        Layout.preferredHeight: parent.height
                        icon: !rootItem._armed?UIConstants.iArmed:UIConstants.iDisarmed
                        btnText: !rootItem._armed?"Arm":"Disarm"
                        isEnable: vehicle.link
                        isTimerActive: true
                        timeout: 1000
                        onClicked: {
                            rootItem.doArm(!rootItem._armed);
                        }
                    }
                }
            }

            //--- Index = 1
            Rectangle {
                id: corresPreflightCheck
                color: UIConstants.transparentColor
                Layout.alignment: Qt.AlignCenter
                RowLayout {
                    id: preflightCheckGroup1
                    x: 0
                    spacing: 5
                    height: parent.height
                    width: parent.height
                    SwitchFlatButton {
                        id: btnTrackerConnection
                        Layout.preferredWidth: parent.height
                        Layout.preferredHeight: parent.height
                        icon: UIConstants.iTracker
                        btnText: "Tracker"
                        isOn: true
                    }
                    FooterButton {
                        id: btnTurnLeft
                        Layout.preferredWidth: parent.height
                        Layout.preferredHeight: parent.height
                        iconRotate: 180
                        icon: UIConstants.iTurnRight
                        btnText: "Turn\nLeft"
                    }

                    FooterButton {
                        id: btnTurnRight
                        Layout.preferredWidth: parent.height
                        Layout.preferredHeight: parent.height
                        icon: UIConstants.iTurnRight
                        btnText: "Turn\nRight"
                    }
                }

                RowLayout {
                    id: preflightCheckGroup2
                    spacing: 5
                    height: parent.height
                    anchors.horizontalCenter: parent.horizontalCenter
                    width: (spacing*0+height*1)

                    FooterButton {
                        id: btnCheck
                        Layout.preferredWidth: parent.height
                        Layout.preferredHeight: parent.height
                        icon: UIConstants.iChecked
                        btnText: "Check"
                        onClicked: {
                            rootItem.doPreflightItemCheck();
                        }
                    }
                }

                RowLayout {
                    id: preflightCheckGroup3
                    anchors{ right: parent.right;}
                    layoutDirection: Qt.RightToLeft
                    height: parent.height
                    width: parent.height
                    anchors.verticalCenter: parent.verticalCenter
                    spacing: 5
                    FooterButton {
                        id: btnNext
                        Layout.preferredWidth: parent.height
                        Layout.preferredHeight: parent.height
                        icon: UIConstants.iStop
                        btnText: "Next"
                        onClicked: {
                            rootItem.preflightCheckNext();
                        }
                    }
                    FooterButton {
                        id: btnPrev
                        Layout.preferredWidth: parent.height
                        Layout.preferredHeight: parent.height
                        icon: UIConstants.iCaretLeft
                        btnText: "Previous"
                        onClicked: {
                            rootItem.preflightCheckPrev();
                        }
                    }
                }
            }

            //--- Index = 2
            Rectangle {
                id: corresFlyView
                color: UIConstants.transparentColor
                Layout.alignment: Qt.AlignCenter
                RowLayout {
                    spacing: 5
                    height: parent.height
                    width: parent.height
                    FooterSpecialButton {
                        id: btnFlightAltitude
                        Layout.preferredWidth: parent.height
                        Layout.preferredHeight: parent.height
                        Layout.alignment: Qt.AlignTop
                        topValue: "100"
                        middleValue: (vehicle?Number(vehicle.altitudeRelative).toFixed(0).toString():"0")
                        bottomValue: "m"
                        isEnable: vehicle.link
                                  &&
                                  ((vehicle.vehicleType === 2 && vehicle.flightMode === "Guided")
                                   ||
                                   (vehicle.vehicleType === 1))
                        Connections{
                            target: missionController
                            onCurrentIndexChanged:{
                                if((vehicle.flightMode === "Auto" || vehicle.flightMode === "RTL" )&&
                                        planController.missionItems.length > sequence
                                        ){
                                    if(sequence !== rootItem.waypointIDSetProperty){
                                        if(sequence > 0){
                                            rootItem.setFlightAltitudeTarget(
                                                sequence,
                                                planController.missionItems[sequence].param7);
                                        }else{
                                            rootItem.setFlightAltitudeTarget(
                                                sequence,
                                                vehicle.homePosition.altitude);
                                        }
                                    }else {
                                        if(sequence === 0){
                                            rootItem.setFlightAltitudeTarget(
                                                sequence,
                                                vehicle.homePosition.altitude);
                                        }
                                    }
                                }
                            }
                        }
                        onClicked: {
                            rootItem.doFlyAction(4);
                        }
                    }

                    FooterSpecialButton {
                        id: btnFlightSpeed
                        Layout.preferredWidth: parent.height
                        Layout.preferredHeight: parent.height
                        Layout.alignment: Qt.AlignTop
                        topValue: Math.round(vehicle.paramAirSpeed*3.6)
                        middleValue: (vehicle?Math.round(
                                                   (vehicle.vehicleType === 2?vehicle.groundSpeed:vehicle.airSpeed)
                                                   *3.6).toString():"0")
                        bottomValue: "km/h"
                        isEnable: vehicle.link
                                  &&
                                  ((vehicle.vehicleType === 2 && vehicle.flightMode === "Guided")
                                   ||
                                   (vehicle.vehicleType === 1))
                        onClicked: {
                            rootItem.doFlyAction(5);
                        }

                    }
                    Item{
                        Layout.preferredWidth: parent.height*1.5
                        Layout.preferredHeight: parent.height
                        Label{
                            id: lblGroundSpeed
                            anchors.top: parent.top
                            anchors.left: parent.left
                            color: UIConstants.textColor
                            text: "Ground Speed:\n"+(vehicle?Number(vehicle.groundSpeed*3.6).toFixed(2).toString():"--")+" km/h"
                            font.pixelSize: UIConstants.fontSize - 2
                            font.family: UIConstants.appFont
                        }
                        Label{
                            id: lblAMSL
                            anchors.bottom: parent.bottom
                            anchors.left: parent.left
                            anchors.bottomMargin: 5
                            color: UIConstants.textColor
                            text: "AMSL:\n"+(vehicle?Math.round(vehicle.altitudeAMSL).toString():"--")+" m"
                            font.pixelSize: UIConstants.fontSize - 2
                            font.family: UIConstants.appFont
                        }
                    }
                    Item{
                        Layout.preferredWidth: parent.height*1.5
                        Layout.preferredHeight: parent.height
                        Label{
                            id: lblClimbSpeed
                            anchors.top: parent.top
                            anchors.left: parent.left
                            color: UIConstants.textColor
                            text: "Climb Speed:\n"+(vehicle?Number(vehicle.climbSpeed*3.6).toFixed(2).toString():"--")+" km/h"
                            font.pixelSize: UIConstants.fontSize - 2
                            font.family: UIConstants.appFont
                        }
                        Label{
                            id: lblAGL
                            anchors.bottom: parent.bottom
                            anchors.left: parent.left
                            anchors.bottomMargin: 5
                            color: UIConstants.textColor
                            text: "AGL:\n"+(vehicle?Number(vehicle.altitudeRelative).toFixed(2).toString():"--")+" m"
                            font.pixelSize: UIConstants.fontSize - 2
                            font.family: UIConstants.appFont
                        }
                    }
                    FooterButton {
                        id: btnFlightToWP
                        Layout.preferredWidth: parent.height
                        Layout.preferredHeight: parent.height
                        Layout.alignment: Qt.AlignTop
                        Layout.leftMargin: -20
                        icon: UIConstants.iRtl
                        btnText: mapPane.selectedWP !== undefined ? "Go to\nWP":
                                                                    ("Go to\nLocation")
                        isEnable: vehicle.link
                                  &&
                                  ((vehicle.vehicleType === 2)
                                   ||
                                   (vehicle.vehicleType === 14)
                                   ||
                                   (vehicle.vehicleType === 1 && mapPane.selectedWP !== undefined ) )
                        onClicked: {
                            if(mapPane.selectedWP !== undefined){
                                rootItem.doGoWP();
                            }else{
                                if(vehicle.vehicleType === 2 || vehicle.vehicleType === 14)
                                rootItem.doGoPosition();
                            }
                        }
                    }

                    FooterButton {
                        id: btnNextWP02
                        Layout.preferredWidth: parent.height
                        Layout.preferredHeight: parent.height
                        Layout.alignment: Qt.AlignTop
                        icon: UIConstants.iNextWP
                        btnText: "Next WP"
                        onClicked: {
                            rootItem.doNextWP();
                        }
                    }

                    FooterButton {
                        id: btnDeleteWP02
                        Layout.preferredWidth: parent.height
                        Layout.preferredHeight: parent.height
                        Layout.alignment: Qt.AlignTop
                        icon: UIConstants.iDeleteWP
                        btnText: "Delete\nWP"
                        isEnable: mapPane.selectedWP !== undefined
                                  && (mapPane.selectedWP.attributes.attributeValue("id") > 0)
                                  && (UIConstants.mouseOnMapMode === UIConstants.mouseOnMapModeWaypoint)
                        onClicked: {
                            rootItem.deleteWP();
                        }
                    }
                    FooterButton {
                        id: btnAddWP02
                        Layout.preferredWidth: parent.height
                        Layout.preferredHeight: parent.height
                        Layout.alignment: Qt.AlignTop
                        icon: UIConstants.iAddWaypoint
                        btnText: "Add WP"
                        isAutoReturn: false
                        isEnable: mapPane.mousePressed && UIConstants.mouseOnMapMode === UIConstants.mouseOnMapModeWaypoint
                        onClicked: {
                            if(btnAddWP.isPressed != btnAddWP02.isPressed){
                                btnAddWP.isPressed = btnAddWP02.isPressed
                            }
                        }
                    }
                    FooterButton {
                        id: btnFlightAuto
                        Layout.preferredWidth: parent.height
                        Layout.preferredHeight: parent.height
                        Layout.alignment: Qt.AlignTop
                        icon: UIConstants.iAuto
                        btnText: "Auto"
                        isEnable: vehicle.link
                        onClicked: {
                            rootItem.doFlyAction(1);
                        }
                    }
                    FooterButton {
                        id: btnFlightGuided
                        Layout.preferredWidth: parent.height
                        Layout.preferredHeight: parent.height
                        Layout.alignment: Qt.AlignTop
                        icon: UIConstants.iManual
                        btnText: "Guided"
                        isEnable: vehicle.link
                        onClicked: {
                            rootItem.doFlyAction(2);
                        }
                    }
                }

                RowLayout {
                    spacing: 5
                    height: parent.height
                    width: parent.height
                    layoutDirection: Qt.RightToLeft
                    anchors { right: parent.right; rightMargin: 0 }
                    FooterButton {
                        id: btnTakeOff
                        Layout.preferredWidth: parent.height
                        Layout.preferredHeight: parent.height
                        icon: UIConstants.iDeparture
                        btnText: "Take Off"
                        isEnable: vehicle.link
                        onClicked: {
                            rootItem.doFlyAction(3);
                        }
                    }
                    FooterButton {
                        id: btnArm02
                        Layout.preferredWidth: parent.height
                        Layout.preferredHeight: parent.height
                        icon: !rootItem._armed?UIConstants.iArmed:UIConstants.iDisarmed
                        btnText: !rootItem._armed?"Arm":"Disarm"
                        isEnable: vehicle.link
//                        isTimerActive: true
//                        timeout: 1000
                        onClicked: {
                            rootItem.doArm(!rootItem._armed);
                        }
                    }
                    StackLayout{
                        id: stkFlightView
                        Layout.preferredWidth: parent.height*6
                        Layout.preferredHeight: parent.height
                        currentIndex: rootItem.flightView == "MAP"?0:1
                        RowLayout {
                            spacing: 5
                            height: parent.height
                            layoutDirection: Qt.RightToLeft
                            width: (spacing*5+height*6)
                            FooterButton {
                                id: btnUploadPlan02
                                Layout.preferredWidth: parent.height
                                Layout.preferredHeight: parent.height
                                icon: UIConstants.iUpload
                                btnText: qsTr("Upload\nPlan")
                                isTimerActive: true
                                timeout: 1000
                                isEnable: vehicle.link
                                onClicked: {
                                    rootItem.doUploadPlan();
                                }
                            }
                            FooterButton {
                                id: btnDownloadPlan02
                                Layout.preferredWidth: parent.height
                                Layout.preferredHeight: parent.height
                                icon: UIConstants.iDownload
                                btnText: "Down\nPlan"
                                isTimerActive: true
                                timeout: 1000
                                isEnable: vehicle.link
                                onClicked: {
                                        rootItem.doDownloadPlan();
                                }
                            }
                            FooterButton {
                                id: btnDeleteMarker02
                                Layout.preferredWidth: parent.height
                                Layout.preferredHeight: parent.height
                                btnText: "Delete\nMarker"
                                icon: UIConstants.iRemoveMarker
                                isEnable: mapPane.selectedMarker !== undefined && UIConstants.mouseOnMapMode === UIConstants.mouseOnMapModeWaypoint
                                onClicked: {
                                    rootItem.doDeleteMarker();
                                }
                            }
                            FooterButton {
                                id: btnAddMarker02
                                Layout.preferredWidth: parent.height
                                Layout.preferredHeight: parent.height
                                btnText: "Add\nMarker"
                                isEnable: mapPane.mousePressed && UIConstants.mouseOnMapMode === UIConstants.mouseOnMapModeWaypoint
                                icon: UIConstants.iAddMarker
                                onClicked: {
                                    rootItem.doAddMarker();
                                }
                            }
                        }
                        RowLayout {
                            spacing: 5
                            height: parent.height
                            layoutDirection: Qt.RightToLeft
                            width: (spacing*5+height*6)
                            FooterButton {
                                id: btnLoiterRadius
                                Layout.preferredWidth: parent.height
                                Layout.preferredHeight: parent.height
                                icon: Math.round(vehicle.paramLoiterRadius)
                                isEnable: btnLoiter.isOn
                                isSolid: true
                                btnText: "Loiter\nRadius"
                                onClicked: {
                                    rootItem.doFlyAction(6);
                                }
                            }
                            FooterButton {
                                id: btnLoiterDir
                                property bool isClockWise: true
                                Layout.preferredWidth: parent.height
                                Layout.preferredHeight: parent.height
                                icon: !isClockWise?UIConstants.iCircleClockWise:UIConstants.iCircleCounterClock
                                btnText: "Loiter\nDir"
                                isEnable: btnLoiter.isOn
                                onClicked: {
                                    isClockWise = !isClockWise;
                                    doCircle(isClockWise);
                                }
                            }
                            SwitchFlatButton {
                                id: btnLoiter
                                Layout.preferredWidth: parent.height
                                Layout.preferredHeight: parent.height
                                isSolid: false
                                icon: UIConstants.iCircle
                                btnText: "Loiter"
                                isOn: false
                                onClicked: {
                                    if(isOn === false){
                                        btnLoiterDir.isClockWise = true;
                                        doCircle(true);
                                    }else{
                                        doWaypoint();
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        gradient: Gradient {
            GradientStop { position: 0.0; color: UIConstants.bgColorOverlay }
            GradientStop { position: 0.8; color: UIConstants.cfProcessingOverlayBg }
            GradientStop { position: 1.0; color: UIConstants.cfProcessingOverlayBg }
        }
    }
    Rectangle{
        id: rectDirtyUpload02
        anchors.bottom: topFooterBar.top
        anchors.bottomMargin: UIConstants.rectRadius
        width: UIConstants.sRect*20
        height: UIConstants.sRect*2
        border.width: 1
        border.color: UIConstants.grayColor
        radius: UIConstants.rectRadius
        anchors.horizontalCenter: parent.horizontalCenter
        color: "transparent"
        visible: false
        Label {
            id: label
            text: qsTr("Plan is not synchronized")
            color: UIConstants.textColor
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignHCenter
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.verticalCenter: parent.verticalCenter
        }

    }

    SequentialAnimation{
        id: animDirtyUpload02
        loops: Animation.Infinite
        ColorAnimation{
            target: rectDirtyUpload02
            properties: "color"
            to: UIConstants.transparentBlueDarker
            duration: 1000
            easing.type: Easing.Linear
        }
        ColorAnimation{
            target: rectDirtyUpload02
            properties: "color"
            to: UIConstants.transparentColor
            duration: 1000
            easing.type: Easing.Linear
        }
    }
    Connections{
        target: mapPane
        onIsMapSyncChanged:{
            if(!mapPane.isMapSync){
                animDirtyUpload02.start();
                rectDirtyUpload02.visible = true;
            }else{
                animDirtyUpload02.stop();
                rectDirtyUpload02.visible = false;
            }
        }
    }
}
