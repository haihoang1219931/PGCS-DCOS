/**
 * ==============================================================================
 * @Project: FCS-Groundcontrol-based
 * @Module: Main Navbar Custom
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
import CustomViews.ToolBar 1.0
import CustomViews.Dialogs 1.0

import "qrc:/Maplib/transform.js" as Conv
//---------------- Component definition ---------------------------------------
Item {
    id: rootItem
    width: 1366
    height: 40
    //----- Properties
    property string   dialogShow: ""
    property bool   isShowDialog: false
    property bool   _communicationLost:   true
    property int    _batteryPercent:       20
    property bool   _isMessageImportant:    false
    property int    _flightTime: 0 // second
    property bool   _oldLandedFlag: true
    property int    _second: 0
    property int    _minute: 0
    property int    _hour: 0
    property bool   _showParams: true
    property int   _lastSeq: -1
    property int fontSize: 30
    property alias batteryInfoVisible: btnBattery.visible
    property alias engineInfoVisible: btnEngine.visible
    //----- Signals
    signal itemNavChoosed(real seq) 
    signal requestDisplayMessages()
    signal requestDisplayGPS()
    signal doSwitchPlaneMode(var previousMode, var currentMode)
    signal toggleWarning()
    signal doShowParamsSelect(var show)
    signal doShowJoystickConfirm(var show)
    property alias  _isPayloadActive:  btnPayloadControl.isPressed
    //----- Element attributes

    //----- Child elements

    //----- List tab button
    property var listTab: [tabMP,tabPC,tabFlight]
    property var itemListName: UIConstants.itemTextMultilanguages["NAVBAR"]

    function startFlightTimer(){
        timerFlightTime.start();
    }

    function changeTabFocus(id){
        console.log("Focus on NAV["+id+"]");
        if(btnSystemConfig.idxBefore<0){
            btnSystemConfig.idxBefore = id;
        }

        for(var tabID = 0; tabID < listTab.length; tabID ++){
            if(listTab[tabID].idx !== id) {
                listTab[tabID].active = false;
                listTab[tabID].setInactive();
            }else{
                listTab[tabID].active = true;
                listTab[tabID].setActive();
            }
        }
        if(btnSystemConfig.idx !== id){
            btnSystemConfig.isActive = false;
            btnSystemConfig.setButtonNormal();
            btnSystemConfig.idxBefore = id;
        }
    }
    function setFlightMode(flightmode){
        lstFlightMode.setCurrentText(flightmode);
    }

    function setFlightModes(flightmodes){
        if(lstFlightMode.model.length !== flightmodes.length){
            lstFlightMode.model = flightmodes;
            lstFlightMode.prevItem = "";
        }
        if(lstFlightMode.prevItem !== FlightVehicle.flightMode && !lstFlightMode.visible){
            setFlightMode(FlightVehicle.flightMode);
        }
    }
    Connections{
        target: vehicle
        onLandedChanged:{
            if(FlightVehicle.landed){
                setFlightModes(FlightVehicle.flightModesOnGround);
            }else{
                setFlightModes(FlightVehicle.flightModesOnAir);
            }
        }
        onFlightModesChanged:{
            console.log("List flight mode update");
            if(FlightVehicle.landed){
                setFlightModes(FlightVehicle.flightModesOnGround);
            }else{
                setFlightModes(FlightVehicle.flightModesOnAir);
            }
        }
        onFlightModeChanged:{
            console.log("flightmode change to "+flightMode);
            setFlightMode(flightMode);
        }
        onMissingParametersChanged:{
            console.log("Load param "+(!missingParameters));
            toastFlightControler.callActionAppearance = false;
            toastFlightControler.rejectButtonAppearance = false;
            toastFlightControler.toastContent = "Load full param return "+(!missingParameters);
            toastFlightControler.show();
            if(!missingParameters){
                rectParamLoading.width = 0;
            }
        }
        onLoadProgressChanged:{
            rectParamLoading.width = rootItem.width*value;
        }
    }

    Rectangle {
        id: navbarWrapper
        anchors.fill: parent
        color: UIConstants.bgColorOverlay
        //----------- Nav menu
        RowLayout {
            id: leftBtnsRow
            anchors { left: parent.left; top: parent.top; bottom: parent.bottom; leftMargin: 15 }
            height: parent.height
            spacing: 0
            //---------- Menu navigation
            FlatButton {
                id: tabMP
                btnText: itemListName["MISSION"][UIConstants.language[UIConstants.languageID]]
                btnTextColor: UIConstants.textFooterColor
                Layout.preferredHeight: parent.height
                Layout.preferredWidth: width
                z: navbarWrapper.z + 1
                iconVisible: true
                icon: UIConstants.iMission
                color: active ? UIConstants.sidebarActiveBg : UIConstants.transparentColor
                radius: 5
                active: false
                idx: 0
                onClicked: {
                    rootItem._showParams = true;
                    rootItem.itemNavChoosed(idx);
                    rootItem.changeTabFocus(idx);
                }
            }
            FlatButton {
                id: tabPC
                btnText: itemListName["PRECHECK"][UIConstants.language[UIConstants.languageID]]
                btnTextColor: UIConstants.textFooterColor
                Layout.preferredHeight: parent.height
                Layout.preferredWidth: width
                z: navbarWrapper.z + 1
                iconVisible: true
                icon: UIConstants.iPrecheck
                color: active ? UIConstants.sidebarActiveBg : UIConstants.transparentColor
                radius: 5
                active: false
                idx: 1
                onClicked: {
                    rootItem._showParams = false;
                    rootItem.itemNavChoosed(idx);
                    rootItem.changeTabFocus(idx);
                }
            }
            FlatButton {
                id: tabFlight
                btnText: itemListName["FLIGHT"][UIConstants.language[UIConstants.languageID]]
                btnTextColor: UIConstants.textFooterColor
                Layout.preferredHeight: parent.height
                Layout.preferredWidth: width
                z: navbarWrapper.z + 1
                iconVisible: true
                icon: UIConstants.iFlight
                color: active ? UIConstants.sidebarActiveBg : UIConstants.transparentColor
                radius: 5
                active: false
                idx: 2
                onClicked: {
                    rootItem._showParams = true;
                    rootItem.itemNavChoosed(idx);
                    rootItem.changeTabFocus(idx);
                }
            }
        }

        //----------- Division
        SignalIndicator{
            id: btnLinkStatus
            anchors.right: btnMessage.left
            anchors.rightMargin: UIConstants.sRect * 3 / 2
            anchors.top: parent.top
            anchors.topMargin: 5
            height: parent.height
            width: parent.height
            iconSize: UIConstants.sRect*3/2
        }

        MessageIndicator{
            id: btnMessage
            anchors.right: btnSignal.left
            anchors.rightMargin: UIConstants.sRect
            anchors.top: parent.top
            anchors.topMargin: 5
            height: parent.height
            width: parent.height
            iconSize: UIConstants.sRect*3/2
            z: navbarWrapper.z + 1
            showIndicator: dialogShow === "DIALOG_MESSAGE"
            onClicked: {
                if(dialogShow !== "DIALOG_MESSAGE"){
                    dialogShow = "DIALOG_MESSAGE";
                }else{
                    dialogShow = "";
                }
            }
        }
        GPSIndicator{
            id: btnSignal
            anchors.right: btnEngine.left
            anchors.rightMargin: UIConstants.sRect * 3 / 2
            anchors.top: parent.top
            anchors.topMargin: 5
            height: parent.height
            width: parent.height
            iconSize: UIConstants.sRect*3/2
            z: navbarWrapper.z + 1
            showIndicator: dialogShow === "DIALOG_SIGNAL"
            onClicked: {
                if(dialogShow !== "DIALOG_SIGNAL"){
                    dialogShow = "DIALOG_SIGNAL";
                }else{
                    dialogShow = "";
                }
            }
        }

        EngineIndicator{
            id: btnEngine
            anchors.right: btnBattery.left
            anchors.rightMargin: visible?UIConstants.sRect * 3 / 2:0
            anchors.top: parent.top
            anchors.topMargin: 5
            height: parent.height
            width: visible?parent.height:0
            iconSize: UIConstants.sRect*3/2
            z: navbarWrapper.z + 1
            visible: FlightVehicle.vehicleType === 1
            showIndicator: dialogShow === "DIALOG_ENGINE"
            onClicked: {
                if(dialogShow !== "DIALOG_ENGINE"){
                    dialogShow = "DIALOG_ENGINE";
                }else{
                    dialogShow = "";
                }
            }
        }
        BatteryIndicator
        {
            id: btnBattery
            anchors.right: btnJoystick.left
            anchors.rightMargin: visible?UIConstants.sRect * 3 / 2:0
            anchors.top: parent.top
            anchors.topMargin: 5
            height: parent.height
            width: visible?parent.height:0
            iconSize: UIConstants.sRect*3/2
            z: navbarWrapper.z + 1
            visible: FlightVehicle.vehicleType === 1
            showIndicator: dialogShow === "DIALOG_BATTERY"
            onClicked: {
                if(dialogShow !== "DIALOG_BATTERY"){
                    dialogShow = "DIALOG_BATTERY";
                }else{
                    dialogShow = "";
                }
            }
        }
        FlatButtonIcon{
            id: btnJoystick
            anchors.right: btnParams.left
            anchors.rightMargin: 2
            anchors.top: parent.top
            height: parent.height
            width: parent.height
            isSolid: true
            z: navbarWrapper.z + 1
            icon: UIConstants.iGamepad
            iconSize: UIConstants.sRect*3/2 - 5
            isAutoReturn: true
            iconColor: isEnable ? (!Joystick.useJoystick ? UIConstants.textColor:UIConstants.greenColor) : UIConstants.grayColor
            isEnable: Joystick.connected
            property bool showIndicator: dialogShow === "DIALOG_JOYSTICK"
            onClicked: {
                if(dialogShow !== "DIALOG_JOYSTICK"){
                    dialogShow = "DIALOG_JOYSTICK";
                    rootItem.doShowJoystickConfirm(showIndicator);
                }
            }
        }
        FlatButtonIcon{
            id: btnParams
            anchors.right: btnDistance.left
            anchors.rightMargin: 2
            anchors.top: parent.top
            height: parent.height
            width: parent.height
            isSolid: true
            z: navbarWrapper.z + 1
            icon: UIConstants.iMinorInfo
            iconSize: UIConstants.sRect*3/2 - 5
            isAutoReturn: true
            property bool showIndicator: dialogShow === "DIALOG_PARAMS"
            onClicked: {
                if(dialogShow !== "DIALOG_PARAMS"){
                    dialogShow = "DIALOG_PARAMS";
                    rootItem.doShowParamsSelect(showIndicator);
                }
            }
        }
        FlatButtonIcon{
            id: btnDistance
            anchors.right: btnPayloadControl.left
            anchors.rightMargin: 2
            anchors.top: parent.top
            height: parent.height
            width: parent.height
            isSolid: true
            z: navbarWrapper.z + 1
            icon: UIConstants.iMeasureDistance
            iconSize: UIConstants.sRect*3/2 - 5
            isAutoReturn: false
            onClicked: {
                if(UIConstants.mouseOnMapMode === UIConstants.mouseOnMapModeMeasure)
                    UIConstants.mouseOnMapMode = UIConstants.mouseOnMapModeWaypoint;
                else if(UIConstants.mouseOnMapMode === UIConstants.mouseOnMapModeWaypoint)
                    UIConstants.mouseOnMapMode = UIConstants.mouseOnMapModeMeasure;
                mapPane.updateMouseOnMap();
                rectProfilePath.visible = false;
            }
        }

        FlatButtonIcon{
            id: btnPayloadControl
            anchors.right: systemTime.left
            anchors.rightMargin: 10
            anchors.top: parent.top
            height: parent.height
            width: parent.height
            isSolid: true
            z: navbarWrapper.z + 1
            icon: UIConstants.iPayload
            iconColor: isEnable ? UIConstants.textColor : UIConstants.grayColor
            iconSize: UIConstants.sRect*3/2 - 5
            idx: 3
            isEnable: !btnSystemConfig.isPressed && CAMERA_CONTROL
            onClicked: {
                rootItem._showParams = true;
                if(isActive){
                    setButtonNormal();
                }else{
                    setButtonActive();

                }
                rootItem.itemNavChoosed(idx);

            }
        }
        //----------- System Time
        SystemTime {
            id: systemTime
            height: parent.height
            width: UIConstants.sRect*6.5
            anchors.right: btnSystemConfig.left
            anchors.rightMargin: 10
            anchors.verticalCenter: parent.verticalCenter
        }
        FlatButtonIcon {
            id: btnSystemConfig
            anchors.right: parent.right
            anchors.top: parent.top
            height: parent.height
            width: parent.height
            z: navbarWrapper.z + 1
            icon: UIConstants.iSystemConfig
            isSolid: true
            iconSize: UIConstants.sRect*3/2 - 5
            idx: 4
            property int idxBefore: -1
            onClicked: {
                if(isActive){
                    setButtonNormal();
                    rootItem.changeTabFocus(idxBefore);
                    rootItem.itemNavChoosed(idxBefore);
                    if(idxBefore != 1)
                        rootItem._showParams = true;
                }else{
                    setButtonActive();
                    rootItem.changeTabFocus(idx);
                    rootItem.itemNavChoosed(idx);
                    rootItem._showParams = false;
                }

            }
        }

        //------------ Gradient
        gradient: Gradient {
            GradientStop { position: 0.0; color: UIConstants.cfProcessingOverlayBg }
            GradientStop { position: 0.8; color: UIConstants.bgColorOverlay }
            GradientStop { position: 1.0; color: UIConstants.bgColorOverlay }
        }
    }

    //----------- Informations
    Rectangle {
        id: information
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.top: navbarWrapper.bottom
        height: UIConstants.sRect + 4
        color: UIConstants.transparentBlue
        visible: rootItem._showParams
        RowLayout {
            id: uavInfosGroup
            spacing: 2
            height: UIConstants.sRect
            anchors.verticalCenter: parent.verticalCenter
            anchors.margins: 2
            width: parent.height
            Rectangle{
                id: rectLink
                Layout.alignment: Qt.AlignVCenter
                width: UIConstants.sRect * 3
                height: parent.height
                color: FlightVehicle.link?UIConstants.greenColor:UIConstants.redColor
                radius: UIConstants.rectRadius
                property var timeCount: 0 // time count in second
                Label{
                    id: lblLink
                    color: UIConstants.textColor
                    text: "LINK"
                    anchors.fill: parent
                    verticalAlignment: Text.AlignVCenter
                    horizontalAlignment: Text.AlignHCenter
                    font.family: UIConstants.appFont
                    font.pixelSize: UIConstants.fontSize
                }
                Timer{
                    id: timerLinkLost
                    interval: 1000
                    repeat: true
                    running: true
                    onTriggered: {
                        rectLink.timeCount++;
                        var second = rectLink.timeCount % 60;
                        var minute = (rectLink.timeCount - second) / 60;
                        lblLink.text = (minute< 100?Conv.pad(minute,2):minute)
                                +":"+Conv.pad(second,2);
                    }
                }
                Connections{
                    target: vehicle
                    onLinkChanged:{
                        console.log("onLinkChanged "+FlightVehicle.link)
                        rectLink.timeCount = 0;
                        if(FlightVehicle.link === false){
                            timerLinkLost.start();
                        }else{
                            timerLinkLost.stop();
                            lblLink.text = "LINK";
                        }
                    }
                }
            }
            Rectangle{
                id: rectGPS
                Layout.alignment: Qt.AlignVCenter
                width: UIConstants.sRect * 2
                height: parent.height
                color: FlightVehicle.gpsSignal?UIConstants.greenColor:UIConstants.redColor
                radius: UIConstants.rectRadius
                Label{
                    color: UIConstants.textColor
                    text: "GPS"
                    font.family: UIConstants.appFont
                    anchors.fill: parent
                    verticalAlignment: Text.AlignVCenter
                    horizontalAlignment: Text.AlignHCenter
                    font.pixelSize: UIConstants.fontSize
                }
            }
            Rectangle{
                id: rectEKF
                Layout.alignment: Qt.AlignVCenter
                width: UIConstants.sRect * 2
                height: parent.height
                color: FlightVehicle.ekfSignal !== "green"? FlightVehicle.ekfSignal: UIConstants.greenColor
                radius: UIConstants.rectRadius
                Label{
                    color: UIConstants.textColor
                    text: "EKF"
                    font.family: UIConstants.appFont
                    anchors.fill: parent
                    verticalAlignment: Text.AlignVCenter
                    horizontalAlignment: Text.AlignHCenter
                    font.pixelSize: UIConstants.fontSize
                }
            }
            Rectangle{
                id: rectVIBE
                Layout.alignment: Qt.AlignVCenter
                width: UIConstants.sRect * 2
                height: parent.height
                color: FlightVehicle.vibeSignal !== "green"? FlightVehicle.vibeSignal: UIConstants.greenColor
                radius: UIConstants.rectRadius
                Label{
                    color: UIConstants.textColor
                    text: "VIBE"
                    font.family: UIConstants.appFont
                    anchors.fill: parent
                    verticalAlignment: Text.AlignVCenter
                    horizontalAlignment: Text.AlignHCenter
                    font.pixelSize: UIConstants.fontSize
                }
            }
            Rectangle{
                id: rectPICCIC
                Layout.alignment: Qt.AlignVCenter
                width: UIConstants.sRect * 2
                height: parent.height
                color: Joystick.pic?UIConstants.blueColor:UIConstants.greenColor
                radius: UIConstants.rectRadius
                Label{
                    color: UIConstants.textColor
                    text: Joystick.pic?"PIC":"CIC"
                    font.family: UIConstants.appFont
                    anchors.fill: parent
                    verticalAlignment: Text.AlignVCenter
                    horizontalAlignment: Text.AlignHCenter
                    font.pixelSize: UIConstants.fontSize
                }
            }
            Label{
                id: lblU1
                Layout.alignment: Qt.AlignVCenter
                text: "U1"
                font.pixelSize: UIConstants.fontSize
                font.family: UIConstants.appFont
                color: UIConstants.textColor
                width: UIConstants.sRect
            }
            Rectangle{
                id: rectU1
                Layout.alignment: Qt.AlignVCenter
                width: UIConstants.sRect * 3
                height: parent.height
                color: "transparent"
                border.color: "gray"
                radius: UIConstants.rectRadius
                Label{
                    color: UIConstants.textColor
                    font.pixelSize: UIConstants.fontSize
                    font.family: UIConstants.appFont
                    text: (vehicle?Number(FlightVehicle.batteryVoltage).toFixed(2).toString():"")+"V"
                    anchors.fill: parent
                    verticalAlignment: Text.AlignVCenter
                    horizontalAlignment: Text.AlignHCenter
                }
            }
            Label{
                id: lblI1
                Layout.alignment: Qt.AlignVCenter
                text: "I1"
                font.pixelSize: UIConstants.fontSize
                font.family: UIConstants.appFont
                color: UIConstants.textColor
                width: UIConstants.sRect
            }
            Rectangle{
                id: rectI1
                Layout.alignment: Qt.AlignVCenter
                width: UIConstants.sRect * 3
                height: parent.height
                color: "transparent"
                border.color: "gray"
                radius: UIConstants.rectRadius
                Label{
                    color: UIConstants.textColor
                    font.pixelSize: UIConstants.fontSize
                    font.family: UIConstants.appFont
                    text: (vehicle?Number(FlightVehicle.batteryAmpe).toFixed(2).toString():"")+"A"
                    anchors.fill: parent
                    verticalAlignment: Text.AlignVCenter
                    horizontalAlignment: Text.AlignHCenter
                }
            }
        }

        RowLayout {
            id: uavInfosGroup02
            spacing: 2
            height: parent.height
            width: parent.height
            anchors.right: parent.right
            layoutDirection: Qt.RightToLeft
            RowLayout{
                id: rectFlightTime
                Layout.fillWidth: true
                Label{
                    id: lblFlightTime
                    Layout.alignment: Qt.AlignVCenter
                    text: itemListName["FLIGHT_TIME"][UIConstants.language[UIConstants.languageID]]
                    font.pixelSize: UIConstants.fontSize
                    font.family: UIConstants.appFont
                    color: UIConstants.textColor
                    width: UIConstants.sRect
                }
                Rectangle{
                    id: rectTxtFlightTime
                    Layout.alignment: Qt.AlignVCenter
                    width: UIConstants.sRect * 4
                    height: UIConstants.sRect
                    color: "transparent"
                    border.color: "gray"
                    radius: UIConstants.rectRadius
                    Label{
                        id: lblFlightTimeData
                        color: UIConstants.textColor
                        font.family: UIConstants.appFont
                        font.pixelSize: UIConstants.fontSize
                        text: "00:00:00"
                        anchors.fill: parent
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                    }
                }
                Connections{
                    target: vehicle
                    onLandedChanged:{
                        if(FlightVehicle.landed === false && _oldLandedFlag === true){
                            console.log("flight time started")
                            _flightTime = 0;
                            timerFlightTime.start();
                        }else if(FlightVehicle.landed === true){
                            timerFlightTime.stop();
                        }
                        _oldLandedFlag = FlightVehicle.landed;
                    }
                }

                Timer{
                    id: timerFlightTime
                    interval: 1000
                    repeat: true
                    running: false
                    onTriggered: {
                        _flightTime++;
//                        console.log("_flightTime = "+_flightTime);
                        _second = Number(_flightTime % 60).toFixed(0);

                        _minute = Number((_flightTime-_second) / 60 % 60).toFixed(0);
                        _hour   = Number((_flightTime - _second - _minute*60)/3600).toFixed(0);
                        lblFlightTimeData.text = (_hour < 10?Conv.pad(_hour,2):Number(_hour).toFixed(0).toString())+":"+
                                            Conv.pad(_minute,2)+":"+
                                            Conv.pad(_second,2)
                    }
                }
            }
            RowLayout{
                id: rectWP0
                Layout.fillWidth: true
                Label{
                    id: lblWP0
                    Layout.alignment: Qt.AlignVCenter
                    text: itemListName["UAV_WP"][UIConstants.language[UIConstants.languageID]]+"["+
                          (vehicle?Number(FlightVehicle.currentWaypoint).toFixed(0).toString():"")
                    +"]"
                    color: UIConstants.textColor
                    font.pixelSize: UIConstants.fontSize
                    font.family: UIConstants.appFont
                    width: UIConstants.sRect
                }
                Rectangle{
                    id: rectTxtWP0
                    Layout.alignment: Qt.AlignVCenter
                    width: UIConstants.sRect * 4
                    height: UIConstants.sRect
                    color: "transparent"
                    border.color: "gray"
                    radius: UIConstants.rectRadius
                    Label{
                        color: UIConstants.textColor
                        text: vehicle? (Number(FlightVehicle.distanceToCurrentWaypoint).toFixed(2).toString() + "m"):"0m"
                        anchors.fill: parent
                        font.pixelSize: UIConstants.fontSize
                        font.family: UIConstants.appFont
                        verticalAlignment: Text.AlignVCenter
                        horizontalAlignment: Text.AlignHCenter
                    }
                }
            }
            RowLayout{
                id: rectHome
                Layout.fillWidth: true
                Label{
                    id: lblHome
                    Layout.alignment: Qt.AlignVCenter
                    text: itemListName["UAV_HOME"][UIConstants.language[UIConstants.languageID]]
                    font.pixelSize: UIConstants.fontSize
                    font.family: UIConstants.appFont
                    color: UIConstants.textColor
                    width: UIConstants.sRect
                }
                Rectangle{
                    id: rectTxtHome
                    Layout.alignment: Qt.AlignVCenter
                    width: UIConstants.sRect * 4
                    height: UIConstants.sRect
                    color: "transparent"
                    border.color: "gray"
                    radius: UIConstants.rectRadius
                    Label{
                        color: UIConstants.textColor
                        font.pixelSize: UIConstants.fontSize
                        font.family: UIConstants.appFont
                        text: vehicle? (Number(FlightVehicle.distanceToHome).toFixed(2).toString() + "m"):"0m"
                        anchors.fill: parent
                        verticalAlignment: Text.AlignVCenter
                        horizontalAlignment: Text.AlignHCenter
                    }
                }
            }
            RowLayout{
                id: rectFlightMode
                Layout.fillWidth: true
                Label{
                    id: lblFlightMode
                    Layout.alignment: Qt.AlignVCenter
                    font.pixelSize: UIConstants.fontSize
                    text: itemListName["FLIGHT_MODE"][UIConstants.language[UIConstants.languageID]]
                    font.family: UIConstants.appFont
                    color: UIConstants.textColor
                    width: UIConstants.sRect
                }
                Rectangle{
                    id: rectTxtFlightMode
                    Layout.alignment: Qt.AlignVCenter
                    width: UIConstants.sRect * 5
                    height: UIConstants.sRect
                    color: "transparent"
                    border.color: "gray"
                    radius: UIConstants.rectRadius
                    Label{
                        id: lblFlightModeData
                        color: UIConstants.textColor
                        font.pixelSize: UIConstants.fontSize
                        font.family: UIConstants.appFont
//                        text: lstFlightMode.model.length > 0? lstFlightMode.model[lstFlightMode.currentIndex]:"UNDEFINED"
                        text: FlightVehicle.flightMode
                        anchors.fill: parent
                        verticalAlignment: Text.AlignVCenter
                        horizontalAlignment: Text.AlignHCenter
                    }
                    MouseArea{
                        anchors.fill: parent
//                        enabled: lstFlightMode.model.length > 0
                        onClicked: {
                            if(dialogShow !== "FLIGHT_MODES"){
                                dialogShow = "FLIGHT_MODES";
//                                lstFlightMode.setCurrentText(lstFlightMode.prevItem);
                            }else{
                                dialogShow = "";
                            }
                        }
                    }
                    Connections{
                        target: mapPane
                        onMapClicked:{
                            if(dialogShow === "FLIGHT_MODES"){
                                footerBar.isShowConfirm = false;
                                if(footerBar.confirmDialogObj !== undefined &&
                                        footerBar.confirmDialogObj !== null){
                                    lstFlightMode.setCurrentText(lstFlightMode.prevItem);
                                    footerBar.confirmDialogObj.destroy();
                                }
                                if(footerBar.compo !== undefined &&
                                        footerBar.compo !== null)
                                    footerBar.compo.destroy();
                                dialogShow = "";
                            }
                        }
                    }
                    MouseArea{
                        anchors.fill: lstFlightMode
                        visible: lstFlightMode.visible
                        hoverEnabled: true
                    }

                    SubNav{
                        id: lstFlightMode
                        anchors.top: parent.bottom
                        anchors.horizontalCenter: parent.horizontalCenter
                        width: parent.width*1.5
                        height: UIConstants.sRect*model.length
                        visible: dialogShow === "FLIGHT_MODES"
                        enabled: !footerBar.isShowConfirm
                        color: UIConstants.transparentBlue
                        orientation: ListView.Vertical
                        layoutDirection: Qt.LeftToRight
                        model: []
                        onListViewClicked: {
                            rootItem.doSwitchPlaneMode(prevItem,choosedItem);
                        }
                    }
                }
            }
        }
    }

    Rectangle{
        id: rectParamLoading
        anchors.top: information.bottom
        height: 5
        width: 0
        color: UIConstants.greenColor
        anchors.left: parent.left
        visible: rootItem._showParams
        Connections{
            target: FlightVehicle.planController
            onProgressPct:{
                rectParamLoading.width = rootItem.width*progressPercentPct;
            }
            onRequestMissionDone:{
                rectParamLoading.width = 0;
            }
            onUploadMissionDone:{
                rectParamLoading.width = 0;
            }

        }
    }

    Component.onCompleted: {
        rootItem.changeTabFocus(2);
    }
}

