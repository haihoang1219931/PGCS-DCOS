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

Rectangle {
    id: rootItem
    width: 40
    height: 768
    color: UIConstants.transparentBlue
//    border.color: "gray"
//    border.width: 1
    radius: UIConstants.rectRadius
    property int spacing: 2
    property alias currentIndex: rightBarStack.currentIndex
    property var itemListName:  UIConstants.itemTextMultilanguages["RIGHTBAR"]
    function hideRightBar(){
        btnEditMarker.isPressed = false;
    }


    //-------------------- Childrems
    StackLayout {
        id: rightBarStack
        anchors.fill: parent
        currentIndex: 0
        //--- Index = 0
        ColumnLayout {
            id: group1
            x: 0
            spacing: rootItem.spacing
            height: parent.height
            width: parent.width
            FooterButton {
                id: btnSensor
                Layout.preferredWidth: parent.width
                Layout.preferredHeight: parent.width
                icon: UIConstants.iSensor
                iconRotate: camState.sensorID === camState.sensorIDEO?0:180
                btnText: itemListName["SENSOR"]
                         [UIConstants.language[UIConstants.languageID]]
                color: UIConstants.bgAppColor
                onClicked: {
                    if(USE_VIDEO_CPU || USE_VIDEO_GPU){
                        if(camState.sensorID === camState.sensorIDIR){
                            cameraController.gimbal.changeSensor("EO");
                        }else{
                            cameraController.gimbal.changeSensor("IR");
                        }
                    }
                }
            }

//            FooterButton {
//                id: btnObserveMode
//                Layout.preferredWidth: parent.width
//                Layout.preferredHeight: parent.width
//                icon: UIConstants.iObserve
//                btnText: "Observe\nMode"
//                color: UIConstants.bgAppColor
//                onClicked: {
//                }
//            }

            FooterButton {
                id: btnPreset
                Layout.preferredWidth: parent.width
                Layout.preferredHeight: parent.width
                icon: UIConstants.iPreset
                btnText: itemListName["PRESET"][camState.presetMode][UIConstants.language[UIConstants.languageID]]
                color: UIConstants.bgAppColor
                onClicked: {
//                    btnPreset.setButtonDisable()
                    cameraController.gimbal.setGimbalPreset("NEXT");
                }
//                Connections {
//                    target: cameraController.gimbal
//                    onPresetChanged:{
//                        console.log("end preset");
//                        if(result === true && btnPreset.isEnable === false)
//                        {
//                            switch(btnPreset.btnText)
//                            {
//                            case itemListName["MISSION_PRESET_FRONT"][UIConstants.language[UIConstants.languageID]]:
//                                btnPreset.btnText = itemListName["MISSION_PRESET_RIGHTWING"][UIConstants.language[UIConstants.languageID]]
//                                break;
//                            case itemListName["MISSION_PRESET_RIGHTWING"][UIConstants.language[UIConstants.languageID]]:
//                                btnPreset.btnText = itemListName["MISSION_PRESET_BEHIND"][UIConstants.language[UIConstants.languageID]]
//                                break;
//                            case itemListName["MISSION_PRESET_BEHIND"][UIConstants.language[UIConstants.languageID]]:
//                                btnPreset.btnText = itemListName["MISSION_PRESET_LEFTWING"][UIConstants.language[UIConstants.languageID]]
//                                break;
//                            case itemListName["MISSION_PRESET_LEFTWING"][UIConstants.language[UIConstants.languageID]]:
//                                btnPreset.btnText = itemListName["MISSION_PRESET_NADIR"][UIConstants.language[UIConstants.languageID]]
//                                break;
//                            case itemListName["MISSION_PRESET_NADIR"][UIConstants.language[UIConstants.languageID]]:
//                                btnPreset.btnText = itemListName["MISSION_PRESET_FRONT"][UIConstants.language[UIConstants.languageID]]
//                                break;
//                            default:
//                                break;
//                            }
//                        }
//                        btnPreset.setButtonEnable()
//                    }
//                }
            }
            FooterButton {
                id: btnSensorColor
                Layout.preferredWidth: parent.width
                Layout.preferredHeight: parent.width
                icon: itemListName["COLOR"][camState.sensorID][camState.colorMode[camState.sensorID]]
                        [UIConstants.language[UIConstants.languageID]]
                iconSize: UIConstants.fontSize
                btnText: itemListName["COLOR"][camState.sensorID]["TILTLE"]
                         [UIConstants.language[UIConstants.languageID]]
                color: UIConstants.bgAppColor

                onClicked: {
                    if(USE_VIDEO_CPU || USE_VIDEO_GPU){
                        var color = camState.colorMode[camState.sensorID];
                        if(color === "WHITE_HOT"){
                            color = "COLOR";
                        }else{
                            color = "WHITE_HOT";
                        }
                        camState.colorMode = {"EO":"NORMAL","IR":color};
                        cameraController.gimbal.setSensorColor(camState.sensorID,color);
                    }
                }
            }
            FooterButton {
                id: btnSnapshot
                Layout.preferredWidth: parent.width
                Layout.preferredHeight: parent.width
                icon: UIConstants.iSnapshot
                btnText: itemListName["SNAPSHOT"]
                         [UIConstants.language[UIConstants.languageID]]
                color: UIConstants.bgAppColor
                onClicked: {
                    if(USE_VIDEO_CPU || USE_VIDEO_GPU){
                        cameraController.gimbal.snapShot();
                    }
                }
            }
//            FooterButton {
//                id: btnMeasureDistance
//                Layout.preferredWidth: parent.width
//                Layout.preferredHeight: parent.width
//                icon: UIConstants.iMeasureDistance
//                btnText: "Measure\ndistance"
//                color: UIConstants.bgAppColor
//                onClicked: {
//                }
//            }
//            SwitchFlatButton {
//                id: btnGimbalRecord
//                Layout.preferredWidth: parent.width
//                Layout.preferredHeight: parent.width
//                icon: UIConstants.iGCSRecord
//                btnText: "Gimbal\nrecord"
//                color: UIConstants.bgAppColor
//                isSync: false
//                isOn: camState.gimbalRecord
//                onClicked: {
//                    if(gimbalNetwork.isGimbalConnected === true){
//                        gimbalNetwork.ipcCommands.changeRecordMode(!isOn?"RECORD_OFF":"RECORD_FULL",0,0);
//                    }
//                }
//            }            
            SwitchFlatButton {
                id: btnGCSRecord
                Layout.preferredWidth: parent.width
                Layout.preferredHeight: parent.width
                icon: UIConstants.iGCSRecord
                btnText: itemListName["GCS_RECORD"]
                         [UIConstants.language[UIConstants.languageID]]
                color: UIConstants.bgAppColor
                isSync: true
                isOn: camState.record
                onClicked: {
//                    console.log("setVideoSavingState to "+camState.gcsRecord)
                    if(USE_VIDEO_CPU || USE_VIDEO_GPU){
                        cameraController.gimbal.setRecord(!camState.record);
                    }
                }
            }
            SwitchFlatButton {
                id: btnGCSObjectDetect
                Layout.preferredWidth: parent.width
                Layout.preferredHeight: parent.width
                icon: UIConstants.iCar
                btnText: itemListName["OBJECT_DETECT"]
                         [UIConstants.language[UIConstants.languageID]]
                color: UIConstants.bgAppColor
                isSync: true
                isOn: camState.gcsOD
                onClicked: {
                    camState.gcsOD=!camState.gcsOD;
//                    console.log("setVideoSavingState to "+camState.gcsRecord)
                    if(USE_VIDEO_CPU || USE_VIDEO_GPU){
                        cameraController.videoEngine.setObjectDetect(camState.gcsOD);
                    }
                }
            }
            SwitchFlatButton {
                id: btnGCSLineDetect
                Layout.preferredWidth: parent.width
                Layout.preferredHeight: parent.width
                icon: UIConstants.iBolt
                btnText: itemListName["POWERLINE_DETECT"]
                         [UIConstants.language[UIConstants.languageID]]
                color: UIConstants.bgAppColor
                isSync: true
                isOn: camState.gcsPD
                onClicked: {
                    camState.gcsPD=!camState.gcsPD;
//                    console.log("setVideoSavingState to "+camState.gcsRecord)
                    if(USE_VIDEO_CPU || USE_VIDEO_GPU){
                        cameraController.videoEngine.setPowerLineDetect(camState.gcsPD);
                    }
                }
            }

//            SwitchFlatButton {
//                id: btnVisualLock
//                Layout.preferredWidth: parent.width
//                Layout.preferredHeight: parent.width
//                icon: UIConstants.iVisualLock
//                isSync: true
//                btnText: "Visual\nlock"
//                color: UIConstants.bgAppColor
//                isOn: camState.lockMode === "VISUAL"
//                onClicked: {
//                    if(isOn){
//                        camState.changeLockMode("FREE")
//                        if(gimbalNetwork.isGimbalConnected === true)
//                            gimbalNetwork.ipcCommands.changeLockMode("LOCK_FREE", "GEOLOCATION_OFF");
//                    }else{
//                        camState.changeLockMode("VISUAL")
//                            if(gimbalNetwork.isGimbalConnected === true)
//                        gimbalNetwork.ipcCommands.doSceneSteering(cameraController.gimbal.frameID);
//                    }
//                }
//            }

//            SwitchFlatButton {
//                id: btnDigitalStab
//                Layout.preferredWidth: parent.width
//                Layout.preferredHeight: parent.width
//                icon: UIConstants.iDigitalStab
//                isSync: true
//                btnText: "Digital\nStab"
//                color: UIConstants.bgAppColor
//                isOn: camState.digitalStab
//                onClicked: {

//                    if(CAMERA_CONTROL){
//                        if(gimbalNetwork.isGimbalConnected){
//                            console.log("set digital stab to "+!isOn);
//                            gimbalNetwork.ipcCommands.enableImageStab(!isOn?"ISTAB_ON":"ISTAB_OFF", !isOn?0.2:0.0);
//                        }
//                    }
//                }
//            }

//            FooterButton {
//                id: btnVideoConfig
//                Layout.preferredWidth: parent.width
//                Layout.preferredHeight: parent.width
//                icon: UIConstants.iVideoConfig
//                btnText: "Video\nconfig"
//                color: UIConstants.bgAppColor
//                onClicked: {
//                    configPane.visible = !configPane.visible;
//                }
//            }
        }
        ColumnLayout {
            id: group2
            x: 0
            spacing: rootItem.spacing
            height: parent.height
            width: parent.width
            SwitchFlatButton {
                id: btnVideoExternal
                Layout.preferredWidth: parent.width
                Layout.preferredHeight: parent.width
                icon: UIConstants.iDisplay
                btnText: itemListName["EXPORT_VIDEO"]
                         [UIConstants.language[UIConstants.languageID]]
                color: UIConstants.bgAppColor
                isSync: true
                isOn: camState.gcsExportVideo
                onClicked: {
                    if(!camState.gcsExportVide)
                        camState.gcsExportVideo = true;
                }
            }
            SwitchFlatButton {
                id: btnVisualLock
                Layout.preferredWidth: parent.width
                Layout.preferredHeight: parent.width
                icon: UIConstants.iVisualLock
                btnText: itemListName["VISUAL_LOCK"]
                         [UIConstants.language[UIConstants.languageID]]
                color: UIConstants.bgAppColor
                isSync: true
                isOn: camState.lockMode === "VISUAL"
                onClicked: {
                    if(camState.lockMode === "VISUAL"){
                        camState.lockMode = "FREE"
                    }else{
                        camState.lockMode = "VISUAL"
                    }
                    cameraController.gimbal.setLockMode(camState.lockMode);
                }
            }
            SwitchFlatButton {
                id: btnDigitalStab
                Layout.preferredWidth: parent.width
                Layout.preferredHeight: parent.width
                icon: UIConstants.iGCSStab
                isSync: true
                btnText: itemListName["GCS_STAB"]
                         [UIConstants.language[UIConstants.languageID]]
                color: UIConstants.bgAppColor
                isOn: camState.digitalStab
                onClicked: {
                    if(USE_VIDEO_CPU || USE_VIDEO_GPU){
                        cameraController.gimbal.setDigitalStab(!camState.gcsStab)
                    }
                }
            }
            SwitchFlatButton {
                id: btnSearch
                Layout.preferredWidth: parent.width
                Layout.preferredHeight: parent.width
                icon: UIConstants.iGCSStab
                isSync: true
                btnText: itemListName["GCS_SEARCH"]
                         [UIConstants.language[UIConstants.languageID]]
                color: UIConstants.bgAppColor
                isOn: camState.gcsSearch
                onClicked: {
                    if(USE_VIDEO_CPU || USE_VIDEO_GPU){
                        cameraController.gimbal.setObjectSearch(!camState.gcsSearch)
                    }
                }
            }
            SwitchFlatButton {
                id: btnTL
                Layout.preferredWidth: parent.width
                Layout.preferredHeight: parent.width
                icon: UIConstants.iGCSStab
                isSync: true
                btnText: itemListName["GCS_TL"]
                         [UIConstants.language[UIConstants.languageID]]
                color: UIConstants.bgAppColor
                isOn: camState.gcsTargetLocalization
                onClicked: {
                    camState.gcsTargetLocalization = !camState.gcsTargetLocalization
//                    if(USE_VIDEO_CPU || USE_VIDEO_GPU){
//                        cameraController.gimbal.setObjectSearch(!camState.gcsSearch)
//                    }
                }
            }
            FooterButton {
                id: btnInvertPan
                Layout.preferredWidth: parent.width
                Layout.preferredHeight: parent.width
                icon: UIConstants.iInvertPan
                btnText: itemListName["INVERT_PAN"]
                         [UIConstants.language[UIConstants.languageID]]
                color: UIConstants.bgAppColor
                onClicked: {
                    camState.invertPan = !camState.invertPan;
                    joystick.setInvertCam("PAN",camState.invertPan);
                    joystick.saveConfig();
                }
            }
            FooterButton {
                id: btnInvertTilt
                Layout.preferredWidth: parent.width
                Layout.preferredHeight: parent.width
                icon: UIConstants.iInvertTilt
                btnText: itemListName["INVERT_TILT"]
                         [UIConstants.language[UIConstants.languageID]]
                color: UIConstants.bgAppColor
                onClicked: {
                    camState.invertTilt = !camState.invertTilt;
                    joystick.setInvertCam("TILT",camState.invertTilt);
                    joystick.saveConfig();
                }
            }
            FooterButton {
                id: btnInvertZoom
                Layout.preferredWidth: parent.width
                Layout.preferredHeight: parent.width
                icon: UIConstants.iZoomIn
                btnText: itemListName["INVERT_ZOOM"]
                         [UIConstants.language[UIConstants.languageID]]
                color: UIConstants.bgAppColor
                onClicked: {
                    camState.invertZoom = !camState.invertZoom;
                    joystick.setInvertCam("ZOOM",camState.invertZoom);
                    joystick.saveConfig();
                }
            }
        }
    }
    FooterButton {
        id: btnAdvanced
        width: parent.width
        height: parent.width
        anchors.bottom: parent.bottom
        icon: rightBarStack.currentIndex === 0?
                  UIConstants.iAdvanced : UIConstants.iBack
        btnText: rightBarStack.currentIndex === 0?
                 itemListName["ADVANCED"]
                    [UIConstants.language[UIConstants.languageID]]:
                 itemListName["BACK"]
                    [UIConstants.language[UIConstants.languageID]]
        color: UIConstants.bgAppColor
        onClicked: {
            if(rightBarStack.currentIndex > 0)
                rightBarStack.currentIndex--;
            else{
                rightBarStack.currentIndex = 1;
            }
        }
    }

}
