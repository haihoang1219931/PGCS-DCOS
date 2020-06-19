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
    property var itemListName:
    {
    "MISSION_SENSOR":["Sensor","Hình ảnh"],
        // sensor id
        "MISSION_SENSOR_EO":["DAY","NGÀY"],
        "MISSION_SENSOR_IR":["IR","NHIỆT"],
    "MISSION_OBSERV":["Observe\nmode","Chế độ\nTheo dõi"],
        // observe mode
        "MISSION_OBSERV_FREE":["FREE","TỰ DO"],
        "MISSION_OBSERV_VISUAL":["TLOCK","GIỮ\nKHUNG"],
        "MISSION_OBSERV_GEO":["GEO","GIỮ\nTỌA ĐỘ"],
        "MISSION_OBSERV_TRACK":["TRACK","BÁM\nBẮT"],
    "MISSION_PRESET":["Preset","Quan sát\nvị trí"],
        // preset mode
        "MISSION_PRESET_FRONT":["FRONT","PHÍA\nTRƯỚC"],
        "MISSION_PRESET_RIGHT":["RIGHT\nWING","CÁNH\nPHẢI"],
        "MISSION_PRESET_BEHIND":["BEHIND","PHÍA\nSAU"],
        "MISSION_PRESET_LEFT":["LEFT\nWING","CÁNH\nTRÁI"],
        "MISSION_PRESET_NADIR":["NADIR","DƯỚI\nBỤNG"],
        "MISSION_PRESET_OFF":["OFF","OFF"],
    "MISSION_SNAPSHOT":["Snapshot","Chụp ảnh"],
        // snapshot mode
    "MISSION_COLOR":["Daylight\nmode","Màu ảnh"],
        // sensor color
        // eo
        "MISSION_COLOR_EO_AUTO":["AUTO","TỰ\nĐỘNG"],
        "MISSION_COLOR_EO_COLOR":["COLOR","MÀU"],
        "MISSION_COLOR_EO_DAWN":["DAWN","NHẠY\nSÁNG"],
        //ir
        "MISSION_COLOR_IR_WHITE_HOT":["WHITE\nHOT","NHIỆT\nTRẮNG"],
        "MISSION_COLOR_IR_BLACK_HOT":["BLACK\nHOT","NHIỆT\nĐEN"],
        "MISSION_COLOR_IR_REDDISH":["REDDISH","NHIỆT\nĐỎ"],
        "MISSION_COLOR_IR_COLOR":["COLOR","NHIỆT\nMÀU"],
    "MISSION_DEFOG":{
            "EO":["Defog\nEO","Khử mây\nảnh ngày"],
            "IR":["Defog\nIR","Khử mây\nảnh nhiệt"]
        },
        // sensor defog mode
        // eo
        "MISSION_DEFOG_EO_OFF":["OFF","TẮT"],
        "MISSION_DEFOG_EO_AUTO":["AUTO","TỰ\nĐỘNG"],
        "MISSION_DEFOG_EO_LOW":["LOW","THẤP"],
        "MISSION_DEFOG_EO_MEDIUM":["MEDIUM","TRUNG"],
        "MISSION_DEFOG_EO_HIGH":["HIGH","CAO"],
        // ir
        "MISSION_DEFOG_IR_OFF":["OFF","TẮT"],
        "MISSION_DEFOG_IR_AUTO":["AUTO","TỰ\nĐỘNG"],
        "MISSION_DEFOG_IR_LOW":["LOW","THẤP"],
        "MISSION_DEFOG_IR_MEDIUM":["MEDIUM","TRUNG"],
        "MISSION_DEFOG_IR_HIGH":["HIGH","CAO"],
    "MISSION_MEASURE_DISTANCE":["Measure\ndistance","Khoảng\ncách"],
    "MISSION_GIMBAL":["Gimbal\nmode","Chế độ\ngimbal"],
        // gimbal mode
        "MISSION_GIMBAL_OFF":["OFF","TẮT"],
        "MISSION_GIMBAL_ON":["ON","BẬT"],
        "MISSION_GIMBAL_SECURE":["SECURE","AN\nTOÀN"],
        "MISSION_GIMBAL_SLEEP":["SLEEP","NGHỈ"],
        "MISSION_GIMBAL_NA":["?","?"],
    "MISSION_RECORDER":["Gimbal\nrecorder","Ghi lại\nhình ảnh"],
        // recorder mode
        "MISSION_RECORDER_NA":["?","?"],
        "MISSION_RECORDER_OFF":["OFF","TẮT"],
        "MISSION_RECORDER_ON":["ON","BẬT"],
    "MISSION_CONTROLLER":["Gimbal\ncontroller","Điều khiển\ncamera"],
    // selet button
    "SELECT_UAV": ["Select\nUAV","Vị trí\nUAV"],
    "SELECT_GDT": ["Select\nGDT","Vị trí\nGDT"],
    // markers button
    "MARKER_SAVE": ["Save\nmarkers","Lưu lại\nnhãn"],
    "MARKER_LOAD": ["Load\nmarkers","Tải lên\nnhãn"],
    "MARKER_CLEAR": ["Clear\nmarkers","Xóa tất \ncả nnhãn"],
    // map tab
    "MAP_LOOK": ["Look\nhere","Khóa\nvị trí"],
    "MAP_ADD": ["Add\nmarker","Thêm\nnhãn"],
    "MAP_REMOVE": ["Remove\nmarker","Xóa\nnhãn"],
    "MAP_CLEAR": ["Clear\nmarkers","Xóa tất\ncả nhãn"],
    "MAP_EDIT": ["Edit\nmarker","Sửa\nnhãn"],
    // image processing
    "GCS_DETECT": ["Object\ndetect","Phat\nhien"],
    // advance tab
    "ADVANCED":["Advanced","Nâng cao"],
    "BACK":["Back","Trở lại"],
    "ADVANCED_STAB_GIMBAL":["Gimbal\nStab","Cân bằng\nđiện"],
    "ADVANCED_LOCK_VISUAL":["Visual\nlock","Giữ trường\nnhìn"],
    "ADVANCED_STAB_DIGITAL":["Digital\nStab","Cân bằng\nảnh"],
    "ADVANCED_STAB_GCS":["GCS\nStab","Cân bằng\ntrên GCS"],
    "ADVANCED_OBJECT_DETECT":["Object\nDetect","Phat hien\nDoi tuong"],
    "ADVANCED_HUD":["HUD","HUD"],
    "ADVANCED_LANGUAGE":["Language","Ngôn ngữ"],
        // language
        "ADVANCED_LANGUAGE_VI":"VI",
        "ADVANCED_LANGUAGE_EN":"EN",
    "ADVANCED_INVERT_PAN":["Invert\nPan","Đảo\nPan"],
    "ADVANCED_INVERT_TILT":["Invert\nTilt","Đảo\nTilt"],
    "ADVANCED_VIDEO_CONFIG":["Video\nConfig","Cấu hình\nhình ảnh"],
    "ADVANCED_SYSTEM_CONFIG":["System\nConfig","Cấu hình\nhệ thống"],
    }
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
                btnText: itemListName["MISSION_SENSOR"]
                         [camState.language[camState.languageID]]
                color: UIConstants.bgAppColor
                onClicked: {
                    if(USE_VIDEO_CPU || USE_VIDEO_GPU){
                        if(camState.sensorID === camState.sensorIDIR){
                            cameraController.gimbal.changeSensor("EO");
                        }else{
                            cameraController.gimbal.changeSensor("IR");
                        }
                    }
                    if(CAMERA_CONTROL){
                        if(camState.isConnected && camState.isPingOk && gimbalNetwork.isGimbalConnected){
                            gimbalNetwork.ipcCommands.changeSensorID(camState.sensorID);
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
                btnText: itemListName["MISSION_PRESET_"+camState.presetMode][camState.language[camState.languageID]]
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
//                            case itemListName["MISSION_PRESET_FRONT"][camState.language[camState.languageID]]:
//                                btnPreset.btnText = itemListName["MISSION_PRESET_RIGHTWING"][camState.language[camState.languageID]]
//                                break;
//                            case itemListName["MISSION_PRESET_RIGHTWING"][camState.language[camState.languageID]]:
//                                btnPreset.btnText = itemListName["MISSION_PRESET_BEHIND"][camState.language[camState.languageID]]
//                                break;
//                            case itemListName["MISSION_PRESET_BEHIND"][camState.language[camState.languageID]]:
//                                btnPreset.btnText = itemListName["MISSION_PRESET_LEFTWING"][camState.language[camState.languageID]]
//                                break;
//                            case itemListName["MISSION_PRESET_LEFTWING"][camState.language[camState.languageID]]:
//                                btnPreset.btnText = itemListName["MISSION_PRESET_NADIR"][camState.language[camState.languageID]]
//                                break;
//                            case itemListName["MISSION_PRESET_NADIR"][camState.language[camState.languageID]]:
//                                btnPreset.btnText = itemListName["MISSION_PRESET_FRONT"][camState.language[camState.languageID]]
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
                icon: camState.sensorID === camState.sensorIDEO ? "Normal" : colorMap[colorMode]
                iconSize: UIConstants.fontSize
                btnText: camState.sensorID === camState.sensorIDEO ? "EO color":"IR color"
                color: UIConstants.bgAppColor
                property var colorMap: {
                    "WHITE_HOT":"WHITE\nHOT",
                    "COLOR":"COLOR"
                }
                property string colorMode: "WHITE_HOT"
                onClicked: {
                    if(USE_VIDEO_CPU || USE_VIDEO_GPU){
                        var nextColor = colorMode;
                        if(colorMode === "WHITE_HOT"){
                            nextColor = "COLOR";
                        }else{
                            nextColor = "WHITE_HOT";
                        }
                        colorMode = nextColor;
                        cameraController.gimbal.setSensorColor(camState.sensorID,nextColor);
                    }
                }
            }
            FooterButton {
                id: btnSnapshot
                Layout.preferredWidth: parent.width
                Layout.preferredHeight: parent.width
                icon: UIConstants.iSnapshot
                btnText: "Snapshot"
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
                btnText: "GCS\nRecord"
                color: UIConstants.bgAppColor
                isSync: true
                isOn: camState.gcsRecord
                onClicked: {
                    camState.gcsRecord=!camState.gcsRecord;

//                    console.log("setVideoSavingState to "+camState.gcsRecord)
                    if(USE_VIDEO_CPU || USE_VIDEO_GPU){
                        cameraController.gimbal.setRecord(camState.gcsRecord);
                    }
                }
            }
            SwitchFlatButton {
                id: btnGCSObjectDetect
                Layout.preferredWidth: parent.width
                Layout.preferredHeight: parent.width
                icon: UIConstants.iCar
                btnText: "Object\ndetect"
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
                btnText: "Power\nlines"
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
                btnText: "Export\nvideo"
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
                btnText: "Visual\nLock"
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
                btnText: "GCS\nStab"
                color: UIConstants.bgAppColor
                isOn: camState.digitalStab
                onClicked: {
                    if(USE_VIDEO_CPU || USE_VIDEO_GPU){
                        cameraController.gimbal.setDigitalStab(!camState.gcsStab)
                    }
                }
            }
            FooterButton {
                id: btnInvertPan
                Layout.preferredWidth: parent.width
                Layout.preferredHeight: parent.width
                icon: UIConstants.iInvertPan
                btnText: "Invert\npan"
                color: UIConstants.bgAppColor
                onClicked: {
                    camState.invertPan = !camState.invertPan;
                    joystick.setInvert("PAN",camState.invertPan);
                }
            }
            FooterButton {
                id: btnInvertTilt
                Layout.preferredWidth: parent.width
                Layout.preferredHeight: parent.width
                icon: UIConstants.iInvertTilt
                btnText: "Invert\ntilt"
                color: UIConstants.bgAppColor
                onClicked: {
                    camState.invertTilt = !camState.invertTilt;
                    joystick.setInvert("TILT",camState.invertTilt);
                }
            }
            FooterButton {
                id: btnInvertZoom
                Layout.preferredWidth: parent.width
                Layout.preferredHeight: parent.width
                icon: UIConstants.iZoomIn
                btnText: "Invert\nzoom"
                color: UIConstants.bgAppColor
                onClicked: {
                    camState.invertZoom = !camState.invertZoom;
                    joystick.setInvert("ZOOM",camState.invertZoom);
                }
            }
        }
        ColumnLayout {
            id: group3
            x: 0
            spacing: rootItem.spacing
            height: parent.height
            width: parent.width
            FooterButton {
                id: btnAddMarker
                Layout.preferredWidth: parent.width
                Layout.preferredHeight: parent.width
                icon: UIConstants.iAddMarker
                btnText: "Add\nmarker"
                color: UIConstants.bgAppColor
                onClicked: {
                }
            }

            FooterButton {
                id: btnRemoveMarker
                Layout.preferredWidth: parent.width
                Layout.preferredHeight: parent.width
                icon: UIConstants.iRemoveMarker
                btnText: "Remove\nmarker"
                color: UIConstants.bgAppColor
                onClicked: {
                }
            }

            FooterButton {
                id: btnEditMarker
                Layout.preferredWidth: parent.width
                Layout.preferredHeight: parent.width
                icon: UIConstants.iEditMarker
                btnText: "Edit\nmarker"
                color: UIConstants.bgAppColor
                onClicked: {
//                    lstEditMarker.visible = !lstEditMarker.visible
                }
                SubRight{
                    id: lstEditMarker
                    visible: btnEditMarker.isPressed
                    anchors.right: parent.left
                    anchors.rightMargin: 2
                    anchors.top: parent.top
                    anchors.bottom :parent.bottom
                    size: height
                }
            }

            FooterButton {
                id: btnClearMarker
                Layout.preferredWidth: parent.width
                Layout.preferredHeight: parent.width
                icon: UIConstants.iClearMarkers
                btnText: "Clear\nmarker"
                color: UIConstants.bgAppColor
                onClicked: {
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
        btnText: rightBarStack.currentIndex === 0?"Advanced":"Back"
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
