import QtQuick 2.9
import QtQuick.Controls 2.2
import QtQuick.Controls.Styles 1.0
//------------------ Include Custom modules/plugins
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0
Rectangle {
    id: root
    color: "transparent"
    width: 650
    height:450
    property bool bold: false
    property bool hudGcsEnable: camState.hudGcsEnable
    property bool hudCameraEnable: camState.hudCameraEnable
    property bool digitalStab: camState.digitalStab

    GroupBox {
        id: gbxGCSOverlay
        x: 8
        y: 10
        width: 310
        height: 243
        title: qsTr("GCS Overlay")
        label: Text {
            color: "white"
            font.bold: root.bold
            text: parent.title
            elide: Text.ElideLeft
            horizontalAlignment: Text.AlignLeft
            verticalAlignment: Text.AlignTop
        }

        CheckBox {
            id: cbxSignalUAV
            x: -12
            y: 47
            width: 135
            height: 32
            text: "<font color=\"white\">Plane RSSI</font>"
            onCheckedChanged: {
                camState.planeRSSIVisible = !camState.planeRSSIVisible
            }
        }

        CheckBox {
            id: cbxCenter
            x: 151
            y: 47
            width: 135
            height: 32
            text: "<font color=\"white\">Center</font>"
            checked: true
            onCheckedChanged: {
                camState.centerVisible = !camState.centerVisible
            }
        }

        CheckBox {
            id: cbxSignalGDT
            x: -12
            y: 85
            width: 135
            height: 32
            text: "<font color=\"white\">GDT RSSI</font>"
            onCheckedChanged: {
                camState.gdtRSSIVisible = !camState.gdtRSSIVisible
            }
        }

        CheckBox {
            id: cbxTarget
            x: 151
            y: 85
            width: 135
            height: 32
            text: "<font color=\"white\">Target info</font>"
            onCheckedChanged: {
                camState.targetVisible = !camState.targetVisible
            }
        }

        CheckBox {
            id: cbxFlightMode
            x: -12
            y: 123
            width: 135
            height: 32
            text: "<font color=\"white\">Flight Mode</font>"
            onCheckedChanged: {
                camState.flightModeVisible = !camState.flightModeVisible
            }
        }

        CheckBox {
            id: cbxNavigator
            x: 151
            y: 123
            width: 135
            height: 32
            text: "<font color=\"white\">Navigator</font>"
            onCheckedChanged: {
                camState.navigatorVisible = !camState.navigatorVisible
            }
        }

        CheckBox {
            id: cbxZoom
            x: -12
            y: 161
            width: 135
            height: 32
            text: "<font color=\"white\">Zoom</font>"
            checked: true
            onCheckedChanged: {
                camState.zoomVisible = !camState.zoomVisible
            }
        }

        Button {
            id: btnGCSOverlay
            x: -6
            y: -7
            width: 298
            height: 48
            text: hudGcsEnable === false? qsTr("ENABLE"):qsTr("DISABLE")
            onClicked: {
                hudGcsEnable = !hudGcsEnable;
                camState.hubVisible = hudGcsEnable;
            }
        }

    }
    GroupBox {
        id: gbxVideoOverlay
        x: 332
        y: 10
        width: 310
        height: 337
        title: qsTr("CM160 Overlay")
        label: Text {
            color: "white"
            font.bold: root.bold
            text: parent.title
            elide: Text.ElideLeft
            horizontalAlignment: Text.AlignLeft
            verticalAlignment: Text.AlignTop
        }

        CheckBox {
            id: cbxTrackBoxes
            x: -12
            y: 48
            width: 135
            height: 32
            text: "<font color=\"white\">Track Boxes</font>"
            checked: true
            onCheckedChanged: {
                if(camState.isConnected && camState.isPingOk && CameraController.gimbal.isConnected){
                    CameraController.gimbal.setOverlay(
                                hudCameraEnable,
                                cbxLaserActive.checked,
                                cbxLimitWarning.checked,
                                cbxStabMode.checked,
                                cbxGimbalMode.checked,
                                cbxTrackBoxes.checked,
                                cbxHFOV.checked,
                                cbxSlantRange.checked,
                                cbxTargetLoc.checked,
                                cbxTimestamp.checked,
                                cbxCrosshair.checked);
                }
            }
        }

        CheckBox {
            id: cbxTargetLoc
            x: -12
            y: 86
            width: 135
            height: 32
            text: "<font color=\"white\">Target Loc</font>"
            onCheckedChanged: {
                if(camState.isConnected && camState.isPingOk && CameraController.gimbal.isConnected){
                    CameraController.gimbal.setOverlay(
                                hudCameraEnable,
                                cbxLaserActive.checked,
                                cbxLimitWarning.checked,
                                cbxStabMode.checked,
                                cbxGimbalMode.checked,
                                cbxTrackBoxes.checked,
                                cbxHFOV.checked,
                                cbxSlantRange.checked,
                                cbxTargetLoc.checked,
                                cbxTimestamp.checked,
                                cbxCrosshair.checked);
                }
            }
        }

        CheckBox {
            id: cbxHFOV
            x: -12
            y: 125
            width: 135
            height: 32
            text: "<font color=\"white\">H-FOV</font>"
            onCheckedChanged: {
                if(camState.isConnected && camState.isPingOk && CameraController.gimbal.isConnected){
                    CameraController.gimbal.setOverlay(
                                hudCameraEnable,
                                cbxLaserActive.checked,
                                cbxLimitWarning.checked,
                                cbxStabMode.checked,
                                cbxGimbalMode.checked,
                                cbxTrackBoxes.checked,
                                cbxHFOV.checked,
                                cbxSlantRange.checked,
                                cbxTargetLoc.checked,
                                cbxTimestamp.checked,
                                cbxCrosshair.checked);
                }
            }
        }

        CheckBox {
            id: cbxLimitWarning
            x: -12
            y: 163
            width: 149
            height: 32
            text: "<font color=\"white\">Limit Warning</font>"
            onCheckedChanged: {
                if(camState.isConnected && camState.isPingOk && CameraController.gimbal.isConnected){
                    CameraController.gimbal.setOverlay(
                                hudCameraEnable,
                                cbxLaserActive.checked,
                                cbxLimitWarning.checked,
                                cbxStabMode.checked,
                                cbxGimbalMode.checked,
                                cbxTrackBoxes.checked,
                                cbxHFOV.checked,
                                cbxSlantRange.checked,
                                cbxTargetLoc.checked,
                                cbxTimestamp.checked,
                                cbxCrosshair.checked);
                }
            }
        }

        CheckBox {
            id: cbxCrosshair
            x: -12
            y: 201
            width: 135
            height: 32
            text: "<font color=\"white\">Crosshair</font>"
            onCheckedChanged: {
                if(camState.isConnected && camState.isPingOk && CameraController.gimbal.isConnected){
                    CameraController.gimbal.setOverlay(
                                hudCameraEnable,
                                cbxLaserActive.checked,
                                cbxLimitWarning.checked,
                                cbxStabMode.checked,
                                cbxGimbalMode.checked,
                                cbxTrackBoxes.checked,
                                cbxHFOV.checked,
                                cbxSlantRange.checked,
                                cbxTargetLoc.checked,
                                cbxTimestamp.checked,
                                cbxCrosshair.checked);
                }
            }
        }

        CheckBox {
            id: cbxSlantRange
            x: 151
            y: 48
            width: 135
            height: 32
            text: "<font color=\"white\">Slant Range</font>"
            onCheckedChanged: {
                if(camState.isConnected && camState.isPingOk && CameraController.gimbal.isConnected){
                    CameraController.gimbal.setOverlay(
                                hudCameraEnable,
                                cbxLaserActive.checked,
                                cbxLimitWarning.checked,
                                cbxStabMode.checked,
                                cbxGimbalMode.checked,
                                cbxTrackBoxes.checked,
                                cbxHFOV.checked,
                                cbxSlantRange.checked,
                                cbxTargetLoc.checked,
                                cbxTimestamp.checked,
                                cbxCrosshair.checked);
                }
            }
        }

        CheckBox {
            id: cbxTimestamp
            x: 151
            y: 86
            width: 135
            height: 32
            text: "<font color=\"white\">Timestamp</font>"
            onCheckedChanged: {
                if(camState.isConnected && camState.isPingOk && CameraController.gimbal.isConnected){
                    CameraController.gimbal.setOverlay(
                                hudCameraEnable,
                                cbxLaserActive.checked,
                                cbxLimitWarning.checked,
                                cbxStabMode.checked,
                                cbxGimbalMode.checked,
                                cbxTrackBoxes.checked,
                                cbxHFOV.checked,
                                cbxSlantRange.checked,
                                cbxTargetLoc.checked,
                                cbxTimestamp.checked,
                                cbxCrosshair.checked);
                }
            }
        }

        CheckBox {
            id: cbxStabMode
            x: 151
            y: 125
            width: 135
            height: 32
            text: "<font color=\"white\">Stab Mode</font>"
            onCheckedChanged: {
                if(camState.isConnected && camState.isPingOk && CameraController.gimbal.isConnected){
                    CameraController.gimbal.setOverlay(
                                hudCameraEnable,
                                cbxLaserActive.checked,
                                cbxLimitWarning.checked,
                                cbxStabMode.checked,
                                cbxGimbalMode.checked,
                                cbxTrackBoxes.checked,
                                cbxHFOV.checked,
                                cbxSlantRange.checked,
                                cbxTargetLoc.checked,
                                cbxTimestamp.checked,
                                cbxCrosshair.checked);
                }
            }
        }

        CheckBox {
            id: cbxGimbalMode
            x: 151
            y: 163
            width: 147
            height: 32
            text: "<font color=\"white\">Gimbal Mode</font>"
            onCheckedChanged: {
                if(camState.isConnected && camState.isPingOk && CameraController.gimbal.isConnected){
                    CameraController.gimbal.setOverlay(
                                hudCameraEnable,
                                cbxLaserActive.checked,
                                cbxLimitWarning.checked,
                                cbxStabMode.checked,
                                cbxGimbalMode.checked,
                                cbxTrackBoxes.checked,
                                cbxHFOV.checked,
                                cbxSlantRange.checked,
                                cbxTargetLoc.checked,
                                cbxTimestamp.checked,
                                cbxCrosshair.checked);
                }
            }
        }

        CheckBox {
            id: cbxLaserActive
            x: 151
            y: 201
            width: 135
            height: 32
            text: "<font color=\"white\">Laser Active</font>"
            onCheckedChanged: {
                if(camState.isConnected && camState.isPingOk && CameraController.gimbal.isConnected){
                    CameraController.gimbal.setOverlay(
                                hudCameraEnable,
                                cbxLaserActive.checked,
                                cbxLimitWarning.checked,
                                cbxStabMode.checked,
                                cbxGimbalMode.checked,
                                cbxTrackBoxes.checked,
                                cbxHFOV.checked,
                                cbxSlantRange.checked,
                                cbxTargetLoc.checked,
                                cbxTimestamp.checked,
                                cbxCrosshair.checked);
                }
            }
        }

        Button {
            id: btnCameraOverlay
            x: -6
            y: -6
            width: 298
            height: 48
            text: hudCameraEnable === false? qsTr("ENABLE"):qsTr("DISABLE")
            onClicked: {
                hudCameraEnable = !hudCameraEnable;
                camState.hudCameraEnable = hudCameraEnable;
                if(camState.isConnected && camState.isPingOk && CameraController.gimbal.isConnected){
                    CameraController.gimbal.setOverlay(
                                hudCameraEnable,
                                cbxLaserActive.checked,
                                cbxLimitWarning.checked,
                                cbxStabMode.checked,
                                cbxGimbalMode.checked,
                                cbxTrackBoxes.checked,
                                cbxHFOV.checked,
                                cbxSlantRange.checked,
                                cbxTargetLoc.checked,
                                cbxTimestamp.checked,
                                cbxCrosshair.checked);
                }
            }
        }
    }
    GroupBox {
        id: gbxEStabCamera
        x: 8
        y: 259
        width: 310
        height: 88
        title: qsTr("CM160 EStab")
        label: Text {
            color: "white"
            font.bold: root.bold
            text: parent.title
            elide: Text.ElideLeft
            horizontalAlignment: Text.AlignLeft
            verticalAlignment: Text.AlignTop
        }

        Button {
            id: btnEStab
            x: -6
            y: -6
            width: 79
            height: 48
            text: digitalStab === false? qsTr("ENABLE"):qsTr("DISABLE")
            onClicked: {
                digitalStab = !digitalStab;
                camState.digitalStab = digitalStab;
                if(camState.isConnected && camState.isPingOk && CameraController.gimbal.isConnected){
                    CameraController.gimbal.setEStabilisationParameters(
                                root.digitalStab,
                                255,255,15,camState.stabBackground);
                    CameraController.gimbal.setStabiliseOnTrack(true);
                }
            }
        }

        ComboBox {
            id: cbxEStabBackGround
            x: 88
            y: -6
            width: 203
            height: 48
            model: ["Previous Background","Gray Background"]
            onCurrentIndexChanged: {
                camState.stabBackground = currentIndex;
                if(camState.isConnected && camState.isPingOk && CameraController.gimbal.isConnected){
                    CameraController.gimbal.setEStabilisationParameters(
                                root.digitalStab,
                                255,255,15,camState.stabBackground);
                    CameraController.gimbal.setStabiliseOnTrack(true);
                }
            }
        }
    }
}
