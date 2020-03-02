import QtQuick 2.9
import QtQuick.Controls 2.2
//------------------ Include Custom modules/plugins
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0
Rectangle {
    id: root
    color: "transparent"
    width: 650
    height:450

//    Button {
//        id: btnPanSetParams
//        x: 527
//        y: 42
//        text: "Set Pan"
//        onClicked: {
//            if(camState.isConnected && camState.isPingOk && gimbalNetwork.isGimbalConnected){
//                var kpValue =(panKP.inputText === "") ? 3.5 : Number(panKP.inputText).toFixed(2);
//                var kiValue =(panKI.inputText === "") ? 1 : Number(panKI.inputText).toFixed(2);

//                if(isNaN(kpValue) || isNaN(kiValue)) {
//                    return;
//                }

//                if(kpValue > 5) kpValue = 5;
//                if(kpValue < 1.2) kpValue = 1.2;

//                if(kiValue > 2) kiValue = 2;
//                if(kiValue < 0.3) kiValue = 0.3;

//                gimbalNetwork.montionCCommands.setMCPanParams(kpValue, kiValue);
//                console.log("Pan Params: KP = " + kpValue + " | KI = " + kiValue);
//            }
//        }
//    }

//    CustomizedTextInput {
//        id: panKP
//        x: 74
//        y: 42
//        width: 136
//        height: 40
//        inputTextMax: 12
//    }

//    CustomizedTextInput {
//        id: panKI
//        x: 302
//        y: 42
//        width: 136
//        height: 40
//        inputTextMax: 12
//    }

//    Button {
//        id: btnTiltSetParams
//        x: 527
//        y: 151
//        text: "Set Tilt"
//        onClicked: {
//            if(camState.isConnected && camState.isPingOk && gimbalNetwork.isGimbalConnected){
//                var kpValue =(tiltKP.inputText === "") ? 5 : Number(tiltKP.inputText).toFixed(2);
//                var kiValue =(tiltKI.inputText === "") ? 5 : Number(tiltKI.inputText).toFixed(2);

//                if(isNaN(kpValue) || isNaN(kiValue)) {
//                    return;
//                }

//                if(kpValue > 8) kpValue = 8;
//                if(kpValue < 1.5) kpValue = 1.5;

//                if(kiValue > 8) kiValue = 8;
//                if(kiValue < 1.5) kiValue = 1.5;

//                gimbalNetwork.montionCCommands.setMCTiltParams(kpValue, kiValue);
//                console.log("Tilt Params: KP = " + kpValue + " | KI = " + kiValue);
//            }
//        }
//    }

//    CustomizedTextInput {
//        id: tiltKP
//        x: 74
//        y: 151
//        width: 136
//        height: 40
//        inputTextMax: 12
//    }

//    CustomizedTextInput {
//        id: tiltKI
//        x: 302
//        y: 151
//        width: 136
//        height: 40
//        inputTextMax: 12
//    }

//    Text {
//        id: text2
//        x: 270
//        y: 55
//        width: 20
//        height: 20
//        color: "#ffffff"
//        text: qsTr("Ki")
//        font.pointSize: 13
//    }

//    Text {
//        id: text3
//        x: 36
//        y: 55
//        width: 20
//        height: 20
//        color: "#ffffff"
//        text: qsTr("Kp")
//        font.pointSize: 13
//    }

//    Text {
//        id: text4
//        x: 270
//        y: 161
//        width: 20
//        height: 20
//        color: "#ffffff"
//        text: qsTr("Ki")
//        font.pointSize: 13
//    }

//    Text {
//        id: text5
//        x: 36
//        y: 161
//        width: 20
//        height: 20
//        color: "#ffffff"
//        text: qsTr("Kp")
//        font.pointSize: 13
//    }

//    Button {
//        id: btnCalibIMU
//        x: 134
//        y: 292
//        text: "Calib IMU"
//        onClicked: {
//            if(camState.isConnected && camState.isPingOk && gimbalNetwork.isGimbalConnected){
//                gimbalNetwork.systemCommands.calibIMU();
//                console.log("Calib IMU");
//            }
//        }
//    }

//    Button {
//        id: btnResetIMU
//        x: 320
//        y: 292
//        text: "Reset IMU"
//        onClicked: {
//            if(camState.isConnected && camState.isPingOk && gimbalNetwork.isGimbalConnected){
//                gimbalNetwork.systemCommands.resetIMU();
//                console.log("Reset IMU");
//            }
//        }
//    }
}
