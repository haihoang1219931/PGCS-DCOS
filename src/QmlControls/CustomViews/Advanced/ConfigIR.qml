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
    property bool bold: false
    property color textColor: "white"
//    CustomizedLongButton{
//        id: btnFFC
//        x: 8
//        y: 8
//        swithState: false
//        width: 286
//        height: 50
//        text: "FFC"
//        iconNormal: "qrc:/GUI/svgs/solid/low-vision.svg"
//        onClicked: {
//            console.log("FFC to "+enable);
//            if(camState.isConnected && camState.isPingOk && gimbalNetwork.isGimbalConnected){
//                gimbalNetwork.irCommand.setFFCMode(enable);
//            }
//        }
//    }
//    CustomizedAjustmentButton{
//        id: btnAdjustmentColor
//        x: 8
//        y: 64
//        width: 286
//        height: 100
//        name: "Color"
//        listValue: ["White hot","Black hot","Fusion","Rain"]
//        onValueIDChanged: {
//            var cmd = "Whitehot";
//            switch(valueID){
//                case 0:
//                    cmd = "Whitehot";
//                    break;
//                case 1:
//                    cmd = "Blackhot";
//                    break;
//                case 2:
//                    cmd = "Fusion";
//                    break;
//                case 3:
//                    cmd = "Rain";
//                    break;
//            }
//            console.log("IR palette = "+cmd);
//            if(camState.isConnected && camState.isPingOk && gimbalNetwork.isGimbalConnected){
//                gimbalNetwork.irCommand.setIRPalette(cmd);
//            }
//        }
//    }
//    CustomizedAjustmentButton{
//        id: btnAdjustmentZoom
//        x: 8
//        y: 170
//        width: 286
//        height: 100
//        name: "Zoom"
//        listValue: ["1x","2x","4x","8x"]
//        onValueIDChanged: {
//            var cmd = 1;
//            switch(valueID){
//                case 0:
//                    cmd = 1;
//                    break;
//                case 1:
//                    cmd = 3;
//                    break;
//                case 2:
//                    cmd = 5;
//                    break;
//                case 3:
//                    cmd = 7;
//                    break;
//            }
//            console.log("IR Zoom = "+cmd);
//            if(camState.isConnected && camState.isPingOk && gimbalNetwork.isGimbalConnected){
//                gimbalNetwork.irCommand.setIRZoom(cmd);
//            }
//        }
//    }

//    CustomizedAjustmentButton {
//        id: btnAdjustmentMWIRTempPresets
//        name: "MWIR Temp Presets"
//        x: 8
//        y: 276
//        width: 286
//        height: 100
//        listValue: ["HOT","MEDIUM","COLD"]
//        onValueIDChanged: {
//            var cmd = listValue[valueID];
////            switch(valueID){
////                case 0:
////                    cmd = "HOT";
////                    break;
////                case 1:
////                    cmd = "MEIDUM";
////                    break;
////                case 2:
////                    cmd = "COLD";
////                    break;
////            }
//            console.log("MWIR Temp Presets = "+cmd);
//            if(camState.isConnected && camState.isPingOk && gimbalNetwork.isGimbalConnected){
//                gimbalNetwork.irCommand.setMWIRTempPreset(cmd);
//            }
//        }
//    }
}
