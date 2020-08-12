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
//        id: btnIRCutFilter
//        x: 8
//        y: 8
//        width: 286
//        height: 50
//        text: "IR Cut filter"
//        enable: false
//        iconNormal: "qrc:/GUI/svgs/solid/adjust.svg"
//        onClicked: {
//            console.log("IR Cut filter to "+enable);
//            if(camState.isConnected && camState.isPingOk && gimbalNetwork.isGimbalConnected){
//                gimbalNetwork.eoCommand.disableInfraredCutFilter(!enable);
//            }
//        }
//    }
//    CustomizedAjustmentButton{
//        id: btnAdjustmentDefog
//        x: 8
//        y: 64
//        width: 286
//        height: 100
//        name: "Defog"
//        listValue: ["Off","Auto","Low","Medium","High"]
//        onValueIDChanged: {
//            var cmd = "01";
//            switch(valueID){
//                case 0:
//                    cmd = "01";
//                    break;
//                case 1:
//                    cmd = "07";
//                    break;
//                case 2:
//                    cmd = "02";
//                    break;
//                case 3:
//                    cmd = "03";
//                    break;
//                case 4:
//                    cmd = "04";
//                    break;
//            }
//            console.log("Defog = "+cmd);
//            if(camState.isConnected && camState.isPingOk && gimbalNetwork.isGimbalConnected){
//                gimbalNetwork.ipcCommands.configCamera("3208",cmd);
//            }
//        }
//    }
//    CustomizedAjustmentButton{
//        id: btnAdjustmentZoom
//        x: 8
//        y: 170
//        width: 286
//        height: 100
//        name: "Optical Zoom"
//        listValue: ["1x","2x","3x","4x","5x","6x","7x","8x","9x","10x",
//                    "11x","12x","13x","14x","15x","16x","17x","18x","19x","20x",
//                    "21x","22x","23x","24x","25x","26x","27x","28x","29x","30x"]
//        onValueIDChanged: {
//            var zoomPosition = valueID+1;
//            console.log("EO change value ID to "+zoomPosition);
//            if(camState.isConnected && camState.isPingOk && gimbalNetwork.isGimbalConnected){
//                gimbalNetwork.eoCommand.setEOOpticalZoomPosition(zoomPosition);
//            }
//        }
//    }
//    CustomizedAjustmentButton{
//        id: btnAdjustmentAlpha
//        x: 300
//        y: 8
//        width: 292
//        height: 100
//        name: "Alpha"
//        listValue: ["1x","2x","3x","4x","5x","6x","7x","8x","9x","10x",
//                    "11x","12x","13x","14x","15x","16x","17x","18x","19x","20x",
//                    "21x","22x","23x","24x","25x","26x","27x","28x","29x","30x"]
//    }
//    CustomizedAjustmentButton{
//        id: btnAdjustmentBeta
//        x: 300
//        y: 114
//        width: 292
//        height: 100
//        name: "Beta"
//        listValue: ["1x","2x","3x","4x","5x","6x","7x","8x","9x","10x",
//            "11x","12x","13x","14x","15x","16x","17x","18x","19x","20x",
//                    "21x","22x","23x","24x","25x","26x","27x","28x","29x","30x"]
//    }
//    CustomizedAjustmentButton{
//        id: btnAdjustmentGama
//        x: 300
//        y: 220
//        width: 292
//        height: 100
//        name: "Gama"
//        listValue: ["1x","2x","3x","4x","5x","6x","7x","8x","9x","10x",
//            "11x","12x","13x","14x","15x","16x","17x","18x","19x","20x",
//            "21x","22x","23x","24x","25x","26x","27x","28x","29x","30x"]
//    }

//    CustomizedAjustmentButton {
//        id: btnAdjustmentZoomDigital
//        name: "Digital Zoom"
//        x: 8
//        y: 276
//        width: 286
//        height: 100
//        listValue: ["1x","2x","3x","4x","5x","6x","7x","8x","9x","10x",
//                        "11x","12x","13x","14x","15x","16x","17x","18x","19x","20x",
//                        "21x","22x","23x","24x","25x","26x","27x","28x","29x","30x"]
//        onValueIDChanged: {
//            var zoomPosition = valueID+1;
//            console.log("EO change value ID to "+zoomPosition);
//            if(camState.isConnected && camState.isPingOk && gimbalNetwork.isGimbalConnected){
//                gimbalNetwork.eoCommand.setEOOpticalZoomPosition(zoomPosition);
//            }
//        }
//    }
}
