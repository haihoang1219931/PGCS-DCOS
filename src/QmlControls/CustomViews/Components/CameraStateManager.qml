import QtQuick 2.0

Item {
    id: root
    property int countAutoConnectTimes: 0
    property bool isJoystickConnected: false
    property bool isConnected: true
    property bool isPingOk: true
    property bool objectLocalization: false
    property string camIP: ""
    property string joyID: ""
    property int defaultWidth: 1366
    property int defaultHeight: 768
    property int defaultCoordinateWidth: defaultWidth/3
    property int defaultCoordinateHeight: defaultHeight/5
    property string layout: "multipanes"
    property string singlePane: "video"
    property bool fullScreen: false
    property color disable: "#B0808080"
    property color rightBarBackground: "#80171111"
    property color subFunctionColor: "#4d4d4d"
    property color themeColor: "#222F3E" //"#222F3E","#1A1A1A"
    property bool ctrlPress: false
    property bool autoConnect: false
    property string fd_icon: "EN"
    property var language:
    {
        "EN":0,
        "VI":1
    }
    property real zoomState: 0
    property int irZoom: 1
    property string sensorID: sensorIDEO
    property string sensorIDEO: "EO"
    property string sensorIDIR: "IR"
    property string lockMode: "FREE" // "VISUAL","GEO","TRACK"
    property string presetMode: "OFF" // "RIGHT_WING","NADIR","OFF"
    property string colorModeEO: "AUTO"
    property string colorModeIR: "WHITE_HOT"
//    property string colorMode: "AUTO"
    property var colorMode: {
                            "EO":"AUTO",// "COLOR","DAWN"
                            "IR":"WHITE_HOT" // "BLACK_HOT","REDDISH","COLOR"
                            }
    property string defogModeEO: "OFF"
    property string defogModeIR: "OFF"
    property var defogMode: {
                            "EO":"OFF",//"AUTO","LOW","MEDIUM","HIGH"
                            "IR":"OFF" //"AUTO","LOW","MEDIUM","HIGH"
                            }
    property string gimbalMode: "NA" //"OFF" "ON","SECURE","SLEEP"
    property bool gimbalRecord: false // true
    property real panPos: 0
    property real tiltPos: 0
    property real alphaSpeed: 3
    property real hfov: 63
    property real vfov: 27
    property int invertPan: 1
    property int invertTilt: 1
    property real latitude: 21.120457
    property real longitude: 105.120457
    property real altitude: 400.12
    property real roll: 9.12
    property real pitch: 0.12
    property real yaw: 369.12
    property bool isGPSValid: (latitude !== 0 && longitude !== 0) &&
                              (latitude > -90 && latitude < 90) &&
                              (longitude > -180 && longitude < 180)
    property bool digitalStab: true
    property bool gimbalStab: false
    property bool gcsStab: true
    property bool gcsRecord: true
    property bool gcsOD: false
    property bool gcsPD: false
    property bool gcsExportVideo: false
    property bool gcsShare: true
    property bool objDetect: false
    // hud of gcs
    property bool hubVisible: true
    property bool infoVisible: false
    property bool planeRSSIVisible: false
    property bool gdtRSSIVisible: false
    property bool flightModeVisible: false
    property bool zoomVisible: true
    property bool centerVisible: true
    property bool navigatorVisible: false
    property bool targetVisible: false
    // hub of camera
    property bool enableGimbalOverlay: true
    property bool enableLaserDevice: false
    property bool enableLimitWarning: false
    property bool enableGyroStabilization: false
    property bool enableGimbalMode: false
    property bool enableTrackingBoxes: true
    property bool enableHFOV: false
    property bool enableSlantRange: false
    property bool enableTargetLocation: false
    property bool enableTimestamp: false
    property bool enableCrosshair: false
    // track params
    property int trackSize: 50
    property int nextTrackSize: 50
    property int dTrackSize: 5
    property int trackSizeMax: 255
    property int trackSizeMin: 8
    property int stabBackground: 1
    property bool hudGcsEnable: true
    property bool hudCameraEnable: true
    signal lanChanged()
    function changeLanguage(newLanguage){
        if(fd_icon !== newLanguage){
            fd_icon = newLanguage
            lanChanged()
        }
    }
    function updateTrackSize(size){
        trackSize = size
    }

    function changeTrackParams(type){
        var sendCmd = false;
        switch(type){
        case "size_up":
            if(trackSize+dTrackSize <= trackSizeMax){
                nextTrackSize=trackSize + dTrackSize;
                sendCmd = true;
            }
            break;
        case "size_down":
            if(trackSize-dTrackSize >= trackSizeMin){
                nextTrackSize = trackSize - dTrackSize;
                sendCmd = true;
            }
            break;
        case "size_max":
            if(trackSize < trackSizeMax){
                nextTrackSize = trackSizeMax;
                sendCmd = true;
            }
            break;
        case "size_min":
            if(trackSize > trackSizeMin){
                nextTrackSize = trackSizeMin;
                sendCmd = true;
            }
            break;
        }

        return sendCmd;
    }
    function changeSensorColor(sensor,color){
        var lastColor;
        if(sensor === "EO"){
            lastColor = colorMode["IR"];
            colorMode = {"EO":color,"IR":lastColor}
        }else if(sensor === "IR"){
            lastColor = colorMode["EO"];
            colorMode = {"EO":lastColor,"IR":color}
        }
//        sensorID = sensorID;
        colorModeEO = colorModeEO;
        colorModeIR = colorModeIR;
//        console.log("change "+sensor+" to "+color)
    }
    function changeLockMode(lockMode){
        root.lockMode = lockMode
//        console.log("changeLockMode to "+lockMode)
    }
    function changePresetMode(presetMode){
        root.presetMode = presetMode
    }
    function changeSensorDefog(sensor,defog){
        var lastDefog;
        if(sensor === "EO"){
            lastDefog = defogMode["IR"];
            defogMode = {"EO":defog,"IR":lastDefog}
        }else if(sensor === "IR"){
            lastDefog = defogMode["EO"];
            defogMode = {"EO":lastDefog,"IR":defog}
        }
//        console.log("change defog "+sensor+" to "+defog)
//        defogMode[sensor] = defog
//        sensorID = sensorID;
    }
    function changeGimbalMode(gimbalMode){
        root.gimbalMode = gimbalMode

    }
    function changeRecordMode(recordMode){
        root.recordMode = recordMode
    }
    function changeGimbalStab(enable){
        gimbalStab = enable;
    }
    function changeDigitalStab(enable){
        digitalStab = enable;
    }
    function changeGcsStab(enable){
        gcsStab = enable;
    }
    function changeObjectDetect(enable){
        objDetect = enable;
    }
    function changeHubVisible(enable){
        hubVisible = enable;
    }
    function changeSensorMode(mode){
        sensorID = mode;
    }

    onGimbalModeChanged: {
        console.log("changeGimbalMode to"+gimbalMode)
    }
}
