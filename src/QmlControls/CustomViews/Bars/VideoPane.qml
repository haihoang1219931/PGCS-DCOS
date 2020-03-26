//------------------ Include QT libs ------------------------------------------
import QtQuick.Window 2.2
import QtQuick 2.6
import QtQuick.Controls 2.1
import QtQuick.Layouts 1.3
import QtGraphicalEffects 1.0
import QtMultimedia 5.5
import QtQuick 2.0

//------------------ Include Custom modules/plugins
import CustomViews.Components   1.0
import CustomViews.UIConstants  1.0
//import QGroundControl 1.0
import io.qdt.dev               1.0
//---------------- Component definition ---------------------------------------
Flickable{
    id: rootItem
    width: UIConstants.sRect*19
    height: UIConstants.sRect*13   
    clip: true
    property var camState
    property real iPan: 0.0
    property real cPan: 0.0
    property real dPanOld: 0.0
    property real panRate: 0.0
    property real uPan: 0.0

    property real kpPan: 45.0
    property real kiPan: 1.0
    property real kdPan: 0.05

    property real iTilt: 0.0
    property real cTilt: 0.0
    property real dTiltOld: 0.0
    property real tiltRate: 0.0
    property real uTilt: 0.0

    property real kpTilt: 50.0
    property real kiTilt: 5.0
    property real kdTilt: 0.05
    property bool isVideoOn: false
    property var player
    function searchByClass(selectedList){
        player.searchByClass(selectedList)
    }
    function resetTrackParam(){
        iPan = 0.0
        cPan = 0.0
        dPanOld = 0.0
        panRate = 0.0
        uPan = 0.0

        iTilt = 0.0
        cTilt = 0.0
        dTiltOld = 0.0
        tiltRate = 0.0
        uTilt = 0.0
        console.log("\nreset track")
    }
    Rectangle{
        anchors.fill: parent
        color: "black"
        border.color: UIConstants.grayColor
        border.width: 1
        radius: UIConstants.rectRadius
    }

    Label{
        anchors.verticalCenter: parent.verticalCenter
        anchors.horizontalCenter: parent.horizontalCenter
        text: rootItem.width > UIConstants.sRect * 8?"NO VIDEO":"NO\nVIDEO"
        verticalAlignment: Text.AlignVCenter
        horizontalAlignment: Text.AlignHCenter
        color: UIConstants.textColor
        font.family: UIConstants.appFont
        font.bold: true
        font.pixelSize: UIConstants.fontSize * 2
        visible: !rootItem.isVideoOn
    }
    Connections{
        target: player

    }
    VideoOutput {
        id: videoOutput
        anchors.fill: parent
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.verticalCenter: parent.verticalCenter
        source: player
    }

    MouseArea{
        id: mouse
        hoverEnabled: true
        anchors.fill: parent
        onClicked: {
            player.setTrackAt(player.frameID , mouseX, mouseY, width, height)
            //            console.log("Clicked New Track " + player.frameID + " - [" +mouseX+", "+mouseY+", "+width+", "+height+"]");
        }


    }
    Item{
        id: item1
        anchors.fill: parent
        Slider {
            id: slider
            x: 8
            width: 40
            anchors.top: lblMin.bottom
            anchors.topMargin: 12
            anchors.bottom: lblMax.top
            anchors.bottomMargin: 6
            to: 0
            from: 1
            orientation: Slider.SnapOnRelease
            value: 0.1
            onValueChanged: {
                var newTrackSize = parseInt(value*256);
                if(camState.isConnected && camState.isPingOk && gimbalNetwork.isGimbalConnected){
                    gimbalNetwork.ipcCommands.changeTrackSize(newTrackSize);
                }
            }
        }

        Label {
            id: lblMin
            width: 40
            height: 17
            color: "#ffffff"
            text: qsTr("0")
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            anchors.left: parent.left
            anchors.leftMargin: 8
            anchors.top: parent.top
            anchors.topMargin: 37
        }

        Label {
            id: lblMax
            x: 2
            y: 212
            width: 40
            height: 17
            color: "#ffffff"
            text: qsTr("256")
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 31
            anchors.leftMargin: 8
            anchors.left: parent.left
        }

        Label {
            id: label
            anchors.left: parent.left
            anchors.leftMargin: 45
            y: (slider.value)*(slider.height-20)+slider.y
            color: "#ffffff"
            text: Number(slider.value*256).toFixed(0).toString()
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter

        }
    }
    Component.onCompleted: {
        if(USE_VIDEO_GPU || USE_VIDEO_CPU){
            var component = Qt.createQmlObject('import io.qdt.dev 1.0;
                                    Player {
                                    }',
                                               rootItem,
                                               "dynamicSnipet1");
            rootItem.player = component;
            rootItem.player.plateLog = listPlateLog.plateLog;
            rootItem.player.determinedTrackObjected.connect( function (_id,_px,_py,_oW,_oH,_w,_h){
    //            console.log("ObjectTrack==> [" + _id + ", " + _px + ", " + _py + ", " + _w + ", " + _h + ", " + _oW + ", " + _oH + "]" + "["+ camState.hfov +"]");
                var px = (_px + _oW/2) - _w/2
                var py = (_py + _oH/2) - _h/2
                var hfov = camState.hfov;

                if(hfov > 0.0963){
                    var focalLength = _w / 2 / Math.tan(hfov/2)

                    var deltaPan = Math.atan(-px/focalLength) * 180.0 / Math.PI
        //            if(deltaPan > 10)deltaPan = 10
        //            else if(deltaPan < -10)deltaPan = -10
                    iPan+=deltaPan/30.0
                    cPan+=(panRate - uPan)/30.0
                    var dPan = (deltaPan - dPanOld) * 30
                    uPan = kpPan * deltaPan + kiPan * iPan + kdPan * dPan
                    dPanOld = deltaPan

                    if(uPan > 1023){
                        console.log("\n limit pan")
                        panRate = 1023
                    }
                    else if (uPan < -1023){
                        console.log("\n limit pan")
                        panRate = -1023
                    }
                    else panRate = uPan

                    var deltaTilt = Math.atan(-py/focalLength) * 180.0 / Math.PI
        //            if(deltaTilt > 10)deltaTilt = 10
        //            else if(deltaTilt < -10)deltaTilt = -10
                    iTilt+=deltaTilt/30.0
                    cTilt+=(tiltRate - uTilt)/30.0
                    var dTilt = (deltaTilt - dTiltOld) * 30
                    uTilt = kpTilt * deltaTilt + kiTilt * iTilt + kdTilt * dTilt
                    dTiltOld = deltaTilt

                    if(uTilt > 1023){
                        console.log("\n limit tilt")
                        tiltRate = 1023
                    }
                    else if (uTilt < -1023){
                        console.log("\n limit tilt")
                        tiltRate = -1023
                    }
                    else tiltRate = uTilt

                    if(gimbalNetwork.isGimbalConnected){
                        gimbalNetwork.ipcCommands.gimbalControl(0, panRate, tiltRate);
        //                console.log("rate ===> " + px + " | " + py);
                    }
                }
            });
            rootItem.player.objectLost.connect(function () {
                if(gimbalNetwork.isGimbalConnected){
                    camState.changeLockMode("FREE");
                    hud.changeLockMode("LOCK_FREE");
                    gimbalNetwork.ipcCommands.gimbalControl(0, 0, 0);
                }
            })

            rootItem.player.determinedPlateOnTracking.connect(function (_imgPath,_plateID){
                listPlate.add(_imgPath, _plateID);
            })
            console.log("Player component="+rootItem.player);
            videoOutput.source = rootItem.player;
        }
    }
}
