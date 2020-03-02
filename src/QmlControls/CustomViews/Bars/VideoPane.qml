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
    property bool isVideoOn: false
    property var player
    function searchByClass(selectedList){
        player.searchByClass(selectedList)
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
        text: rootItem.width > 200?"NO VIDEO":"NO\nVIDEO"
        verticalAlignment: Text.AlignVCenter
        horizontalAlignment: Text.AlignHCenter
        color: UIConstants.textColor
        font.family: UIConstants.appFont
        font.bold: true
        font.pixelSize: 20
        visible: !rootItem.isVideoOn
    }

    Connections{
        target: player
        onDeterminedTrackObjected: {
            gimbalNetwork.ipcCommands.setClickPoint(_id, _px, _py, _w, _h, _oW, _oH);
        }

        onDeterminedPlateOnTracking: {
            listPlate.add(_imgPath, _plateID);
        }
    }
    VideoOutput {
        id: videoOutput
        anchors.fill: parent
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.verticalCenter: parent.verticalCenter
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
        timer.start();
    }

    Timer{
        id: timer
        interval: 3000
        repeat: false
        running: false
        property bool eo: true
        onTriggered: {
//            eo =! eo;
            if(VIDEO_DECODER || GPU_PROCESS){
                var component = Qt.createQmlObject('import io.qdt.dev 1.0;
                                        Player {
                                            id: player
                                            enStream: true
                                            enSaving: true
                                            sensorMode: 0
                                        }',
                                                   rootItem,
                                                   "dynamicSnipet1");
                rootItem.player = component;
                console.log("Player component="+rootItem.player);
                videoOutput.source = rootItem.player;
            }
            if(VIDEO_DECODER){
                if(eo){
                    player.setVideo(
                                "rtspsrc location=rtsp://192.168.0.103/z3-1.sdp latency=150 ! rtph265depay ! h265parse ! avdec_h265 ! "+
                                "appsink name=mysink sync=true async=true");
                    player.start()
                }
            }else if(GPU_PROCESS){
                var config = PCSConfig.getData();
                player.setVideoSource("232.4.130.146",
                                      parseInt(18888));
                player.play()
            }

        }
    }

}

/*##^## Designer {
    D{i:7;anchors_height:192;anchors_y:37}
}
 ##^##*/
