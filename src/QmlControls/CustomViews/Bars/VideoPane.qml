//------------------ Include QT libs ------------------------------------------
import QtQuick.Window 2.2
import QtQuick 2.6
import QtQuick.Controls 2.1
import QtQuick.Layouts 1.3
import QtMultimedia 5.5
import QtQuick 2.0

//------------------ Include Custom modules/plugins
import CustomViews.Components   1.0
import CustomViews.UIConstants  1.0
//import QGroundControl 1.0
import io.qdt.dev               1.0
//---------------- Component definition ---------------------------------------
Item{
    id: rootItem
    width: UIConstants.sRect*19
    height: UIConstants.sRect*13   
    clip: true
    property bool isVideoOn: false
    property var player: undefined
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
    VideoRender{
        id: videoOutput
        anchors.fill: parent
    }

    MouseArea{
        id: mouse
        hoverEnabled: true
        anchors.fill: parent
        onClicked: {
            if(player !== undefined)
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
            value: 0.5
            onValueChanged: {
                var newTrackSize = value*parseInt(lblMax.text);
                cameraController.gimbal.changeTrackSize(newTrackSize);
            }
        }

        Label {
            id: lblMin
            width: 40
            height: 17
            text: qsTr("0")
            color: UIConstants.textColor
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
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
            text: qsTr("400")
            color: UIConstants.textColor
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
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

            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            color: UIConstants.textColor
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
            y: (slider.value)*(slider.height-20)+slider.y
            text: Number(slider.value*parseFloat(lblMax.text)).toFixed(0).toString()
        }
    }
    Component.onCompleted: {
        if(USE_VIDEO_GPU || USE_VIDEO_CPU){
            rootItem.player = cameraController.videoEngine;
            rootItem.player.plateLog = listPlateLog.plateLog;
            videoOutput.anchors.fill = rootItem;
            cameraController.videoEngine.addVideoRender(videoOutput);
        }
    }
}
