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
    property int minValue: 0
    property int maxValue: 400
    property int currentValue: 200
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
    Column{
        width: UIConstants.sRect * 2
        anchors.verticalCenter: parent.verticalCenter
        anchors.left: parent.left
        anchors.leftMargin: 8
        anchors.bottom: parent.bottom
        anchors.bottomMargin: UIConstants.sRect*2
        Label {
            id: lblMax
            width: UIConstants.sRect*2
            height: UIConstants.sRect
            text: maxValue
            color: UIConstants.textColor
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
        }

        Slider{
            id: sldBar
            orientation: Qt.Vertical
            width: UIConstants.sRect*2
            height: parent.height - lblMax.height - lblMin.height
            from: minValue
            to: maxValue
            value: currentValue
            handle: Rectangle{
                y: sldBar.bottomPadding + sldBar.visualPosition * (sldBar.availableHeight - height)
                x: sldBar.leftPadding + sldBar.availableWidth / 2 - width / 2
                width: sldBar.width
                height: width
                color: !sldBar.pressed ?
                    UIConstants.bgAppColor : UIConstants.textBlueColor
                radius: UIConstants.rectRadius
                border.color: UIConstants.grayColor
                border.width: 1
                Label{
                    anchors.verticalCenter: parent.verticalCenter
                    anchors.horizontalCenter: parent.horizontalCenter
                    verticalAlignment: Label.AlignVCenter
                    horizontalAlignment: Label.AlignHCenter
                    text: Number(sldBar.value).toFixed(0)
                    font.family: UIConstants.appFont
                    font.pixelSize: UIConstants.fontSize
                    color: UIConstants.textColor
                }
            }

            onValueChanged: {
                var newTrackSize = parseInt(value);
                CameraController.gimbal.changeTrackSize(newTrackSize);
            }
        }

        Label {
            id: lblMin
            width: UIConstants.sRect*2
            height: UIConstants.sRect
            text: minValue
            color: UIConstants.textColor
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
        }
    }
    Component.onCompleted: {
        if(USE_VIDEO_GPU || USE_VIDEO_CPU){
            rootItem.player = CameraController.videoEngine;
            rootItem.player.plateLog = listPlateLog.plateLog;
            videoOutput.anchors.fill = rootItem;
            CameraController.videoEngine.addVideoRender(videoOutput);
        }
    }
}
