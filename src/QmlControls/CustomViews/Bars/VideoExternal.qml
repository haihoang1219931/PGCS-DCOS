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
ApplicationWindow{
    id: rootItem
    width: UIConstants.sRect*19
    height: UIConstants.sRect*13
    visible: true
    title: qsTr("Video export")
    property var player: undefined
    property var camState: undefined
    property int viewerID: -1
    Rectangle{
        anchors.fill: parent
        color: "black"
    }

    VideoRender{
        id: videoOutput
        anchors.fill: parent
    }
    MouseArea{
        id: mouse
        hoverEnabled: true
        anchors.fill: parent
        onDoubleClicked: {
            if(rootItem.visibility !== ApplicationWindow.FullScreen){
                rootItem.visibility = ApplicationWindow.FullScreen
            }else{
                rootItem.visibility = ApplicationWindow.Windowed
                rootItem.width = UIConstants.sRect*19
                rootItem.height = UIConstants.sRect*13
            }
        }
    }
    onPlayerChanged: {
        if(player !== undefined){
            viewerID = player.addVideoRender(videoOutput);
        }
    }
    onVisibleChanged: {
        if(!visible && camState !== undefined){
            camState.gcsExportVideo = false;
        }
        if(!visible && player !== undefined){
            player.removeVideoRender(viewerID);
        }
    }
}
