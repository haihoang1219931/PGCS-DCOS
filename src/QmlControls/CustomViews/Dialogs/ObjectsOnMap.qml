import QtQuick 2.9
import QtQuick.Window 2.2
import QtQuick.Controls 2.4
import QtQuick.Layouts 1.3
import QtWebEngine 1.7
import QtGraphicalEffects 1.0

import io.qdt.dev 1.0
import CustomViews.Components   1.0
import CustomViews.UIConstants  1.0
Item {
    id: root
    anchors.fill: parent
    property var player
    function updateObjectPosition(id, newScreenX, newScreenY){
        player.updateTrackObjectInfo("Object","SCREEN_X",newScreenX);
        player.updateTrackObjectInfo("Object","SCREEN_Y",newScreenY);
    }
    Repeater {
        id: listObjects
        model: player.listTrackObjectInfos
        delegate: Rectangle {
            id: object
            width: 30
            height: 30
            x: screenX
            y: screenY
            property string spreadColor: "#1abc9c"
            property string userId_: userId
            property double latitude_: latitude
            property double longitude_: longitude
            property string name_: name
            property double speed_: speed
            property double angle_: angle
            property int screenX_: 0
            property int screenY_: 0
            onLatitude_Changed: {
                mapPane.updateObjectsOnMap();
            }
            onLongitude_Changed: {
                mapPane.updateObjectsOnMap();
            }
            color: "#16a085"
            radius: width/2
            FlatIcon{
                anchors.fill: parent
                icon: UIConstants.iArrow
                rotation: angle_
            }
        }
    }
}
