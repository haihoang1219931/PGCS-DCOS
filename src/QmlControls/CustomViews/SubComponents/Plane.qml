import QtQuick 2.0
import QtLocation 5.9
import QtPositioning 5.5
import QtQuick.Window 2.0

MapQuickItem {
    id: gcs_Plane


    readonly property int  widthPlane: 75
    readonly property int  heighPlane: 75

    readonly property string planeSource: "qrc:/assets/images/icons/drone.png"
    property double heading: 0

    anchorPoint.x: _rec_plane.width/2
    anchorPoint.y: _rec_plane.height/2

//    coordinate: position
    sourceItem: Rectangle {
        id: _rec_plane
        width: widthPlane
        height: heighPlane
        radius: width/2

        color: "transparent"

        Image
        {
            id:_imagePlane
            source: planeSource
            width: widthPlane
            height: heighPlane
            transformOrigin: Item.Center
            rotation: heading
        }
    }

}

