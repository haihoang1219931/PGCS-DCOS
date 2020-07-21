import QtQuick 2.0
import QtLocation 5.9
import QtPositioning 5.5
import QtQuick.Window 2.0

MapQuickItem {
    id: gcsPlane
    property string planeSource: "qrc:/qmlimages/uavIcons/Unknown.png"
    rotation: 0
    anchorPoint.x: imagePlane.width / 2
    anchorPoint.y: imagePlane.height / 2
    sourceItem: Image
    {
        id:imagePlane
        source: planeSource
    }

}

