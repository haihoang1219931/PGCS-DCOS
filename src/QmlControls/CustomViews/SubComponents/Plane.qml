import QtQuick 2.0
import QtLocation 5.9
import QtPositioning 5.5
import QtQuick.Window 2.0
import CustomViews.UIConstants 1.0

MapQuickItem {
    id: gcsPlane
    property string planeSource: "qrc:/qmlimages/uavIcons/Unknown.png"
    rotation: 0
    anchorPoint.x: width / 2
    anchorPoint.y: height / 2
    width: UIConstants.sRect * 2
    height: UIConstants.sRect * 2
    sourceItem: Image
    {
        width: gcsPlane.width
        height: gcsPlane.height
        id:imagePlane
        source: planeSource
    }

}

