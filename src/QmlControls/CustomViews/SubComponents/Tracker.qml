import QtQuick 2.0
import QtLocation 5.9
import QtPositioning 5.5
import QtQuick.Window 2.0
import io.qdt.dev 1.0
import CustomViews.UIConstants 1.0

MapQuickItem {
    id: _tracker

    property real angle: 0

    readonly property int  widthSymbol: 56
    readonly property int  heighSymbol: 56

    readonly property color roundColor: "gray"
    readonly property string tracker_icon_Source: "qrc:/qmlimages/uavIcons/Unknown.png"

    anchorPoint.x: _rec_symbol.width/2
    anchorPoint.y: _rec_symbol.height/2

//  coordinate: position
    sourceItem: Rectangle {
        id: _rec_symbol
        opacity: 0.85
        width: widthSymbol
        height: heighSymbol
        radius: width/2

        color: "transparent"

        Component.onCompleted:
        {

        }

        Text
        {
            id:_tracker_text
            anchors.horizontalCenter: _rec_symbol.horizontalCenter
            y:56
            text: qsTr("Tracker")
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment:  Text.AlignVCenter
            font.weight: Font.Medium
            font.family: "Arial"
            font.pointSize: 11
            color: "white"
        }

        Image
        {
            id:_icon
            width: widthSymbol
            height: heighSymbol
            anchors.horizontalCenter: _rec_symbol.horizontalCenter
            anchors.verticalCenter: _rec_symbol.verticalCenter
            source: tracker_icon_Source
            rotation: angle
            opacity: 0.8
        }
        Rectangle
        {
            id: _rounded_tracker
            anchors.verticalCenter:  _rec_symbol.verticalCenter
            anchors.horizontalCenter: _rec_symbol.horizontalCenter
            color: "transparent"
            width: widthSymbol
            height: heighSymbol
            radius: width/2
            border.color: roundColor
            border.width: 2
            opacity: 0.8
        }

    }

}

