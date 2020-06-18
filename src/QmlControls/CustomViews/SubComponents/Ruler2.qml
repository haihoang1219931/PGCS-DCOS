import QtQuick 2.0
import QtLocation 5.9
import QtPositioning 5.5
import QtQuick.Window 2.0


Item
{
    property variant coord1
    property variant coord2
    property string color :"#228888"
    property int linewidth : 2
    property var uavmap : null
    property var mlineOject: null
    property var mcircleOject: null
    property var marrowOject: null
    property var minforOject: null
    property var showText: true
    property var lineColor: "red"

    id:rulerItem
    anchors.fill: parent

    Component.onCompleted:{
        var heading = coord1.azimuthTo(coord2)
        var distance = coord1.distanceTo(coord2)
        var coordCenter = coord1.atDistanceAndAzimuth(distance/2,heading)
        mlineOject=creatLine(coord1,coord2)
        mcircleOject=createCircle(coord1)
        marrowOject=createArrow(coord2,heading)
        minforOject=createInfor(distance,coordCenter,heading)
    }

    Component
    {
        id:circleComponent
        MapQuickItem
            {
                anchorPoint.x: rectCircle.width/2
                anchorPoint.y: rectCircle.height/2
                sourceItem: Rectangle {
                    id: rectCircle
                    opacity: 1
                    width: 8
                    height: 8
                    radius: width/2
                    color: lineColor
                }
            }
    }

    Component
    {
        id:arrowComponent
        MapQuickItem
            {
                anchorPoint.x: canvasArrow.width/2
                anchorPoint.y: 0

                sourceItem: Canvas
                {
                    id: canvasArrow
                    width:10
                    height:10
                    onPaint:
                    {
                        var ctx = getContext("2d")
                        ctx.reset()
                        ctx.lineWidth = 1;
                        ctx.strokeStyle = lineColor
                        ctx.fillStyle = lineColor
                        ctx.beginPath()
                        ctx.moveTo(width/2, 0)
                        ctx.lineTo(width,height)
                        ctx.lineTo(0, height)
                        ctx.closePath()

                        ctx.fill()
                        //ctx.closePath()
                        ctx.stroke()
                    }
                }
            }
    }

    Component
    {
        id:textComponent
        MapQuickItem
            {
                property string text: "0.0 0m"
                property int widthRect: 80

                id:textQuickItem
                anchorPoint.x: rectText.width/2
                anchorPoint.y: 22
                visible: showText
                sourceItem: Rectangle{
                    height: 22
                    width: widthRect
                    color: "transparent"
                    Rectangle{
                        id: rectText
                        width: widthRect
                        height: 18
                        color: "white"
                        radius: 4

                        Text {
                            id: textInfor
                            anchors.fill: parent
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment:  Text.AlignVCenter
                            color:"red"
                            font.family: "Arial"
                            text: textQuickItem.text
                        }
                    }
                }
            }
    }

    function creatLine(coordinate1,coordinate2)  {
        var component = Qt.createComponent("qrc:/CustomViews/SubComponents/Trajactory.qml");
        var line = component.createObject(uavmap, {color:lineColor});
        if (line === null) {
            // Error Handling
            console.log("Error creating object");
            return;
        }
        else
        {
            line.addCoordinate(coordinate1)
            line.addCoordinate(coordinate2)
            uavmap.addMapItem(line)
            return line
        }
    }

    function createCircle(coord)
    {
        var circleObject = circleComponent.createObject(uavmap,{"coordinate":coord})
        if (circleObject === null) {
            console.log("Error creating object");
            return;
        }
        else
        {
            uavmap.addMapItem(circleObject)
            return circleObject
        }
    }

    function createArrow(coord,heading)
    {
        var arrowObject = arrowComponent.createObject(uavmap,{"coordinate":coord})
        if (arrowObject === null) {
            console.log("Error creating object");
            return;
        }
        else
        {
            arrowObject.transformOrigin = Item.Top
            arrowObject.rotation = heading
            uavmap.addMapItem(arrowObject)
            return arrowObject
        }
    }

    function createInfor(distance,coord,heading)
    {
        var textRotation = 0
        var textObject = textComponent.createObject(uavmap,{"coordinate":coord})
        if (textObject === null) {
            console.log("Error creating object");
            return;
        }
        else
        {
            if(heading < 180)
                textRotation = heading - 90
            else
                textRotation = heading + 90

            textObject.transformOrigin = Item.Bottom
            textObject.rotation = textRotation
            textObject.text = qsTr(heading.toFixed(1) + " " + distance.toFixed(0) + "m")
            textObject.widthRect = textObject.text.length * 8
            uavmap.addMapItem(textObject)
            return textObject
        }
    }


    function destroyChildOject()
    {
        mlineOject.destroy()
        mcircleOject.destroy()
        marrowOject.destroy()
        minforOject.destroy()
//        uavmap.removeMapItem(mlineOject)
//        uavmap.removeMapItem(mcircleOject)
//        uavmap.removeMapItem(marrowOject)
//        uavmap.removeMapItem(minforOject)
    }

}



