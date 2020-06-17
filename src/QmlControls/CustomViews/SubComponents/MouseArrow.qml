import QtQuick 2.0
import QtLocation 5.9
import QtPositioning 5.5
import QtQuick.Window 2.0

MapQuickItem {
    id: gcs_waypoint
    anchorPoint.x: drawingCanvas.width/2
    anchorPoint.y: drawingCanvas.height/2
//    coordinate: position

    sourceItem: Canvas
    {
        id: drawingCanvas
        width:16
        height:16
        onPaint:
        {
            var ctx = getContext("2d")

            ctx.lineWidth = 4;
            ctx.strokeStyle = "red"
            ctx.beginPath()
            ctx.moveTo(0, 0)
            ctx.lineTo(drawingCanvas.width, drawingCanvas.height)
            ctx.moveTo(drawingCanvas.width, 0)
            ctx.lineTo(0, drawingCanvas.height)
            //ctx.closePath()
            ctx.stroke()

        }
    }
}
