import QtQuick 2.0
import QtLocation 5.9
import QtPositioning 5.5
import QtQuick.Window 2.0


MapPolyline {
    id: gcs_plane_trajactory
    property string color :"#228888"
    property int linewidth : 2

    line.width: linewidth
    line.color: color
    opacity: 0.8
    smooth:true
    antialiasing: true

}

//import QtQuick 2.0
//import QtLocation 5.9
//import QtPositioning 5.5
//import QtQuick.Window 2.0

//MapQuickItem {
//        id: gcs_trajactory
//        property var x1
//        property var x2
//        property var y1
//        property var y2

//        anchorPoint.x: 0
//        anchorPoint.y: 0


////    coordinate: position
//    sourceItem: Canvas {
//        id: canvas_trajatory
//        width: 400
//        height: 400
//        opacity: 0.85
//        onPaint: {
//            // Get drawing context
//             var context = getContext("2d");
//             // Draw a line
//             context.beginPath();
//             context.lineWidth = 4;
//             context.moveTo(x1, y1);
//             context.strokeStyle = "#228888"
//             context.lineTo(x2, y2);
//             context.stroke();

//                }

//    }
//}



