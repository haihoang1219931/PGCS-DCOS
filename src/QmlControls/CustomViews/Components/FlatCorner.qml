import QtQuick 2.0

import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0
Item {
    width: 20
    height: 20
    property int lineVertical: width/2
    property int lineHorizontal: height/2
    Canvas{
        id: canvas
        anchors.fill: parent
        onPaint: {
            var ctx = getContext("2d");
            ctx.reset();
            var drawColor = UIConstants.textColor;
//            var drawColor = "black";
            ctx.strokeStyle = drawColor;
            ctx.lineWidth = 4;
            ctx.moveTo(0,height);
            ctx.lineTo(0,lineHorizontal);
            ctx.lineWidth = 2;
            ctx.lineTo(lineVertical,0);
            ctx.lineTo(width,0);
            ctx.stroke();
        }
    }
    onScaleChanged: {
        canvas.requestPaint();
    }
}
