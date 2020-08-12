import QtQuick 2.0
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0
Item {
    id: item1
    width: UIConstants.sRect * 7
    height: UIConstants.sRect * 5
    property int lineWidth: 10
    property int lineHeight: 10

    Canvas{
        id: canvas
        anchors.fill: parent
        anchors.margins: 4
        onPaint: {
            var ctx = getContext("2d");
            ctx.reset();
            var drawColor = UIConstants.transparentBlue;
            ctx.fillStyle = drawColor;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(0,lineHeight);
            ctx.lineTo(0,height - lineHeight);
            ctx.lineTo(lineWidth,height);
            ctx.lineTo(width-lineWidth,height);
            ctx.lineTo(width,height-lineHeight);
            ctx.lineTo(width,lineHeight);
            ctx.lineTo(width-lineWidth,0);
            ctx.lineTo(lineWidth,0);
            ctx.closePath();
            ctx.fill();
        }
    }
    FlatCorner{
        id: connerLeftUp
        anchors.top: parent.top
        anchors.left: parent.left
    }
    FlatCorner{
        id: connerRightUp
        anchors.top: parent.top
        anchors.right: parent.right
        rotation: 90
    }
    FlatCorner{
        id: connerRightDown
        anchors.bottom: parent.bottom
        anchors.right: parent.right
        rotation: 180
    }
    FlatCorner{
        id: connerLeftDown
        anchors.bottom: parent.bottom
        anchors.left: parent.left
        rotation: 270
    }
    onScaleChanged: {
        canvas.requestPaint();
    }
}

/*##^## Designer {
    D{i:3;anchors_x:0}
}
 ##^##*/
