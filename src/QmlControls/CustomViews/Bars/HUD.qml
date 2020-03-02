//------------------ Include QT libs ------------------------------------------
import QtQuick.Window 2.2
import QtQuick 2.6
import QtQuick.Controls 2.1
import QtQuick.Layouts 1.3
import QtGraphicalEffects 1.0
import QtQuick 2.0

//------------------ Include Custom modules/plugins
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0
import CustomViews.HUD 1.0

//---------------- Component definition ---------------------------------------
Item {
    id: rootItem
    width: UIConstants.sRect*12
    height: UIConstants.sRect*6
    Rectangle{
        anchors.fill: parent
        color: UIConstants.transparentBlue
        radius: parent.height/2
        border.color: "gray"
        border.width: 1
    }

//    Canvas{
//        anchors.fill: parent
//        onPaint: {
//            var ctx = getContext("2d");
//            var drawColor = UIConstants.bgAppColor;
////            ctx.strokeStyle = drawColor;
//            ctx.fillStyle = drawColor;
//            ctx.lineWidth = 2
//            ctx.beginPath();
////            ctx.moveTo(height/2,height/2);
//            ctx.arc(height/2,height/2,height/2,0,2*Math.PI,false);
////            ctx.lineTo(height/2,height/2);
//            ctx.fill();
//            ctx.arc(width-height/2,height/2,height/2,0,2*Math.PI);
//            ctx.fillRect(height/2,0,width-height,height);
//            ctx.fill()
//        }
//    }
    QGCAttitudeWidget{
        id: rectAHRS
//        height: parent.height-10
        anchors.left: parent.left
        anchors.leftMargin: 5
        anchors.verticalCenter: parent.verticalCenter
//        width: parent.width / 2-10
        size: parent.width / 2-10
    }
    QGCCompassWidget{
        id: rectNavigator
        x: 100
        height: parent.height-10
        anchors.right: parent.right
        anchors.rightMargin: 5
        anchors.verticalCenterOffset: 0
        anchors.verticalCenter: parent.verticalCenter
        width: parent.width / 2-10
//        source: "qrc:/assets/images/navigator.png"
    }
}
