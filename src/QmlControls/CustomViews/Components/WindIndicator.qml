import QtQuick.Window 2.2
import QtQuick 2.6
import QtQuick.Controls 2.1
import QtQuick.Layouts 1.3
import QtGraphicalEffects 1.0
import QtQuick 2.0

import CustomViews.UIConstants 1.0

Item {
    property real windHeading: FlightVehicle.link ? FlightVehicle.windHeading:0 //deg
    property real windSpeed: FlightVehicle.link ? FlightVehicle.windSpeed * 3.6 :0 //km/h
    width: UIConstants.sRect*3
    height: UIConstants.sRect*3
    Rectangle{
        id: rect_background
        anchors.fill: parent
        color: UIConstants.transparentBlue
        radius: width / 2
        border.color: "gray"
        border.width: 1
//        opacity: 0.7
        rotation: FlightVehicle.link ? FlightVehicle.windHeading:0
        Canvas{
            id: arrowWind
            anchors.fill: parent
            clip:true
            opacity: 0.6
            onPaint: {
                var ctx = getContext("2d")
                ctx.fillStyle = UIConstants.navIconColor;
    //            ctx.strokeStyle = "white"
                //ctx.fillRect(0,0,width,height)
                ctx.lineWidth = 1
                ctx.beginPath();
                ctx.moveTo(width/2,0 + height /8);//1
                ctx.lineTo(width * 3/4 - width/30, height/4 + height /8);//2
                ctx.lineTo(width *5/8 - width/20,height/4 + height /8);//3
                ctx.lineTo(width *5/8 - width/20,height*3/4 + height /8);
                ctx.lineTo(width *3/8 + width/20,height*3/4 + height /8);
                ctx.lineTo(width *3/8 + width/20,height/4 + height /8);
                ctx.lineTo(width /4 + width/30,height/4 + height /8);
                ctx.lineTo(width/2,+ height /8);
                ctx.closePath();
    //            ctx.stroke();
                ctx.fill();
            }

        }
    }

    Label{
        id: lbWindHeading
        anchors.top:parent.top
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.topMargin: UIConstants.sRect * 0.75
        text:  parseInt(windHeading) + "°"
        font.family: "Arial"
        font.pixelSize: UIConstants.fontSize
        font.bold: true
        color: "white"
        opacity: 0.95
    }
    Label{
        id: lbWindSpeed
        anchors.bottom: parent.bottom
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottomMargin: UIConstants.sRect* 0.75
        text:  parseInt(windSpeed)+"kph"
        font.family: "Arial"
        font.pixelSize: UIConstants.fontSize
        font.bold: true
        color: "white"
        opacity: 0.95
    }

}
