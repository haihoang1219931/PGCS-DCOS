/**
 * ==============================================================================
 * @Project: FCS-Groundcontrol-based
 * @Module: PreflightCheck page
 * @Breif:
 * @Author: Hai Nguyen Hoang
 * @Date: 14/05/2019
 * @Language: QML
 * @License: (c) Viettel Aerospace Institude - Viettel Group
 * ============================================================================
 */

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
Rectangle{
    id: root
    width: 600
    height: 600
    color: "transparent"
    property color drawColor: Qt.rgba(0, 255, 0, 255)
    property real pan: 90
    property real tilt: 20
    property real hfov: 63.7
    property real vfov: 38.2
    property real dPan: 1
    property real dTilt: 1
    property real alpha: 0.2
    property string degreeSymbol : "\u00B0";
    property bool bold: false
    Label {
        id: lblTitle1
        height: 54
        wrapMode: Text.WordWrap
        anchors.right: parent.right
        anchors.rightMargin: 8
        anchors.left: parent.left
        anchors.top: parent.top
        anchors.leftMargin: 8
        anchors.topMargin: 0
        verticalAlignment: Text.AlignVCenter
        horizontalAlignment: Text.AlignHCenter
        color: UIConstants.textColor
        font.pixelSize: UIConstants.fontSize
        font.family: UIConstants.appFont
    }

    Rectangle {
        id: rectangle
        x: 120
        y: 543
        width: 520
        height: 49
        color: "#00000000"
        anchors.horizontalCenterOffset: 0
        anchors.horizontalCenter: parent.horizontalCenter

        Label {
            id: label
            y: 384
            height: 50
            text: qsTr("Does Payload working good?")
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 0
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignHCenter
            anchors.right: parent.right
            anchors.rightMargin: 0
            anchors.left: parent.left
            anchors.leftMargin: 0
            color: UIConstants.textColor
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
        }
    }

    Rectangle {
        id: rectangle1
        x: 200
        y: 111
        width: 584
        height: 390
        color: "#72ed80"
        anchors.horizontalCenter: parent.horizontalCenter

        Image {
            id: imgBackground
            anchors.fill: parent
            source: "qrc:/assets/images/hqdefault.jpg"
        }
        Rectangle {
            id: rectButtons
            y: 210
            width: 213
            height: 173
            color: "transparent"
            anchors.left: parent.left
            anchors.leftMargin: 8
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 8
            FlatButtonIcon{
                id: btnDown
                x: 70
                y: 107
                width: UIConstants.sRect*2
                height: UIConstants.sRect*2
                anchors.bottom: parent.bottom
                anchors.bottomMargin: 8
                anchors.horizontalCenterOffset: 0
                icon: UIConstants.iChevronDown
                isAutoReturn: true
                isShowRect: false
                isSolid: true
                anchors.horizontalCenter: parent.horizontalCenter
            }

            FlatButtonIcon{
                id: btnUp
                x: 70
                y: 8
                width: UIConstants.sRect*2
                height: UIConstants.sRect*2
                anchors.horizontalCenterOffset: 0
                icon: UIConstants.iChevronDown
                rotation: 180
                isAutoReturn: true
                isShowRect: false
                isSolid: true
                anchors.horizontalCenter: parent.horizontalCenter
            }

            FlatButtonIcon{
                id: btnRight
                x: 107
                y: 70
                width: UIConstants.sRect*2
                height: UIConstants.sRect*2
                anchors.right: parent.right
                anchors.rightMargin: 8
                anchors.verticalCenterOffset: 0
                icon: UIConstants.iChevronDown
                rotation: -90
                isAutoReturn: true
                isShowRect: false
                isSolid: true
                anchors.verticalCenter: parent.verticalCenter
            }

            FlatButtonIcon{
                id: btnLeft
                y: 70
                width: UIConstants.sRect*2
                height: UIConstants.sRect*2
                anchors.left: parent.left
                anchors.leftMargin: 8
                icon: UIConstants.iChevronDown
                rotation: 90
                isAutoReturn: true
                isShowRect: false
                isSolid: true
                anchors.verticalCenter: parent.verticalCenter
            }

            FlatButtonIcon{
                id: btnZoomIn
                y: 70
                width: UIConstants.sRect*2
                height: UIConstants.sRect*2
                anchors.verticalCenterOffset: 0
                anchors.left: btnLeft.right
                anchors.leftMargin: 14
                icon: UIConstants.iZoomIn
                isAutoReturn: true
                isShowRect: false
                anchors.verticalCenter: parent.verticalCenter
            }

            FlatButtonIcon{
                id: btnZoomOut
                x: 147
                y: 70
                width: UIConstants.sRect*2
                height: UIConstants.sRect*2
                anchors.right: btnRight.left
                anchors.rightMargin: 14
                anchors.verticalCenterOffset: 0
                icon: UIConstants.iZoomOut
                isAutoReturn: true
                isShowRect: false
                isSolid: true
                anchors.verticalCenter: parent.verticalCenter
            }

        }

        Rectangle {
            id: rectInfo
            x: 336
            y: 247
            width: 240
            height: 135
            color: "transparent"
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 8
            anchors.right: parent.right
            anchors.rightMargin: 8

            Label {
                id: lblPan
                x: 132
                y: 120
                width: 100
                height: 16
                color: drawColor
                text: Number(pan).toFixed(1).toString()+degreeSymbol
                verticalAlignment: Text.AlignVCenter
                horizontalAlignment: Text.AlignHCenter
            }

            Label {
                id: lblTilt
                x: 10
                y: 120
                width: 100
                height: 16
                color: drawColor
                text: Number(tilt).toFixed(1).toString()+degreeSymbol
                verticalAlignment: Text.AlignVCenter
                horizontalAlignment: Text.AlignHCenter
            }

            Canvas{
                id: cvsTilt
                x: 10
                y: 8
                width: 100
                height: 100
                onPaint: {
                    var ctx = getContext("2d");
                    ctx.reset();
                    var x = width / 2
                    var y = height / 2
                    var lineSize = 5;
                    var losSize = width /2 - lineSize*2;
                    ctx.beginPath();
                    ctx.lineWidth = 2
                    // draw circle
                    ctx.arc(x, y, (width / 2) - ctx.lineWidth/2, -Math.PI/2, Math.PI, false)
                    ctx.strokeStyle = drawColor
                    ctx.stroke()
                    ctx.beginPath();
                    ctx.moveTo(width/2,0)
                    ctx.lineTo(width/2,lineSize)
                    ctx.moveTo(width/2,height)
                    ctx.lineTo(width/2,height-lineSize)
                    ctx.moveTo(0,height/2)
                    ctx.lineTo(lineSize,height/2)
                    ctx.moveTo(width,height/2)
                    ctx.lineTo(width-lineSize,height/2)
                    ctx.stroke()
                    // draw LOS
                    ctx.beginPath();
                    ctx.fillStyle = drawColor
                    ctx.moveTo(width/2,height/2)
                    ctx.arc(x, y, losSize,
                            (tilt - vfov/2 + 90)/180*Math.PI,
                            (tilt + vfov/2 + 90)/180*Math.PI, false)
                    ctx.lineTo(x,y);
                    //            ctx.fill()
                    ctx.stroke()
                }
            }

            Canvas{
                id: cvsPan
                x: 132
                y: 8
                width: 100
                height: 100
                onPaint: {
                    var ctx = getContext("2d");
                    ctx.reset();
                    var x = width / 2
                    var y = height / 2
                    var lineSize = 5;
                    var losSize = width /2 - lineSize*2;
                    ctx.beginPath();
                    ctx.lineWidth = 2
                    // draw circle
                    ctx.arc(x, y, (width / 2) - ctx.lineWidth/2, 0, Math.PI*2, false)
                    ctx.strokeStyle = drawColor
                    ctx.stroke()
                    // draw sequence guide
                    ctx.beginPath();
                    ctx.moveTo(width/2,0)
                    ctx.lineTo(width/2,lineSize)
                    ctx.moveTo(width/2,height)
                    ctx.lineTo(width/2,height-lineSize)
                    ctx.moveTo(0,height/2)
                    ctx.lineTo(lineSize,height/2)
                    ctx.moveTo(width,height/2)
                    ctx.lineTo(width-lineSize,height/2)
                    ctx.stroke()
                    // draw LOS
                    ctx.beginPath();
                    ctx.fillStyle = drawColor
                    ctx.moveTo(width/2,height/2)
                    ctx.arc(x, y, losSize,
                            (pan - hfov/2 - 90)/180*Math.PI,
                            (pan + hfov/2 - 90)/180*Math.PI, false)
                    ctx.lineTo(x,y);
                    //            ctx.fill()
                    ctx.stroke()
                }
            }
        }

        Timer{
            running: true
            repeat: true
            interval: 100
            onTriggered: {
//                camState.panPos += dPan;
//                camState.tiltPos+= dTilt;
//                if(camState.panPos > 360 || camState.panPos < 0){
//                    dPan = -dPan
//                }
//                if(camState.tiltPos > 120 || camState.tiltPos < -120){
//                    dTilt = -dTilt
//                }
                cvsTilt.requestPaint()
                cvsPan.requestPaint()
            }
        }


    }
}

/*##^## Designer {
    D{i:23;anchors_x:8}D{i:18;anchors_x:8}
}
 ##^##*/
