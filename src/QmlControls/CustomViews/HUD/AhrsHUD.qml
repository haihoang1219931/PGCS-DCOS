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

Item {
    property real pitchAngle:       vehicle ? vehicle.pitch  : 0
    property real rollAngle:        vehicle ? vehicle.roll  : 0
    property real yawAngle:         vehicle ? vehicle.heading  : 0
    property real altitude:         vehicle ? vehicle.altitudeRelative  : 0
    property real airspeed:         vehicle ? vehicle.airSpeed * 3.6  : 0

    property real _reticleHeight:   1
    property real _reticleSpacing:  rootItem.height * 0.10
    property real _reticleSlot:     _reticleSpacing + _reticleHeight
    property real _longDash:        rootItem.width * 0.25
    property real _shortDash:       rootItem.width * 0.14
    property real angularScale: pitchAngle * _reticleSlot / 10

    property real _cr_line1_width: rootItem.width * 0.35
    property real _cr_line2_width: rootItem.width * 0.16
    property real _cr_line3_height: rootItem.width * 0.22
    property real _cr_line_thickness: 3
    property string _cr_line_color : "red"

    property real _headingHeight: UIConstants.sRect*1.5
    property real _headingWidth: rootItem.width * 0.8
    property real _headingSpacing: rootItem.width * 0.09
    property real _headingLongDash:        rootItem.height * 0.04
    property real _headingShortDash:       rootItem.height * 0.02
    property real _headingThickness:       1
    property real _headingSlot:     _headingSpacing + _headingThickness

    property real _airspeedWidth: UIConstants.sRect*2.2
    property real _airspeedHeight: rootItem.height * 0.4

    property real _altitudeWidth: UIConstants.sRect*2.2
    property real _altitudeHeight: rootItem.height * 0.4

    property real _rulerLongDash:        rootItem.height * 0.03
    property real _rulerShortDash:        rootItem.height * 0.015
    property real _rulerThickness:        1
    property real _rulerSpacing: rootItem.width * 0.015
    property real _rulerSlot : _rulerSpacing + _rulerThickness

    property real _radiusCircleRoll : rootItem.width * 0.4

    property real startAngle : -Math.PI*0.25
    property real endAngle : -Math.PI*0.75

    id: rootItem
    width: UIConstants.sRect*14
    height: UIConstants.sRect*14

    layer.enabled: true
    layer.effect: OpacityMask {
        anchors.fill: rootItem
        maskSource: Item {
            anchors.fill: parent
            width: rootItem.width
            height: rootItem.height
            Rectangle {
                anchors.fill: parent
                width: rootItem.width
                height: rootItem.height
                radius: UIConstants.rectRadius
            }
        }
    }
    Rectangle{
        id: borderRect
        z:1
        anchors.fill: parent
        radius: UIConstants.rectRadius
        border.width: 2
        border.color: "gray"
        color: "transparent"
    }

    Rectangle{
        id:rootRect
        anchors.fill: parent
        color: UIConstants.bgPanelColor
        clip: true
        radius: UIConstants.rectRadius
        Item{
            id: pictchClip
            anchors.fill:parent
            anchors.topMargin: _headingHeight + 2
            //color:"green"
            z: 1
            clip:true
            Item {
                id: pitchItem
                x: 0
                y: 0//-(_headingHeight + 2)
                anchors.fill: parent
                z:1

                Rectangle{
                    id: pitchRect
                    anchors.fill: parent
                    color: "transparent"
                    Column{
                        anchors.horizontalCenter: parent.horizontalCenter
                        anchors.verticalCenter:   parent.verticalCenter
                        spacing: _reticleSpacing
                        Repeater {
                            model: 36
                            Rectangle {
                                property int _pitch: -(modelData * 10 - 180)
                                anchors.horizontalCenter: parent.horizontalCenter
                                width: (_pitch % 20) === 0 ? _longDash : _shortDash
                                height: _reticleHeight
                                color: ((_pitch - pitchAngle) < 30 && (_pitch - pitchAngle) > -40)? UIConstants.textColor : UIConstants.transparentColor
                                antialiasing: true
                                smooth: true

                                Text {
                                    anchors.horizontalCenter: parent.horizontalCenter
                                    anchors.horizontalCenterOffset: -(_longDash / 2 + UIConstants.sRect/2)
                                    anchors.verticalCenter: parent.verticalCenter
                                    smooth: true
                                    text: _pitch
                                    color: UIConstants.textColor
                                    font.family: "Arial"
                                    font.pixelSize: UIConstants.fontSize/1.2
                                    visible: (_pitch != 0) && ((_pitch % 20) === 0) &&
                                             (((_pitch - pitchAngle) < 30) && ((_pitch - pitchAngle) > -40))
                                }
                                Text {
                                    anchors.horizontalCenter: parent.horizontalCenter
                                    anchors.horizontalCenterOffset: (_longDash / 2 + UIConstants.sRect/2)
                                    anchors.verticalCenter: parent.verticalCenter
                                    smooth: true
                                    text: _pitch
                                    color: UIConstants.textColor
                                    font.family: "Arial"
                                    font.pixelSize: UIConstants.fontSize/1.2
                                    visible: (_pitch != 0) && ((_pitch % 20) === 0) &&
                                             (((_pitch - pitchAngle) < 30) && ((_pitch - pitchAngle) > -40))
                                }
                            }
                        }
                    }
                    transform: [ Translate {
                            y: (pitchAngle * _reticleSlot / 10) - (_reticleSlot / 2)
                        }]
                }
                transform: [
                    Rotation {
                        origin.x: pictchClip.width  / 2
                        origin.y: pictchClip.height / 2
                        angle:    -rollAngle
                    }
                ]
            }
        }

        Item{
            id: crossHairItem
            anchors.fill: pictchClip
            z:3
            Rectangle{
                id:line1
                anchors.horizontalCenter: parent.horizontalCenter
                height: _cr_line_thickness
                width: _cr_line1_width
                color: _cr_line_color
                antialiasing: true
                smooth: true
                y: parent.height / 2 - height/2
            }
            Rectangle{
                id:line2
                anchors.horizontalCenter: parent.horizontalCenter
                height: _cr_line_thickness
                width: _cr_line2_width
                color: _cr_line_color
                antialiasing: true
                smooth: true
                y: parent.height / 2 - rootItem.height / 9 - height/2
            }
            Rectangle{
                id:line3
                anchors.horizontalCenter: parent.horizontalCenter
                height: _cr_line3_height
                width:  _cr_line_thickness
                color: _cr_line_color
                antialiasing: true
                smooth: true
                y:parent.height / 2 - line3.height
            }
        }

        Item {
            z: 0
            id: artificialHorizon
            width:  pictchClip.width  * 4
            height: pictchClip.height * 8
            anchors.centerIn: pictchClip
            clip:true
            Rectangle {
                id: sky
                anchors.fill: parent
                smooth: true
                antialiasing: true
                gradient: Gradient {
                    GradientStop { position: 0.0; color: Qt.hsla(0.6, 1.0, 0.25) }
                    GradientStop { position: 0.3;  color: Qt.hsla(0.6, 1.0, 0.4) }
                    GradientStop { position: 0.5;  color: Qt.hsla(0.6, 0.5, 0.65) }
                }
            }
            Rectangle {
                id: ground
                height: sky.height / 2
                anchors {
                    left:   sky.left;
                    right:  sky.right;
                    bottom: sky.bottom
                }
                smooth: true
                antialiasing: true
                gradient: Gradient {
                    GradientStop { position: 0.0;  color: Qt.rgba(0.95, 0.73, 0.47,1) }
                    GradientStop { position: 0.15; color: Qt.rgba(0.66, 0.39, 0.07,1) }
                    GradientStop { position: 0.5; color: Qt.rgba(0.59, 0.34, 0.04,1) }
                }
            }
            transform: [
                Translate {
                    y:  angularScale
                },
                Rotation {
                    origin.x: artificialHorizon.width  / 2
                    origin.y: artificialHorizon.height / 2
                    angle:    -rollAngle
                }]
        }

        ///heading
        Item {
            id: headingItem
            anchors.horizontalCenter: parent.horizontalCenter
            height:  _headingHeight
            width:  _headingWidth
            y:0
            z:4
            clip: true

            Rectangle{
                id: headingRect
                anchors.verticalCenter: parent.verticalCenter
                anchors.horizontalCenter: parent.horizontalCenter
                width: parent.height
                height: parent.width
                smooth: true
                antialiasing: true
                rotation: 90
                opacity: 0.3
                gradient: Gradient {
                    GradientStop { position: 0.0;  color: "transparent"; }
                    GradientStop { position: 0.15; color: "gray"; }
                    GradientStop { position: 0.5; color: "gray"; }
                    GradientStop { position: 0.85; color: "gray"; }
                    GradientStop { position: 1.0; color: "transparent"; }
                }
            }

            Row{
                id:yawRow
                anchors.horizontalCenter: parent.horizontalCenter
                anchors.top:   parent.top
                spacing: _headingSpacing

                //clip: true
                Repeater {
                    model: 96
                    Rectangle {
                        property int _yaw: (modelData * 15) - 720
                        anchors.top: parent.top
                        anchors.topMargin: 4
                        height: (_yaw % 45) === 0 ? _headingLongDash : _headingShortDash
                        width: (_yaw % 45) === 0 ?_headingThickness : _headingThickness
                        color: (_yaw % 45) === 0 ? UIConstants.textColor : "silver"

                        //                        antialiasing: true
                        //                        smooth: true
                        Text {
                            id:_text
                            anchors.horizontalCenter: parent.horizontalCenter
                            anchors.top : parent.bottom
                            anchors.topMargin: 0
                            smooth: true
                            visible: (_yaw - yawAngle) < -10 || (_yaw - yawAngle) >10
                            font.family: "Arial"
                            color: UIConstants.textColor
                            font.pixelSize: UIConstants.fontSize/1.2
                            font.bold: (_yaw % 45 == 0) ? true : false
                            text: {
                                var yaw = parseInt(_yaw);
                                if(yaw < 0) yaw += 360;
                                else if(yaw > 360) yaw -= 360;
                                if(yaw === 360) yaw = 0

                                switch(yaw){
                                case 0:
                                    return "N";
                                case 45:
                                    return "NE";
                                case 90:
                                    return "E";
                                case 135:
                                    return "SE";
                                case 180:
                                    return "S";
                                case 225:
                                    return "SW";
                                case 270:
                                    return "W";
                                case 315:
                                    return "NW";
                                default:
                                    return yaw;
                                }
                            }

                            //visible: (_pitch != 0) && ((_pitch % 20) === 0)

                        }

                    }
                }
                transform: [ Translate {
                        x: -(yawAngle * _headingSlot / 15) - (_headingSlot / 2)
                    }]
            }

            Rectangle{
                id:yawRect
                anchors.horizontalCenter: parent.horizontalCenter
                anchors.top: parent.top
                anchors.topMargin: UIConstants.sRect*0.5
                width: UIConstants.sRect*1.5
                height: UIConstants.sRect*0.9
                color: "#cc626363"
                border.width: 1
                radius:3
                border.color: "white"
                Text {
                    id: txtYaw
                    anchors.fill: parent
                    text: parseInt(yawAngle)
                    verticalAlignment: Text.AlignVCenter
                    horizontalAlignment: Text.AlignHCenter
                    font.family: "Arial"
                    font.pixelSize: UIConstants.fontSize
                    font.bold: true
                    color: "white"
                }

            }

            Rectangle{
                id: lineYawValue
                color: "black"
                opacity: 0.35
                width: 3
                anchors.horizontalCenter: parent.horizontalCenter
                anchors.top: parent.top
                anchors.bottom: yawRect.top
            }
        }

        Item {
            id: altitudeItem
            anchors.left: parent.left
            anchors.verticalCenter: pictchClip.verticalCenter
            width: _altitudeWidth
            height: _altitudeHeight
            clip:true
            z:5
            Rectangle{
                id:altitudeRect
                color: UIConstants.transparentColor
                anchors.top:parent.top
                anchors.left: parent.left
                width: parent.width
                height: parent.height
                clip: true
                Column{
                    anchors.right: parent.right
                    anchors.verticalCenter:   parent.verticalCenter
                    spacing: _rulerSpacing
                    Repeater {
                        model: 2 * parseInt(_altitudeHeight / _rulerSpacing)
                        Rectangle {
                            property int _alt: -(modelData - parseInt(_altitudeHeight / _rulerSpacing)) + parseInt(altitude)
                            anchors.right: parent.right
                            width: (_alt % 5) === 0 ? _rulerLongDash : _rulerShortDash
                            height: _rulerThickness
                            color: (_alt % 5) === 0 ? UIConstants.textColor : "silver"
                            Text {
                                anchors.right: parent.left
                                anchors.rightMargin: 3
                                anchors.verticalCenter: parent.verticalCenter
                                text: _alt
                                color: UIConstants.textColor
                                font.family: "Arial"
                                font.pixelSize: UIConstants.fontSize/1.2
                                visible: (_alt != 0) && ((_alt % 5) === 0)
                            }
                        }
                    }
                    transform: [ Translate {
                            y:  - (_rulerSlot / 2) * (1+2*(parseInt(altitude) - altitude))
                        }]
                }

            }

            Canvas{
                id: arrowAltitue
                anchors.left: parent.left
                anchors.verticalCenter: parent.verticalCenter
                width: parent.width - _rulerLongDash
                height: UIConstants.sRect/1.1
                clip:true
                onPaint: {
                    var ctx = getContext("2d")
                    ctx.fillStyle = "#cc626363";
                    ctx.strokeStyle = "white"
                    //ctx.fillRect(0,0,width,height)
                    ctx.lineWidth = 1
                    ctx.beginPath();
                    ctx.moveTo(0,0);
                    ctx.lineTo(width * 0.83 , 0);
                    ctx.lineTo(width,height/2);
                    ctx.lineTo(width * 0.83 , height);
                    ctx.lineTo(0,height);
                    ctx.closePath();
                    ctx.stroke();
                    ctx.fill();
                }

                Text {
                    id: txtAltValue
                    anchors.right: parent.right
                    anchors.verticalCenter: parent.verticalCenter
                    anchors.rightMargin: 6
                    text: parseInt(altitude)
                    font.family: "Arial"
                    font.pixelSize: UIConstants.fontSize
                    font.bold: true
                    color: "white"
                }
            }

        }


        Item {
            id: airspeedItem
            anchors.right: parent.right
            anchors.verticalCenter: pictchClip.verticalCenter
            width: _airspeedWidth
            height: _airspeedHeight
            clip:true
            z:5
            Rectangle{
                id:airspeedRect
                color: UIConstants.transparentColor
                anchors.fill: parent
                clip: true
                Column{
                    anchors.left: parent.left
                    anchors.verticalCenter:   parent.verticalCenter
                    spacing: _rulerSpacing
                    Repeater {
                        model: 2 * parseInt(_airspeedHeight / _rulerSpacing)
                        Rectangle {
                            property int _airspeed: -(modelData - parseInt(_airspeedHeight / _rulerSpacing)) + parseInt(airspeed)
                            anchors.left: parent.left
                            width: (_airspeed % 5) === 0 ? _rulerLongDash : _rulerShortDash
                            height: _rulerThickness
                            color: (_airspeed % 5) === 0 ? UIConstants.textColor : "silver"
                            Text {
                                anchors.left: parent.right
                                anchors.leftMargin: 3
                                anchors.verticalCenter: parent.verticalCenter
                                text: _airspeed
                                color: UIConstants.textColor
                                font.family: "Arial"
                                font.pixelSize: UIConstants.fontSize/1.2
                                visible: (_airspeed != 0) && ((_airspeed % 5) === 0)
                            }
                        }
                    }
                    transform: [ Translate {
                            y: - (_rulerSlot / 2)* (1 + 2*(parseInt(airspeed) - airspeed))
                        }]
                }

            }

            Canvas{
                id: arrowAirspeed
                anchors.right: parent.right
                anchors.verticalCenter: parent.verticalCenter
                width: parent.width - _rulerLongDash
                height: UIConstants.sRect/1.1
                clip:true
                onPaint: {
                    var ctx = getContext("2d")
                    ctx.fillStyle = "#cc626363";
                    ctx.strokeStyle = "white"
                    //ctx.fillRect(0,0,width,height)
                    ctx.lineWidth = 1
                    ctx.beginPath();
                    ctx.moveTo(width,0);
                    ctx.lineTo(width * 0.17 , 0);
                    ctx.lineTo(0,height/2);
                    ctx.lineTo(width * 0.17 , height);
                    ctx.lineTo(width,height);
                    ctx.closePath();
                    ctx.stroke();
                    ctx.fill();
                }

                Text {
                    id: txtArspdValue
                    anchors.left: parent.left
                    anchors.verticalCenter: parent.verticalCenter
                    anchors.leftMargin: 6
                    text: parseInt(airspeed)
                    font.family: "Arial"
                    font.pixelSize: UIConstants.fontSize
                    font.bold: true
                    color: "white"
                }
            }


        }

        Item{
            id: unitText
            anchors.top: airspeedItem.bottom
            width: parent.width
            height: childrenRect.height
            z:5
            Text {
                id: txtKph
                anchors.top:parent.top
                anchors.right: parent.right
                anchors.rightMargin:_airspeedWidth - _rulerLongDash - width
                text: "kph"
                font.family: "Arial"
                font.pixelSize: UIConstants.fontSize
                color: "white"
            }

            Text {
                id: txtMet
                anchors.top:parent.top
                anchors.left: parent.left
                anchors.leftMargin: _altitudeWidth - _rulerLongDash - width
                text: "m"
                font.family: "Arial"
                font.pixelSize: UIConstants.fontSize
                color: "white"
            }
        }

        Item{
            id:roll_pitch_text
            anchors.right: parent.right;
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 3
            anchors.left: parent.left
            height: UIConstants.sRect
            z:7
            Text {
                id: pitchText
                anchors.right: parent.right
                anchors.top:parent.top
                anchors.rightMargin: 5
                width: UIConstants.sRect*1.9
                text: "P:"+ parseInt(pitchAngle);
                font.bold: true
                font.family: "Arial"
                font.pixelSize: UIConstants.fontSize

            }
            Text {
                id: rollText
                anchors.right: pitchText.left
                anchors.top:parent.top
                width: UIConstants.sRect*1.9
                text: "R:"+ parseInt(rollAngle);
                font.bold: true
                font.family: "Arial"
                font.pixelSize: UIConstants.fontSize
            }

        }


        Item{
            id: circleRoll
            anchors.fill: pictchClip
            z:7
            Canvas{
                id: canvasCircleRoll
                anchors.fill: parent
                onPaint: {
                    var ctx = getContext("2d")
                    var xCenter  = width/2
                    var yCenter = height/2
                    ctx.strokeStyle = "white"
                    ctx.lineWidth = 1.5
                    ctx.beginPath();

                    ctx.arc(xCenter,yCenter,_radiusCircleRoll, startAngle,endAngle,true)
                    var R = 0
                    for(var i = 0 ; i <=12 ; i++){
                        if(i % 3 == 0) {
                            R=_radiusCircleRoll+9
                        }
                        else{
                            R=_radiusCircleRoll+5
                        }
                        var angle = startAngle + i*(endAngle - startAngle)/12
                        var x1 = _radiusCircleRoll* Math.cos(angle) + xCenter
                        var y1 = _radiusCircleRoll* Math.sin(angle) + yCenter
                        var x2 = R* Math.cos(angle) + xCenter
                        var y2 = R* Math.sin(angle) + yCenter
                        ctx.moveTo(x1,y1)
                        ctx.lineTo(x2,y2)
                    }
                    ctx.stroke()
                }
            }

            Canvas{
                id: arrowRoll
                anchors.fill: parent
                onPaint: {
                    var ctx = getContext("2d")
                    ctx.fillStyle = "white";
                    //ctx.fillRect(0,0,width,height)
                    ctx.beginPath();

                    ctx.moveTo(width/2,height/2 - _radiusCircleRoll);
                    ctx.lineTo(width/2 + 5 , height/2 - _radiusCircleRoll + 12);
                    ctx.lineTo(width/2 - 5 , height/2 - _radiusCircleRoll + 12);
                    ctx.fill();
                }
                transform: [
                    Rotation {
                        origin.x: arrowRoll.width / 2
                        origin.y: height / 2 - _headingHeight /2
                        angle:    rollAngle * Math.abs(endAngle-startAngle) / (Math.PI * 2)
                    }]
            }

            Text{
                id:text1
                z:4
                anchors.horizontalCenter: parent.horizontalCenter
                anchors.top: parent.top
                text:"-180"
                anchors.topMargin: - UIConstants.sRect/3
                font.family: "Arial"
                font.pixelSize: UIConstants.fontSize / 1.2
                color:"white"
                transform: [
                    Rotation {
                        origin.x: text1.width / 2
                        origin.y: circleRoll.height / 2 + UIConstants.sRect/2.2
                        angle:    -45
                    }]
            }
            Text{
                id:text2
                z:5
                anchors.horizontalCenter: parent.horizontalCenter
                anchors.top: parent.top
                text:"180"
                anchors.topMargin: - UIConstants.sRect/3
                font.family: "Arial"
                font.pixelSize: UIConstants.fontSize / 1.2
                color:"white"
                transform: [
                    Rotation {
                        origin.x: text2.width / 2
                        origin.y: circleRoll.height / 2 + UIConstants.sRect/2.2
                        angle:    45
                    }]
            }
        }
    }


}
