import QtQuick 2.0
import QtLocation 5.9
import QtPositioning 5.5
import QtQuick.Window 2.0
import QtGraphicalEffects 1.0

import io.qdt.dev 1.0
import CustomViews.UIConstants 1.0

MapQuickItem {
    id: gcs_symbol
    property int wpId: 0
    property int markerId:0
    property int missionItemType: 0
    property int symbolId:0
    property int waypointAlt:0
    property int param1:0
    property int param2:0
    property int param3:0
    property int param4:0

    property bool isMarker: false
    property int markerType: 0
    property string textMarker: "default"

    readonly property int  widthsymbol: 50
    readonly property int  heighsymbol: 50

    readonly property color waypoint_Color: "#02e6ed"
    readonly property color guidedpoint_Color: "#b87a33"
    readonly property color current_waypoint_Color: "#992be2"
    readonly property color dojump_Color: "#00a8ad"
    readonly property color takeoffOrland_Color: "yellow"
    readonly property color home_Color: "blue"
    readonly property color symbol_Selected_Color: "#c300ff"

    readonly property string takeoff_Source: "qrc:/assets/images/takeoff.png"
    readonly property string land_Source: "qrc:/assets/images/land.png"
//    readonly property string home_Source: "qrc:/assets/images/BasePosition.png"
    readonly property string home_Source: "qrc:/assets/images/home.png"
    readonly property string anti_clockwiseCircle_Source: "qrc:/qmlimages/markers/AntiClockwiseCircle.png"
    readonly property string clockwiseCircle_Source: "qrc:/qmlimages/markers/ClockwiseCircle.png"

    readonly property string default_marker_Source: "qrc:/qmlimages/markers/FlagIcon.png"
    readonly property string tank_marker_Source: "qrc:/qmlimages/markers/TankIcon.png"
    readonly property string plane_marker_Source: "qrc:/qmlimages/markers/PlaneIcon.png"
    readonly property string ship_marker_Source: "qrc:/qmlimages/markers/BattleShip.png"
    readonly property string target_marker_Source: "qrc:/qmlimages/markers/TargetIcon.png"
    
    property bool isSelected: false
    property bool iscurrentWP: false

    property bool isShowSymbolEditor: false
    signal showSymbolEditor()

    anchorPoint.x: _rec_symbol.width/2
    anchorPoint.y: _rec_symbol.height/2

//    coordinate: position
    sourceItem: Rectangle {
        id: _rec_symbol
        opacity: 0.85
        width: widthsymbol
        height: heighsymbol
        radius: width/2

        color: "transparent"

        Component.onCompleted:
        {
            if(isMarker)
            {
                _rec_symbol.color = "transparent"
                _imageTakeOfforLandorHome.visible=true
                _waypoint_text.visible = true
                _waypoint_text.color = "#dce3e3"
                _waypoint_text.y = 54

                switch(markerType)
                {
                    case 0:
                        _imageTakeOfforLandorHome.source = default_marker_Source
                        break;

                    case 1:
                        _imageTakeOfforLandorHome.source = tank_marker_Source
                        break;

                    case 2:
                        _imageTakeOfforLandorHome.source = plane_marker_Source
                        break;

                    case 3:
                        _imageTakeOfforLandorHome.source = target_marker_Source
                        break;

                    case 4:
                        _imageTakeOfforLandorHome.source = ship_marker_Source
                        break;

                    default:
                        break;
                }
            }
            else
            {
                switch(missionItemType)
                {
                    case UIConstants.waypointType:
                        if(wpId === 0)
                        {
                            _rec_symbol.color = home_Color
                            _imageTakeOfforLandorHome.visible=true
                            _imageTakeOfforLandorHome.source=home_Source
                            _waypoint_text.visible=true
                        }
                        else if(wpId === -1){ //guided waypoint
                            _rec_symbol.color = guidedpoint_Color
                            _waypoint_id.visible=true
                            _waypoint_id.text = "G";
                            _waypoint_text.visible=false
                        }
                        else
                        {
                            if(iscurrentWP)
                                _rec_symbol.color = current_waypoint_Color
                            else
                                 _rec_symbol.color = waypoint_Color
                            _waypoint_id.visible=true
                            _waypoint_text.visible=true
                        }
                        break;

                    case UIConstants.loitertimeType:
                        if(iscurrentWP)
                            _rec_symbol.color = current_waypoint_Color
                        else
                             _rec_symbol.color = waypoint_Color
                        _waypoint_id.visible=true
                        _waypoint_text.visible=true
                        _imageTakeOfforLandorHome.visible=true
                        if(param3 == 1)  //check loiter dir
                            _imageTakeOfforLandorHome.source=clockwiseCircle_Source
                        else
                            _imageTakeOfforLandorHome.source=anti_clockwiseCircle_Source
                        break;

                    case UIConstants.landType:
                        _waypoint_text.visible=true
                        _rec_symbol.color = waypoint_Color //takeoffOrland_Color
                        _imageTakeOfforLandorHome.visible=true
                        _imageTakeOfforLandorHome.source=land_Source
                        _imageTakeOfforLandorHome.opacity = 0.7
                        break;

                    case UIConstants.takeoffType:
                        _waypoint_text.visible=true
                        _rec_symbol.color = waypoint_Color //takeoffOrland_Color
                        _imageTakeOfforLandorHome.visible=true
                        _imageTakeOfforLandorHome.source=takeoff_Source
                        _imageTakeOfforLandorHome.opacity = 0.7
                        break;

                    case UIConstants.vtoltakeoffType:
                        _waypoint_text.visible=true
                        _rec_symbol.color = waypoint_Color //takeoffOrland_Color
                        _imageTakeOfforLandorHome.visible=true
                        _imageTakeOfforLandorHome.source=takeoff_Source
                        _imageTakeOfforLandorHome.opacity = 0.7
                        break;

                    case UIConstants.vtollandType:
                        _waypoint_text.visible=true
                        _rec_symbol.color = waypoint_Color //takeoffOrland_Color
                        _imageTakeOfforLandorHome.visible=true
                        _imageTakeOfforLandorHome.source=land_Source
                        _imageTakeOfforLandorHome.opacity = 0.7
                        break;

                    case UIConstants.dojumpType:
                        _rec_symbol.color = dojump_Color
                        _waypoint_id.visible=true
                        _waypoint_text.visible=true
                        break;

                    default:
                        break;
                }
            }
        }

        Text
        {
            id:_waypoint_id
            anchors.verticalCenter:  _rec_symbol.verticalCenter
            anchors.horizontalCenter: _rec_symbol.horizontalCenter
            text: qsTr(wpId.toString())
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment:  Text.AlignVCenter
            font.family: "Arial"
            font.weight: Font.Bold
            font.bold: true
            font.pointSize: 18
            color: "white"
            visible: false
        }

        Rectangle{
            id: _waypoint_text
            color: "transparent"
            anchors.horizontalCenter: _rec_symbol.horizontalCenter
            y:52
            width: childrenRect.width + 4
            height: childrenRect.height + 2
            visible: true
            radius: 2
            Text
            {
                x: 2
                text: isMarker? textMarker : (missionItemType === UIConstants.dojumpType? qsTr("→"+param1): qsTr(waypointAlt.toString() + "m"))
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment:  Text.AlignVCenter
                font.weight: Font.Medium
                font.family: "Arial"
                font.pointSize: 13
                color: isMarker? "black" : "white"
            }
        }

        Image
        {
            id:_imageTakeOfforLandorHome
            width: widthsymbol
            height: heighsymbol
            anchors.horizontalCenter: _rec_symbol.horizontalCenter
            anchors.verticalCenter: _rec_symbol.verticalCenter
            visible: false
            opacity: 0.9
            smooth: true
            antialiasing: true
        }
        Rectangle
        {
            id: _selected_symbol
            visible: isSelected
            anchors.verticalCenter:  _rec_symbol.verticalCenter
            anchors.horizontalCenter: _rec_symbol.horizontalCenter
            x:-20
            y:-20
            color: "transparent"
            width: widthsymbol+40
            height: heighsymbol+40
            radius: width/2
//            border.color: symbol_Selected_Color
            //opacity: 0.75
//            border.width: 10
            RadialGradient{
                anchors.fill: parent
                gradient:  Gradient{
                    GradientStop{position: 0.0 ; color: "transparent"; }
                    GradientStop{position: 0.26 ; color: "transparent" }
                    GradientStop{position: 0.27 ; color: symbol_Selected_Color; }
                    GradientStop{position: 0.5 ; color: "transparent" ; }
                }
            }
        }

        Canvas {
            id: cvsSymbol
            x: parent.x - 10
            y: parent.y - 10
            width: parent.width + 20
            height: parent.height + 20
            contextType: "2d"
            antialiasing: true
            opacity: 0.8
            property real angle: 0
            onPaint: {
                var ctx = getContext('2d');
                ctx.reset();

                var x = width / 2
                var y = height / 2
                var losSize = width /2 - 5;
                ctx.strokeStyle = "red";
                ctx.beginPath();
                ctx.lineWidth = 9;
                ctx.arc(x, y, losSize,
                        0,
                        angle/180*Math.PI, false)
                context.stroke();
            }
            Timer{
                property bool presSymbol: false
                id: timerPressAndHold
                interval: 30
                repeat: true
                running: false
                onTriggered: {
                    cvsSymbol.angle +=12;
                    cvsSymbol.requestPaint();
                    if(cvsSymbol.angle > 360){
                        cvsSymbol.angle = 0;
                        stop();
                    }
                }
                onRunningChanged: {
                    if(running === false && presSymbol === true){
                        presSymbol = false;
                        showSymbolEditor();
//                        markerEditor.visible = true;
//                        markerEditor.changeState();
                    }
                }
            }
        }

    }

    function startTimerEditSymbol()
    {
        timerPressAndHold.start();
        timerPressAndHold.presSymbol = true;
    }
    function stopTimerEditSymbol()
    {
        timerPressAndHold.presSymbol = false;
        timerPressAndHold.stop();
        cvsSymbol.angle = 0;
        cvsSymbol.requestPaint();
    }
}

