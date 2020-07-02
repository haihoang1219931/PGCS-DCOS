import QtQuick 2.0
import io.qdt.dev 1.0
import CustomViews.UIConstants 1.0
import QtGraphicalEffects 1.0

Item
{
    id:rootItem
    property string mapHeightFolder: null
    property variant coord1: null
    property variant coord2: null

    signal pinClicked()

    //anchors.fill: rectProfilePath
    Rectangle{
        id: rectProfilePath
        z:200
        width: UIConstants.sRect * 12
        height: UIConstants.sRect * 5.5
        color: UIConstants.transparentBlue
        radius: UIConstants.rectRadius
        border.width: 1
        border.color: UIConstants.grayColor
        Drag.active: mouseArea.drag.active
        clip:true

        states: State {
            when: mouseArea.drag.active
            ParentChange { target: rectProfilePath; parent: rootItem }
            AnchorChanges { target: rectProfilePath; anchors.verticalCenter: undefined; anchors.horizontalCenter: undefined }
        }
        MouseArea {
            id: mouseArea

            anchors.fill: !btnPin.isPressed ? rectProfilePath : undefined
            enabled: !btnPin.isPressed
            drag.target: !btnPin.isPressed ? rectProfilePath : undefined

            //onReleased: parent = rectProfilePath.Drag.target !== null ? rectProfilePath.Drag.target : rootItem
        }

        FlatButtonIcon{
            id: btnPin
            width: 24
            height: 24
            anchors.top: parent.top
            anchors.right: parent.right
            icon: UIConstants.iPin
            iconSize: 23
            isSolid: true
            isShowRect: true
            isPressed: true
            visible: true
            isAutoReturn : false
            anchors{topMargin: 1;rightMargin: 1;}

            onClicked: {
                pinClicked();
            }
        }

        ProfilePath{
            id: profilePath
            title: ""
            xName: "(m)"
            yName: "(m)"
            fontSize: UIConstants.miniFontSize
            fontFamily: UIConstants.appFont
            anchors.fill: parent
            anchors{leftMargin: -8;rightMargin: -15;topMargin: -42;bottomMargin: -15;}
            isShowLineOfSight: true

        }
        Computer{
            id: cInfo
        }

        Rectangle
        {
            id: _uavPoint
            color: "transparent"
            width: 20
            height: 20
            radius: width/2

            RadialGradient{
                anchors.fill: parent
                gradient:  Gradient{
                    GradientStop{position: 0.0 ; color: "green"; }
                    GradientStop{position: 0.2 ; color: "blue"; }
                    GradientStop{position: 0.5 ; color: "transparent" ; }
                }
            }

            SequentialAnimation{
                id: animationSize
                running: true
                loops: Animation.Infinite
                NumberAnimation{target: _uavPoint; property: "opacity" ; to:0.0; duration: 400;}
                NumberAnimation{target: _uavPoint; property: "opacity" ; to:1; duration: 400;}
                NumberAnimation{target: _uavPoint; property: "opacity" ; to:0.0; duration: 400;}
                NumberAnimation{target: _uavPoint; property: "opacity" ; to:1; duration: 400;}
                NumberAnimation{target: _uavPoint; property: "opacity" ; to:1; duration: 400;}
                NumberAnimation{target: _uavPoint; property: "opacity" ; to:1; duration: 400;}
                NumberAnimation{target: _uavPoint; property: "opacity" ; to:1; duration: 400;}
            }
        }

    }

    function setLocation(_coord1,_coord2){
        coord1 = _coord1;
        coord2 = _coord2;
        profilePath.addElevation(
                    cInfo.homeFolder()+"/ArcGIS/Runtime/Data/elevation/"+mapHeightFolder,
                    coord1,coord2);


    }

    function setVehiclePosition(coord){
        if(coord1!== undefined && coord2!== undefined){
            var p = profilePath.convertCoordinatetoXY(coord,coord1,coord2);

            if( p.x < 0) p.x=0;

            _uavPoint.x = p.x + profilePath.axisXOffset - _uavPoint.width/2 + profilePath.anchors.leftMargin;
            _uavPoint.y = p.y + profilePath.height - profilePath.axisYOffset - _uavPoint.height/2 + profilePath.anchors.topMargin;
        }
    }

    //waypoint
    function setWpLineOfSight(coord1,coord2){

        setLocation(coord1,coord2)
        var altHome = vehicle.link ? (vehicle.altitudeAMSL - vehicle.altitudeRelative) : mapPane.virtualHomeAMSL
        profilePath.setLineOfSight(0,coord1.altitude + altHome, coord1.distanceTo(coord2),coord2.altitude + altHome);
    }

    function setUavProfilePathMode(val){ //0:WPLineOfSight; 1:Vehicle
        if(val === 0){
            profilePath.isShowLineOfSight = true
            _uavPoint.visible = false
        }else if(val === 1){
            profilePath.isShowLineOfSight = false
            _uavPoint.visible = true
        }
    }
}

