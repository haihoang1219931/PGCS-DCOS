import QtQuick 2.0
import io.qdt.dev 1.0
import CustomViews.UIConstants 1.0
import QtGraphicalEffects 1.0

Item
{
    id:rootItem
    property string mapHeightFolder: null
    property variant coord1: undefined
    property variant coord2: undefined
    property var itemListName:
        UIConstants.itemTextMultilanguages
    Rectangle{
        id: rectProfilePath
        z:200
        anchors.fill: parent
        color: UIConstants.transparentBlue
        radius: UIConstants.rectRadius
        border.width: 1
        border.color: UIConstants.grayColor
        clip:true

        ProfilePath{
            id: profilePath
            title: rootItem.itemListName["DIALOG"]["PROFILE_PATH"]["TITTLE"]
                   [UIConstants.language[UIConstants.languageID]] + " UAV"
            xName: rootItem.itemListName["DIALOG"]["PROFILE_PATH"]["DISTANCE"]
                   [UIConstants.language[UIConstants.languageID]]
            yName: rootItem.itemListName["DIALOG"]["PROFILE_PATH"]["ALTITUDE"]
                   [UIConstants.language[UIConstants.languageID]]
            fontSize: UIConstants.fontSize
            fontFamily: UIConstants.appFont
            anchors.fill: parent
            anchors.margins: 4
            isShowLineOfSight: true
            folderPath: computer.homeFolder()+"/ArcGIS/Runtime/Data/elevation/"+mapHeightFolder

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

        Rectangle
        {
            id: _uavPredictionPoint
            color: "transparent"
            width: 14
            height: 14
            radius: width/2
            border.color: "brown"
            border.width: 2

//            RadialGradient{
//                anchors.fill: parent
//                gradient:  Gradient{
//                    GradientStop{position: 0.0 ; color: "red"; }
//                    GradientStop{position: 0.2 ; color: "red"; }
//                    GradientStop{position: 0.5 ; color: "transparent" ; }
//                }
//            }

        }
    }

    function setLocation(_coord1,_coord2){ //relative alt
        //var altHome = vehicle.link ? (vehicle.altitudeAMSL - vehicle.altitudeRelative) : mapPane.virtualHomeAMSL
        coord1 = _coord1;
        coord2 = _coord2;
        profilePath.addElevation(coord1,coord2);
        console.log("set location")

    }

    function setVehiclePosition(coord){
        if(coord1!== undefined && coord2!== undefined){
            var p = profilePath.convertCoordinatetoXY(coord,coord1,coord2);

            if( p.x < 0) p.x=0;

            _uavPoint.x = p.x + profilePath.axisXOffset - _uavPoint.width/2 + profilePath.anchors.leftMargin;
            _uavPoint.y = p.y + profilePath.height - profilePath.axisYOffset - _uavPoint.height/2 + profilePath.anchors.topMargin;

            //update warning

            var prePlanePos = profilePath.planePosPrediction(coord,coord2,vehicle.groundSpeed,60); //60s
            //console.log("coord2:"+coord2.altitude)
            p = profilePath.convertCoordinatetoXY(prePlanePos,coord1,coord2);
            if( p.x < 0) p.x=0;
            _uavPredictionPoint.x = p.x + profilePath.axisXOffset - _uavPredictionPoint.width/2 + profilePath.anchors.leftMargin;
            _uavPredictionPoint.y = p.y + profilePath.height - profilePath.axisYOffset - _uavPredictionPoint.height/2 + profilePath.anchors.topMargin;

            if(profilePath.checkAltitudeWarning(coord,prePlanePos)){
                rectProfilePath.border.color = UIConstants.redColor;
            }
            else
                rectProfilePath.border.color = UIConstants.grayColor;
        }
    }

    //waypoint
    function setWpLineOfSight(coord1,coord2){ //absolute
        setLocation(coord1,coord2)
        console.log("setLocaaiton light of sight")
        //var altHome = vehicle.link ? (vehicle.altitudeAMSL - vehicle.altitudeRelative) : mapPane.virtualHomeAMSL
        profilePath.setLineOfSight(0,coord1.altitude, coord1.distanceTo(coord2),coord2.altitude);
    }

    function setUavProfilePathMode(val){ //0:WPLineOfSight; 1:Vehicle
        if(val === 0){
            profilePath.isShowLineOfSight = true
            _uavPoint.visible = false
            _uavPredictionPoint.visible = false
        }else if(val === 1){
            profilePath.isShowLineOfSight = false
            _uavPoint.visible = true
            _uavPredictionPoint.visible = true
        }
    }
}

