import QtQuick 2.6
import QtQuick.Window 2.2
import QtQuick.Controls 2.1
import QtQuick.Layouts 1.3
import QtGraphicalEffects 1.0
import QtQuick 2.0
//import QtLocation 5.6
//import QtPositioning 5.6
import QtLocation 5.9
import QtPositioning 5.0


import "qrc:/assets/javascripts/Helper.js" as Helper

import CustomViews.Dialogs 1.0
import CustomViews.UIConstants 1.0
import CustomViews.SubComponents 1.0
import CustomViews.Components 1.0

import Qt.labs.platform 1.0

import io.qdt.dev 1.0

Flickable {
    id: rootItem
    width: UIConstants.sRect*67
    height: UIConstants.sRect*32
    clip: true

    property variant map
    property variant plane
    property variant tracker
    property variant targetPolygon
    property variant opticalLine

    property bool isMapSync: false
    property int index_row_model_trajactory_plane:0
    property int index_row_model_symbol: 0
    property int index_row_model_marker: 0
    property var coordinate_clicked:null
    property var listsymbol: []
    property var listwaypoint: []
    property var listmarker: []
    property var listWPLine: []
    property var gotohereSymbol: null
    property bool pressSymbol: false

    property var selectedWP: undefined
    property var selectedMarker: undefined

    property var rollbackhomeposition:null

    property int selectedIndex: -1
    property int selectedmarkerIndex: -1
    property int currentWpIndex:-1
    property int addingWpIndex: -1
    property int old_currentWpIndex: -1

    property var planeTrajactory: null
    property var lastPlanePosition: null
    property var listPlaneTrajactory: []
    property var listRuler: []
    property int rulerCount: 0

    property var rulercoord1: null
    property var rulercoord2: null

    property var clickedLocation

    property bool allowSelectSymbol: true

    property string mapHeightFolder: "ElevationData-H1"

    signal homePositionChanged(real lat, real lon, real alt)
    signal symbolMoving(var id,var position)
    signal mapClicked(bool isMap)

//    property url dataPath: "file:///home/ttuav/ArcGIS/Runtime/Data/"
//    property url dataPath: "file:///home/ttuav/uavMap/"
    property url dataPath: StandardPaths.writableLocation(StandardPaths.HomeLocation)+ "/uavMap/"
    property url mapFolderPath: StandardPaths.writableLocation(StandardPaths.HomeLocation)+ "/uavMap/tpk/"
    property url urlMaps : StandardPaths.writableLocation(StandardPaths.HomeLocation)+"/mapGCS/"

    property var vehicleSymbolLink: {
        "MAV_TYPE_QUADROTOR":"qrc:/qmlimages/uavIcons/QuadRotorX.png",
        "MAV_TYPE_COAXIAL":"qrc:/qmlimages/uavIcons/QuadRotorX.png",
        "MAV_TYPE_VTOL_QUADROTOR":"qrc:/qmlimages/uavIcons/VTOLPlane.png",
        "MAV_TYPE_FIXED_WING":"qrc:/qmlimages/uavIcons/Plane.png",
        "MAV_TYPE_GENERIC":"qrc:/qmlimages/uavIcons/Unknown.png"
    }




    Timer
    {
        id:timerUpdateWP
        running: true
        repeat: true
        interval: 5
        onTriggered:
        {
            for(var i=2;i<listwaypoint.length;i++){
                var previousWP = listwaypoint[i-1]
                var waypoint = listwaypoint[i]
                if(waypoint !== null && waypoint.missionItemType === UIConstants.dojumpType)
                {
                    waypoint.coordinate = QtPositioning.coordinate(previousWP.coordinate.latitude,previousWP.coordinate.longitude + 0.0047 / Helper.getScale(map))
                }
            }
        }
    }

    Elevation{
        id: elevationFinder
        path: "Elevation"
    }

    MarkerList{
        id: lstMarkersSave
    }

    MarkerList{
        id: lstMarkers
    }

    Computer{
        id: cInfo
    }


    ListModel {
        id: _arrowModel
    }

    SymbolModel{
        id:_waypointModel
    }

    SymbolModel{
        id:_markerModel
    }

    SymbolModel{
        id:_trajactoryModel
    }

    Item{
        id:itemMap
        z:UIConstants.z_map
        anchors.fill: parent
    }

    Component
    {
        id:mapComponent
        Map {
            id:_mapComponent
            Plugin {
                id: mapPlugin
                name:"esri"
                PluginParameter{name:"esri.mapping.cache.disk.cost_strategy";value:"unitary";}  //use number of tile
                PluginParameter{name:"esri.mapping.cache.disk.size";value:"1000000";}           //use maximumm 1000000 tile
                PluginParameter{name:"esri.mapping.cache.directory";value:Helper.convertUrltoPath(urlMaps);}
                PluginParameter{name:"esri.mapping.minimumZoomLevel";value:"11";}
                PluginParameter{name:"esri.mapping.maximumZoomLevel";value:"17";}
            }

            property variant scaleLengths: [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000]
            //id:map
            //z:UIConstants.z_map
            maximumZoomLevel: 17
            minimumZoomLevel: 11
            anchors.fill: parent
            plugin: mapPlugin //createPluginMap("esri","/home/ttuav/mapGCS")//mapPlugin
            center: QtPositioning.coordinate(21.041606, 105.490983) // Oslo
            zoomLevel: 18
            activeMapType: supportedMapTypes[1]
            color: "black"
            copyrightsVisible: false

            onBearingChanged: {
                _mapComponent.bearing = 0;
            }
            Keys.onPressed: {
                if(event.key === Qt.Key_S){

                }else if(event.key === Qt.Key_Plus || event.key === Qt.Key_Equal){
                    zoomIn();
                }else if(event.key === Qt.Key_Minus || event.key === Qt.Key_Underscore){
                    zoomOut();
                }

                if(event.key === Qt.Key_C)
                    clearPlaneTrajactory();

                switch(event.key){
                case Qt.Key_0:  if(!waypointEditor.visible)showWaypointId(0); break;
                case Qt.Key_1:  if(!waypointEditor.visible)showWaypointId(1); break;
                case Qt.Key_2:  if(!waypointEditor.visible)showWaypointId(2); break;
                case Qt.Key_3:  if(!waypointEditor.visible)showWaypointId(3); break;
                case Qt.Key_4:  if(!waypointEditor.visible)showWaypointId(4); break;
                case Qt.Key_5:  if(!waypointEditor.visible)showWaypointId(5); break;
                case Qt.Key_6:  if(!waypointEditor.visible)showWaypointId(6); break;
                case Qt.Key_7:  if(!waypointEditor.visible)showWaypointId(7); break;
                case Qt.Key_8:  if(!waypointEditor.visible)showWaypointId(8); break;
                case Qt.Key_9:  if(!waypointEditor.visible)showWaypointId(9); break;
                case Qt.Key_Delete:
                    clearRuler()
                    break;
                default:
                    break;
                }
            }

            MapItemView{
                //            z:5
                id: _waypoint_item_view
                model: _waypointModel
                delegate: waypointcomponent

            }

            MapItemView{
                //            z:4
                id: _marker_item_view
                model: _markerModel
                delegate: markercomponent
            }

            MapItemView{
                // z:3
                id: _arrow_view
                model: _arrowModel
                delegate: arrowcomponent
            }


            Timer {
                id: scaleTimer
                interval: 300
                running: false
                repeat: false
                onTriggered: {
                    Helper.calculateScale(map,scaleLine,scaleImage,scaleImageLeft,map.scaleLengths,scaleText)
                }
            }
            Component.onCompleted: {
                //refreshItemMap();

            }

            Item {
                id: scaleLine
                z: 0
                visible: scaleText.text != "0 m"
                anchors.top: parent.top;
                anchors.left: parent.left
                anchors{topMargin: 40;leftMargin:110;}
                height: scaleText.height * 2
                width: scaleImage.width

                Image {
                    id: scaleImageLeft
                    source: "qrc:/assets/images/scale_end.png"
                    anchors.bottom: parent.bottom
                    anchors.right: scaleImage.left
                }
                Image {
                    id: scaleImage
                    source: "qrc:/assets/images/scale.png"
                    anchors.bottom: parent.bottom
                    anchors.right: scaleImageRight.left
                }
                Image {
                    id: scaleImageRight
                    source: "qrc:/assets/images/scale_end.png"
                    anchors.bottom: parent.bottom
                    anchors.right: parent.right
                }
                Text {
                    id: scaleText
                    color: "red"
                    anchors.centerIn: parent
                    text: "1 m"
                }
                Component.onCompleted: {
                    Helper.calculateScale(map,scaleLine,scaleImage,scaleImageLeft,map.scaleLengths,scaleText)
                }
            }

            onCenterChanged:{
                scaleTimer.restart()
            }

            onZoomLevelChanged:{
                scaleTimer.restart()
            }

            onWidthChanged:{
                scaleTimer.restart()
            }

            onHeightChanged:{
                scaleTimer.restart()
            }

            MouseArea {
                property bool mouse_position_changed : false
                property int old_point_x:0
                property int old_point_y:0
                property var buttonPressed:null
                acceptedButtons: Qt.LeftButton|Qt.RightButton

                anchors.fill: parent
                onPressed:{
                    var precoordinate = map.toCoordinate(Qt.point(mouse.x,mouse.y))
                    mouse_position_changed = false
                    old_point_x = mouse.x
                    old_point_y = mouse.y
                    setFocus(true);
                    selectedWP = undefined;
                    selectedMarker = undefined;
                    clickedLocation = precoordinate

                    if(UIConstants.mouseOnMapMode === UIConstants.mouseOnMapModeWaypoint && mouse.button === Qt.LeftButton){
                        coordinate_clicked=precoordinate
                        //console.log("lat:"+coordinate_clicked.latitude+" lon:"+coordinate_clicked.longitude)
                        _arrowModel.clear()
                        _arrowModel.append({arrow_lat : precoordinate.latitude, arrow_lon: precoordinate.longitude});
                        Helper.unselect_all(map,listsymbol)
                        selectedIndex = -1
                        selectedmarkerIndex = -1

                    }
                    else if(UIConstants.mouseOnMapMode === UIConstants.mouseOnMapModeMeasure && mouse.button === Qt.LeftButton)
                    {
                        rulercoord1 = precoordinate
                        map.gesture.enabled = false
                        buttonPressed = Qt.LeftButton
                        console.log("press Leftbutton")
                    }
                    else if(mouse.button === Qt.RightButton)
                    {
                        rulercoord1 = precoordinate
                        map.gesture.enabled = false
                        buttonPressed = Qt.RightButton
                    }
                    rootItem.mapClicked(true);

                }
                onClicked: {
                    //rootItem.mapClicked(true);

                }

                onPositionChanged:
                {
                    var npoint = normalizePoint(mouse.x,mouse.y)
                    var poscoordinate = map.toCoordinate(npoint)
                    console.log(mouse.button)
                    if(((UIConstants.mouseOnMapMode === UIConstants.mouseOnMapModeMeasure && buttonPressed === Qt.LeftButton)||(buttonPressed === Qt.RightButton))
                            && (Math.abs(mouse.x-old_point_x)>2||(Math.abs(mouse.y-old_point_y)>2)))
                    {
                        console.log("position change")
                        rulercoord2 = poscoordinate

                        addingRuler(rulercoord1,rulercoord2)
                        mouse_position_changed=true
                        old_point_x = mouse.x
                        old_point_y = mouse.y
                    }
                }
                onReleased:{
                    if((UIConstants.mouseOnMapMode === UIConstants.mouseOnMapModeMeasure && buttonPressed === Qt.LeftButton)&& mouse_position_changed === true)
                    {
                        acceptedRuler()
                        rectProfilePath.visible = true
                        profilePath.addElevation(
                                    cInfo.homeFolder()+"/ArcGIS/Runtime/Data/elevation/"+mapHeightFolder,
                                    rulercoord1,rulercoord2);

                        map.gesture.enabled = true
                    }
                    else if(buttonPressed === Qt.RightButton)
                    {
                        acceptedRuler()
                        map.gesture.enabled = true
                    }
                    buttonPressed = null
                }

            }

            Connections{
                target: _waypointModel
                onSymbolModelChanged:{
                    //                index_row_model_symbol=0
                    //                listsymbol = []
                    //                listwaypoint = []
                }
            }

            Connections{
                target: _markerModel
                onSymbolModelChanged:{
                    //                index_row_model_marker=0
                }
            }

            Connections{
                target: _trajactoryModel
                onSymbolModelChanged:{
                    //                index_row_model_trajactory=0
                    createListWPLine()
                }
            }
        }

    }
    Rectangle{
        id: rectProfilePath
        z:200
        width: UIConstants.sRect * 22
        height: UIConstants.sRect * 9
        anchors.centerIn: parent
        visible: false
        color: UIConstants.transparentBlue
        radius: UIConstants.rectRadius
        border.width: 1
        border.color: UIConstants.grayColor
        ProfilePath{
            id: profilePath
            title: "Profile Path"
            xName: "Distance (m)"
            yName: "Altitude (m)"
            fontSize: UIConstants.fontSize
            fontFamily: UIConstants.appFont
            anchors.fill: parent
            anchors.margins: 4
        }
    }


    Component {
        id: myComponent
        MissionItem{
            id: missionSample
        }
    }

    Component {
            id: arrowcomponent
            MouseArrow{
                z: UIConstants.z_mouseArrow
                coordinate: QtPositioning.coordinate(arrow_lat, arrow_lon)
                visible: true
            }
        }


    Component {
            id: waypointcomponent
            Symbol{
                z:UIConstants.z_waypoint
                id: _waypoint
                visible: true
                Component.onCompleted: {

                    var wp_Id     = Id_Role;
                    var wp_Type   = Type_Role;

                    var wp_Param1 = Param1_Role;
                    var wp_Param2 = Param2_Role;
                    var wp_Param3 = Param3_Role;
                    var wp_Param4 = Param4_Role;

                    var wp_Coord  = Coordinate_Role;

                    var wp_Alt    = wp_Coord.altitude;

                    _waypoint.wpId            = Number(wp_Id)
                    _waypoint.missionItemType = Number(wp_Type)
                    _waypoint.waypointAlt     = Number(wp_Alt)
                    _waypoint.param1          = Number(wp_Param1)
                    _waypoint.param2          = Number(wp_Param2)
                    _waypoint.param3          = Number(wp_Param3)
                    _waypoint.param4          = Number(wp_Param4)
                    _waypoint.coordinate      = wp_Coord
                    _waypoint.symbolId        = Number(wp_Id)

                    _waypoint.z = UIConstants.z_waypoint + _waypoint.wpId

                    if(_waypoint.wpId === currentWpIndex)
                        _waypoint.iscurrentWP = true

                    if(_waypoint.symbolId === selectedIndex)
                        _waypoint.isSelected = true

//                    console.log("add wp" + _waypoint.wpId)
                    listsymbol.push(_waypoint) //add waypoint to list symbol
                    listwaypoint[Id_Role]=_waypoint

                }


                MouseArea{
                    anchors.fill: _waypoint
                    hoverEnabled: true

                    property double mouseXold : 0
                    property int pressMouseX:0
                    property int pressMouseY:0
                    property bool positionChanged:false

                    onPressed:{
                        if(UIConstants.mouseOnMapMode === UIConstants.mouseOnMapModeWaypoint && (allowSelectSymbol || addingWpIndex === _waypoint.symbolId)){
                            Helper.unselect_all(map,listsymbol)
                            selectedmarkerIndex = -1
                            selectedIndex = -1
                            _waypoint.isSelected=true
                            selectedWP = _waypoint;
                            selectedIndex = _waypoint.symbolId
                            map.gesture.enabled=false  //unselect map
                            pressMouseX = mouse.x
                            pressMouseY = mouse.y
                            pressSymbol = true
                            rootItem.mapClicked(false);

                            var buffCoord = _waypoint.coordinate;
                            buffCoord.altitude = _waypoint.waypointAlt;
                            waypointEditor.changeCoordinate(buffCoord);

                            if(waypointEditor.visible)
                                waypointEditor.changeState()
                        }
                        else
                            mouse.accepted = false


                    }
                    onReleased:{
                        if(UIConstants.mouseOnMapMode === UIConstants.mouseOnMapModeWaypoint){
                            pressSymbol=false
                            if(positionChanged===true)
                            {
                                positionChanged=false
                                if(_waypoint.symbolId === 0) //home
                                {
                                    homePositionChanged(_waypoint.coordinate.latitude,_waypoint.coordinate.longitude,_waypoint.coordinate.altitude)
                                }
                                _waypointModel.moveSymbol(_waypoint.symbolId,_waypoint.coordinate)
                            }
                        }
                        _waypoint.stopTimerEditSymbol()
                    }

                    onPositionChanged:{
                        if(UIConstants.mouseOnMapMode === UIConstants.mouseOnMapModeWaypoint){
                            if(pressSymbol && (Math.abs(mouseX-pressMouseX)>3 || Math.abs(mouseY-pressMouseY)>3) && _waypoint.missionItemType !== UIConstants.dojumpType)
                            {
                                var npoint = normalizePoint(_waypoint.x+mouseX,_waypoint.y+mouseY)
                                _waypoint.coordinate = map.toCoordinate(npoint)
                                _trajactoryModel.moveSymbol(_waypoint.symbolId,_waypoint.coordinate)
                                symbolMoving(_waypoint.coordinate.latitude , _waypoint.coordinate.longitude)
                                positionChanged=true
                                pressMouseX = mouse.x
                                pressMouseY = mouse.y

                                //waypointEditor.changeASL(_waypoint.waypointAlt);
                                var buffCoord = _waypoint.coordinate;
                                buffCoord.altitude = _waypoint.waypointAlt;
                                waypointEditor.changeCoordinate(buffCoord);
                                waypointEditor.changeState();

                                isMapSync = false;
                            }
                        }

                    }

                    onPressAndHold: {
                        if(!waypointEditor.visible && positionChanged === false)
                            _waypoint.startTimerEditSymbol();
                    }

                }

                onShowSymbolEditor:{
                    waypointEditor.visible = true;
                    waypointEditor.changeState();
                }
            }
        }

    Component {
            id: markercomponent
            Symbol{
                z:5
                id: _marker
                visible: true
                isMarker: true
                Component.onCompleted: {

                    var _id_marker  = Id_Role
                    var _type       = Type_Role
                    var _coordinate = Coordinate_Role
                    var _text       = Text_Role

                    _marker.markerType  = Number(_type)
                    _marker.symbolId    = _id_marker
                    _marker.coordinate  = _coordinate
                    _marker.textMarker  = _text

                    if(_marker.symbolId === selectedmarkerIndex )
                        _marker.isSelected = true

                    listsymbol.push(_marker) //add marker to list symbol
                    listmarker[Id_Role] = _marker
                }

                MouseArea{
                    anchors.fill: _marker
                    hoverEnabled: true

                    property double mouseXold : 0
                    property int pressMouseX:0
                    property int pressMouseY:0
                    property bool positionChanged:false

                    onPressed:{
                        if(UIConstants.mouseOnMapMode === UIConstants.mouseOnMapModeWaypoint){
                            Helper.unselect_all(map,listsymbol)
                            selectedmarkerIndex = -1
                            selectedIndex = -1
                            _marker.isSelected=true
                            selectedmarkerIndex = _marker.symbolId
                            map.gesture.enabled=false  //unselect map
                            pressMouseX = mouse.x
                            pressMouseY = mouse.y
                            pressSymbol=true;
                            positionChanged=false

                            selectedMarker = _marker;


                            var buffCoord = _marker.coordinate;
                            //buffCoord.altitude = _marker.waypointAlt;
                            markerEditor.changeCoordinate(buffCoord);

                            if(markerEditor.visible)
                                markerEditor.changeState()
                        }

                    }
                    onReleased:{
                        if(UIConstants.mouseOnMapMode === UIConstants.mouseOnMapModeWaypoint){
                            pressSymbol=false
                            if(positionChanged===true)
                                _markerModel.moveSymbol(_marker.symbolId,_marker.coordinate)
                        }
                        _marker.stopTimerEditSymbol()
                    }

                    onPositionChanged:{
                        if(UIConstants.mouseOnMapMode === UIConstants.mouseOnMapModeWaypoint){
                            if(pressSymbol && (Math.abs(mouseX-pressMouseX)>4 || Math.abs(mouseY-pressMouseY)>4) && markerEditor.visible === true)
                            {
                                var npoint = normalizePoint(_marker.x+mouseX,_marker.y+mouseY);
                                _marker.coordinate = map.toCoordinate(npoint);
                                positionChanged=true;

                                var buffCoord = _marker.coordinate;
                                buffCoord.altitude = 0;
                                markerEditor.changeCoordinate(buffCoord);
                                markerEditor.changeState();

                            }
                        }
                    }
                    onPressAndHold: {
                        if(!markerEditor.visible && positionChanged === false)
                            _marker.startTimerEditSymbol();
                    }
                }
                onShowSymbolEditor:{
                    markerEditor.visible = true;
                    markerEditor.changeState();
                }
            }
        }


    WaypointEditor{
        id: waypointEditor
        x: parent.height + 10
        property int itemType:0

        visible: false

        function updateWPEditorLocation(coord)
        {
            var p = Helper.convert_coordinator2screen(coord,map)
            var px = p.x;
            var py = p.y;
            x = (rootItem.width - px > waypointEditor.width) ? px + 20 : px - 20 - waypointEditor.width;
            y = (rootItem.height - py > waypointEditor.height) ? py: py - waypointEditor.height;
        }

        function changeState(){
            updateWPEditorLocation(selectedWP.coordinate)
            var lat = selectedWP.coordinate.latitude
            var lon = selectedWP.coordinate.longitude
            var asl = elevationFinder.getAltitude(
                        cInfo.homeFolder()+"/ArcGIS/Runtime/Data/elevation/"+mapHeightFolder,
                        lat,lon);
            waypointEditor.changeASL(asl);

            itemType = selectedWP.missionItemType
            //nhatdn1 comment
            var waypointCoordinate = QtPositioning.coordinate(
                        selectedWP.coordinate.latitude,
                        selectedWP.coordinate.longitude,
                        selectedWP.waypointAlt);
            waypointEditor.loadInfo(waypointCoordinate,
                                    selectedWP.missionItemType,
                                    selectedWP.param1,
                                    selectedWP.param2,
                                    selectedWP.param3,
                                    selectedWP.param4);
            waypointModeEnabled = (selectedWP.wpId !== 0);
        }

        onConfirmClicked: {
            if(rootItem.selectedWP!== undefined)
            {
                acceptEditWP(selectedIndex,QtPositioning.coordinate(latitude,longitude,agl),itemType,param1,param2,param3,param4)
            }
            waypointEditor.visible = false;
            isMapSync = false;
        }
        onCancelClicked: {
            rootItem.restoreWP();
            waypointEditor.visible = false;
        }
        onWaypointModeChanged: {
            if(selectedWP !== undefined){
                console.log("waypointMode = "+waypointMode);
                itemType = Number(lstWaypointCommand[vehicleType][waypointMode]["COMMAND"]);
            }

        }
    }

    MarkerEditor{
        id: markerEditor
        x: parent.height + 10
        visible: false
        property int markerSelectedType: 0
        property string markerSelectedTypeStr: "MARKER_DEFAULT"
        property string markerSelectedText: "default"

        function updateMarkerEditorLocation(coord)
        {
            var p = Helper.convert_coordinator2screen(coord,map)
            var px = p.x;
            var py = p.y;
            x = (rootItem.width - px > width) ? px + 20 : px - 20 - width;
            y = (rootItem.height - py > height) ? py: py - height;
        }

        function changeState(){
            if(selectedMarker!= undefined){
                updateMarkerEditorLocation(selectedMarker.coordinate)
                var lat = selectedMarker.coordinate.latitude
                var lon = selectedMarker.coordinate.longitude
                var altitude = 0;
                var markerCoordinate = QtPositioning.coordinate(lat,lon,altitude);
                var asl = elevationFinder.getAltitude(
                            cInfo.homeFolder()+"/ArcGIS/Runtime/Data/elevation/"+mapHeightFolder,lat,lon);
                markerEditor.asl = asl;
                switch(selectedMarker.markerType)
                {
                    case 0:
                        markerSelectedTypeStr = "MARKER_DEFAULT";
                        break;
                    case 1:
                        markerSelectedTypeStr = "MARKER_TANK";
                        break;
                    case 2:
                        markerSelectedTypeStr = "MARKER_PLANE";
                        break;
                    case 3:
                        markerSelectedTypeStr = "MARKER_TARGET";
                        break;
                    case 4:
                        markerSelectedTypeStr = "MARKER_SHIP";
                        break;
                    default:
                        break;
                }
                markerSelectedType = selectedMarker.markerType;

                markerEditor.loadInfo(markerCoordinate, markerSelectedTypeStr, selectedMarker.textMarker);

            }
        }

        onMarkerIDChanged: {
            //changeMarkerType(selectedMarker,markerType);
            switch(markerType)
            {
                case "MARKER_DEFAULT":
                    markerSelectedType = 0;
                    break;
                case "MARKER_TANK":
                    markerSelectedType = 1;
                    break;
                case "MARKER_PLANE":
                    markerSelectedType = 2;
                    break;
                case "MARKER_TARGET":
                    markerSelectedType = 3;
                    break;
                case "MARKER_SHIP":
                    markerSelectedType = 4;
                    break;
                default:
                    break;
            }
        }
        onConfirmClicked: {
            if(rootItem.selectedMarker!=undefined){
                acceptEditMarker(selectedmarkerIndex,coordinate,markerSelectedType,markerSelectedText)
                selectedMarker = undefined;
            }
            markerEditor.visible = false;
        }
        onCancelClicked: {
            restoreMarker();
            markerEditor.visible = false;
        }
        onTextChanged: {
            markerSelectedText = newText
        }
    }

    function createWPLine(coordinate1,coordinate2)
    {
        var component = Qt.createComponent("qrc:/CustomViews/SubComponents/Trajactory.qml");
        var trajactory_object = component.createObject(map);//, {coord1: coordinate1,coord2: coordinate2});
        if (trajactory_object === null) {
            // Error Handling
            console.log("Error creating object");
            return null;
        }
        else
        {
            map.addMapItem(trajactory_object)
            trajactory_object.addCoordinate(coordinate1)
            trajactory_object.addCoordinate(coordinate2)
            return trajactory_object
        }
    }

    function createListWPLine()
    {
        clearListWPLine()
        for(var index_row_model_trajactory = 0 ; index_row_model_trajactory <_trajactoryModel.rowCount() ; index_row_model_trajactory++)
        {
            var _type_symbol = _trajactoryModel.get(index_row_model_trajactory).Type_Role
            var _id_symbol = _trajactoryModel.get(index_row_model_trajactory).Id_Role
            if(_id_symbol>0 && (_type_symbol === UIConstants.waypointType || _type_symbol === UIConstants.loitertimeType || _type_symbol === UIConstants.takeoffType|| _type_symbol === UIConstants.landType|| _type_symbol === UIConstants.vtoltakeoffType|| _type_symbol === UIConstants.vtollandType)) //waypoint
            {
                for(var i= index_row_model_trajactory+1 ;i < _trajactoryModel.rowCount() ; i++)
                {
                    var _id_next_symbol = _trajactoryModel.get(i).Id_Role
                    var _type_next_symbol= _trajactoryModel.get(i).Type_Role
                    if(_id_next_symbol>0 && (_type_next_symbol === UIConstants.waypointType || _type_next_symbol === UIConstants.loitertimeType|| _type_next_symbol === UIConstants.takeoffType|| _type_next_symbol === UIConstants.landType|| _type_next_symbol === UIConstants.vtoltakeoffType|| _type_next_symbol === UIConstants.vtollandType))
                    {
                        var _coord1=_trajactoryModel.get(index_row_model_trajactory).Coordinate_Role
                        var _coord2=_trajactoryModel.get(i).Coordinate_Role
                        var obj = createWPLine(_coord1,_coord2)
                        listWPLine.push(obj)
                        break;
                    }
                }
             }
             if(index_row_model_trajactory+1===_trajactoryModel.rowCount())
             {
                 for(i = 0 ;i < _trajactoryModel.rowCount() ; i++)
                 {
                      _type_symbol = _trajactoryModel.get(i).Type_Role
                      _id_symbol = _trajactoryModel.get(i).Id_Role
                      if(_id_symbol>0 && (_type_symbol === UIConstants.waypointType || _type_symbol ===UIConstants.loitertimeType|| _type_symbol === UIConstants.takeoffType|| _type_symbol === UIConstants.landType|| _type_symbol === UIConstants.vtoltakeoffType|| _type_symbol === UIConstants.vtollandType))
                      {
                          _coord1 = _trajactoryModel.get(i).Coordinate_Role
                          break;
                      }
                 }

                 for(i = _trajactoryModel.rowCount() - 1 ;i > 0 ; i--)
                 {
                      _type_next_symbol = _trajactoryModel.get(i).Type_Role
                      _id_next_symbol = _trajactoryModel.get(i).Id_Role
                      if(_id_next_symbol>0 && (_type_next_symbol === UIConstants.waypointType || _type_next_symbol ===UIConstants.loitertimeType|| _type_next_symbol === UIConstants.takeoffType|| _type_next_symbol === UIConstants.landType|| _type_next_symbol === UIConstants.vtoltakeoffType|| _type_next_symbol === UIConstants.vtollandType))
                      {
                         _coord2 = _trajactoryModel.get(i).Coordinate_Role
                          break;
                      }
                 }

                 obj = createWPLine(_coord1,_coord2)
                 if(obj !== null && obj !== undefined)
                    listWPLine.push(obj)
             }
        }
    }

    function clearListWPLine()
    {
        for(var i=0 ; i<listWPLine.length ; i++)
        {
            var obj = listWPLine[i]
            obj.destroy()
        }
        listWPLine=[]
    }



    function addWP(index)
    {
        if(UIConstants.mouseOnMapMode === UIConstants.mouseOnMapModeWaypoint)
        {
            Helper.unselect_all(map,listsymbol)
            selectedIndex = -1
            var coord = coordinate_clicked;
            coord.altitude = 100;
            _waypointModel.addSymbol(index,UIConstants.waypointType,0,0,0,0,"",coord)
            _trajactoryModel.addSymbol(index,UIConstants.waypointType,0,0,0,0,"",coord)
//            if(index===0)
//                rollbackhomeposition=coordinate_clicked

            isMapSync = false;
        }
    }

    function addWPPosition(index,position,command,param1,param2,param3,param4){ //id:0 =>Home
        // add Waypoint
        console.log("id:"+index)
        selectedIndex = -1

        _waypointModel.addSymbol(index,command,param1,param2,param3,param4,"",position)
        _trajactoryModel.addSymbol(index,command,0,0,0,0,"",position)

        if(index===0)
            rollbackhomeposition=position
    }

    function acceptEditWP(index,position,command,param1,param2,param3,param4)
    {
        console.log("accept edit wp id:"+index)
        _waypointModel.editSymbol(index,command,param1,param2,param3,param4,"",position)
        _trajactoryModel.editSymbol(index,command,0,0,0,0,"",position)

//        if(index===0)
//            rollbackhomeposition=position

        console.log("alt wp:"+ position.altitude)
        if(index === 0 && selectedWP !== null && selectedWP !== undefined){
            if(position.altitude !== selectedWP.coordinate.altitude){
//                homePositionChanged(selectedWP.coordinate.latitude,
//                                selectedWP.coordinate.longitude,
//                                position.altitude);
                vehicle.setAltitudeRTL(position.altitude)
                console.log("change altitude: "+position.altitude)
            }
        }

        isMapSync = false;
    }

    function restoreWP(){
        _waypointModel.refreshModel();
        _trajactoryModel.refreshModel();
    }

    function changeWPPosition(index,position)
    {
        _waypointModel.moveSymbol(index,position)
        _trajactoryModel.moveSymbol(index,position)
    }

    function lastWPIndex()
    {
        return _waypointModel.rowCount() - 1
    }

    function removeWP(index)
    {
        console.log("remove"+index)
        if(index>-1)
        {
            listsymbol = []
            listwaypoint = []
             _waypointModel.deleteSymbol(index)
             _trajactoryModel.deleteSymbol(index)
            isMapSync = false;
        }
    }

    function clearWPs()
    {
        _waypointModel.clearSymbol()
        _trajactoryModel.clearSymbol()
        listwaypoint.length = 0
    }

    function getCurrentListWaypoint(){
        var currentListWaypoint = [];
        var startIndex = 0;

        for(var id = 0; id < _waypointModel.rowCount() ; id ++){
                var wp_Id     = _waypointModel.get(id).Id_Role
                var wp_Type   = _waypointModel.get(id).Type_Role
                var wp_Param1 = _waypointModel.get(id).Param1_Role
                var wp_Param2 = _waypointModel.get(id).Param2_Role
                var wp_Param3 = _waypointModel.get(id).Param3_Role
                var wp_Param4 = _waypointModel.get(id).Param4_Role
                var wp_Coord  = _waypointModel.get(id).Coordinate_Role

                var missionItem = myComponent.createObject(rootItem);

                missionItem.sequence = wp_Id;
                missionItem.command  = wp_Type;
                missionItem.frame    = wp_Id === 0 ? 0:3;

                missionItem.param1 = wp_Param1;
                missionItem.param2 = wp_Param2;
                missionItem.param3 = wp_Param3;
                missionItem.param4 = wp_Param4;

                missionItem.param5 = wp_Coord.latitude
                missionItem.param6 = wp_Coord.longitude
                missionItem.param7 = wp_Coord.altitude;

                currentListWaypoint.push(missionItem)
        }
        return currentListWaypoint;
    }

    function createMarkerSymbol(coordinate,type,text)
    {
        if(UIConstants.mouseOnMapMode === UIConstants.mouseOnMapModeWaypoint)
        {
            Helper.unselect_all(map,listsymbol)
            var lastmMarkerIndex =  _markerModel.rowCount()
            switch(type)
            {
                case "MARKER_DEFAULT":
                    _markerModel.addSymbol(lastmMarkerIndex,0,0,0,0,0,text,coordinate)
                    break;
                case "MARKER_TANK":
                    _markerModel.addSymbol(lastmMarkerIndex,1,0,0,0,0,text,coordinate)
                    break;
                case "MARKER_PLANE":
                    _markerModel.addSymbol(lastmMarkerIndex,2,0,0,0,0,text,coordinate)
                    break;
                case "MARKER_TARGET":
                    _markerModel.addSymbol(lastmMarkerIndex,3,0,0,0,0,text,coordinate)
                    break;
                case "MARKER_SHIP":
                    _markerModel.addSymbol(lastmMarkerIndex,4,0,0,0,0,0,text,coordinate)
                    break;
                default:
                    break;
            }
        }
    }



    function zoomIn(){
        var maxZoom = map.maximumZoomLevel

        var zoom=map.zoomLevel+1
        if(zoom>20)
            zoom=20
        map.zoomLevel = zoom
        map.center = (selectedWP===undefined)? coordinate_clicked:selectedWP.coordinate;
    }

    function zoomOut(){
        var minZoom = map.minimumZoomLevel
        var zoom=map.zoomLevel-1
        if(zoom<11)
            zoom=11
        map.zoomLevel = zoom
        map.center = (selectedWP===undefined)? coordinate_clicked:selectedWP.coordinate;
    }

    function focusAllObject(){
        if(plane.visible === true)
        {
            map.center = plane.coordinate
            map.zoomLevel = map.minimumZoomLevel
        }
    }

    function updatePlane(position){

        if(plane){
            plane.visible = true
            plane.coordinate = QtPositioning.coordinate(position.latitude,position.longitude)

            createPlaneTrajactory(position)
            if(planeTrajactory !== null)
            {
            if(planeTrajactory.pathLength() > 5000)
                planeTrajactory.removeCoordinate(0);
            }
//            if(lastPlanePosition === null)
//            {
//                lastPlanePosition = position
//            }
//            else if(position!==lastPlanePosition)
//            {
//                var _trajactory = createPlaneTrajactory(lastPlanePosition,position)
//                listPlaneTrajactory.push(_trajactory)
//                lastPlanePosition = position
//            }
//            else if(listPlaneTrajactory.length > 1000)
//            {
//                var lastedObject = listPlaneTrajactory[0]
//                lastedObject.destroy()
////                map.removeMapItem(lastedObject)
//                listPlaneTrajactory.splice(0,1)
//            }
        }
        else
        {
            var component = Qt.createComponent("qrc:/CustomViews/SubComponents/Plane.qml");
            var planeObject = component.createObject(map,{z:UIConstants.z_plane,visible: true});
            if (planeObject === null) {
                // Error Handling
                console.log("Error creating object");
            }
            else {
                map.addMapItem(planeObject)
                planeObject.z=UIConstants.z_plane
                plane =  planeObject
            }
        }

    }


    function updateHeadingPlane(heading)
    {
        if(plane)
            plane.heading = heading
    }

    function changeVehicleType(vehicleType){
        console.log("changeVehicleType to "+vehicleType);
        console.log("vehicle.MAV_TYPE_FIXED_WING="+Vehicle.MAV_TYPE_FIXED_WING);
        var vehicleSymbolUrl = vehicleSymbolLink["MAV_TYPE_GENERIC"];
//        var vehicleHeading = uavGraphic.symbol.angle;
//        var opacity = uavGraphic.symbol.opacity;
        switch(vehicleType){
        case 2:
            rootItem.vehicleType = "MAV_TYPE_QUADROTOR";
            waypointEditor.vehicleType = rootItem.vehicleType;
            vehicleSymbolUrl = vehicleSymbolLink["MAV_TYPE_QUADROTOR"];
            break;
        case 3:
            rootItem.vehicleType = "MAV_TYPE_COAXIAL";
            waypointEditor.vehicleType = rootItem.vehicleType;
            vehicleSymbolUrl = vehicleSymbolLink["MAV_TYPE_COAXIAL"];
            break;
        case 20:
            rootItem.vehicleType = "MAV_TYPE_VTOL_QUADROTOR";
            waypointEditor.vehicleType = rootItem.vehicleType;
            vehicleSymbolUrl = vehicleSymbolLink["MAV_TYPE_VTOL_QUADROTOR"];
            break;
        case 1:
            rootItem.vehicleType = "MAV_TYPE_FIXED_WING";
            waypointEditor.vehicleType = rootItem.vehicleType;
            vehicleSymbolUrl = vehicleSymbolLink["MAV_TYPE_VTOL_QUADROTOR"];
            break;
        }

        plane.planeSource = vehicleSymbolUrl;
    }

    function createPlaneTrajactory(coord)
    {
        if(planeTrajactory === null){
            var component = Qt.createComponent("qrc:/CustomViews/SubComponents/PolyTrajactory.qml");
            var object = component.createObject(map, {z:UIConstants.z_tracjactoryPlane,linewidth: 2,color:UIConstants.planeTrajactoryColor});
            if (object === null) {
                // Error Handling
                console.log("Error creating object");
                return;
            }
            else
            {
                map.addMapItem(object)
                planeTrajactory = object;
                //return object
            }
        }else{
            planeTrajactory.addCoordinate(coord);
        }
    }

    function clearPlaneTrajactory()
    {
//        for(var i=0;i<listPlaneTrajactory.length;i++)
//        {
//            var object = listPlaneTrajactory[i]
//            object.destroy()
////            map.removeMapItem(object)
//        }
//        listPlaneTrajactory=[]
        if(planeTrajactory!== null)
            planeTrajactory.destroy();
    }


    function changeCurrentWP(index)
    {
        currentWpIndex=index;
        if(currentWpIndex !== old_currentWpIndex)
        {
            old_currentWpIndex = currentWpIndex
            _waypointModel.refreshModel()
        }
    }

    function removeSelectedMarker()
    {
        console.log("remove marker" + selectedmarkerIndex)
        if( selectedmarkerIndex >-1)
        {
            listmarker = []
            _markerModel.deleteSymbol(selectedmarkerIndex)
        }
        //unselect marker with id
        selectedmarkerIndex = -1;
    }

    function changeHomePosition(homePosition)
    {
        changeWPPosition(0,homePosition);
        _waypointModel.refreshModel();
        rollbackhomeposition = homePosition;
//        console.log("home alt changed: "+homePosition.latitude);
    }

    function rollback_homePosition()
    {
        changeWPPosition(0,rollbackhomeposition)
    }

    function showWaypointId(index)
    {
        if(index<listwaypoint.length)
        {
            selectedIndex = index;
            Helper.unselect_all(map,listsymbol)
            listwaypoint[index].isSelected = true

//            console.log("index"+index)
//            console.log("idshow"+listwaypoint[index].wpId)
            map.center=listwaypoint[index].coordinate

            rootItem.mapClicked(false);
            selectedWP = listwaypoint[index];
        }
    }

    function showNextWP()
    {
        if(selectedIndex<listwaypoint.length-1)
            showWaypointId(selectedIndex+1)
        else
            showWaypointId(0)
    }

    function normalizePoint(x,y)
    {
        var p = Qt.point(x,y)
        if(x>map.width-3)
            p.x=map.width-3
        else if(x<3)
            p.x=3

        if(y>map.height-3)
            p.y=map.height-3
        else if(y<3)
            p.y=3

        return p
    }

    function convertLocationToScreen(lat,lon)
    {
        var p= Helper.convert_coordinator2screen(QtPositioning.coordinate(lat,lon),map)
        return p;
    }

    function hideProfilePath()
    {
        rectProfilePath.visible = false;
    }

    function createRuler(coordinate1,coordinate2) {
        var component = Qt.createComponent("qrc:/CustomViews/SubComponents/Ruler2.qml");
        var rulerObject = component.createObject(map, {coord1: coordinate1,coord2: coordinate2,uavmap:map});
        if (rulerObject === null) {
            // Error Handling
            console.log("Error creating object");
            return null;
        }
        else {
            map.addMapItem(rulerObject)
            rulerObject.z=UIConstants.z_ruler
            return rulerObject
        }
    }

    function addingRuler(coordinate1,coordinate2){
        if(listRuler.length>rulerCount){
            var object = listRuler[rulerCount]
            object.destroyChildOject();
            object.destroy();
//            map.removeMapItem(object)
            listRuler.splice(rulerCount,1)
        }
            var rulerObject = createRuler(coordinate1,coordinate2)
            if(rulerObject !== null && rulerObject !== undefined)
                listRuler[rulerCount] = rulerObject;
    }

    function acceptedRuler()
    {
        var rulerObject = listRuler[rulerCount]
        if(rulerObject !== null && rulerObject !== undefined)
            rulerCount = rulerCount + 1
    }

    function clearRuler()
    {
        for(var i=0;i<listRuler.length;i++)
        {
            try{
                var object = listRuler[i]
                object.destroyChildOject();
                object.destroy();
//                map.removeMapItem(object)
            }
            catch(ex){
                console.log(ex)
            }
        }
        rulerCount=0
        listRuler=[]
    }


    function focusMap()
    {
        map.focus = true  //important
        console.log("-------------map focus------------");
    }

    function setFocus(enable){
        map.focus = enable;
    }

    Component.onCompleted:
    {
          focusMap()
//        var msg = {}
//        myWorker.sendMessage(msg);
    }


    //tracker update
    function updateTracker(position){
        if(tracker)
        {
            tracker.visible = true
            tracker.coordinate = position;
        }
        else
        {
            var component = Qt.createComponent("qrc:/CustomViews/SubComponents/Tracker.qml");
            var trackerObject = component.createObject(map,{z:UIConstants.z_plane,visible: true});
            if (trackerObject === null) {
                // Error Handling
                console.log("Error creating object");
            }
            else {
                map.addMapItem(trackerObject)
                trackerObject.z=UIConstants.z_plane
                tracker =  trackerObject
            }
        }
    }

    function updateHeadingTracker(angle){
        if(tracker)
            tracker.angle = angle;
    }

    //UC postion
    function focusOnPosition(lat,lon)
    {
        map.center=QtPositioning.coordinate(lat,lon)
    }

    //marker
    function addMarker(){
        createMarkerSymbol(coordinate_clicked,"MARKER_DEFAULT","default");
    }

    function removeMarker(){
        removeSelectedMarker();
    }

    function acceptEditMarker(index,position,command,text)
    {
        console.log("accept edit marker id:"+index)
        _markerModel.editSymbol(index,command,0,0,0,0,text,position)
    }

    function restoreMarker(){
        _markerModel.refreshModel();
    }

    function clearMarkers(){
        _markerModel.clearSymbol()
        selectedMarker = undefined;
        listmarker = [];
    }

    function loadMarker(_name){
        lstMarkers.loadMarkers(_name);
        clearMarkers();
        selectedmarkerIndex = -1
        for(var i=0; i< lstMarkers.numMarker();i++){
            var marker = lstMarkers.getMarker(i);
            createMarkerSymbol(QtPositioning.coordinate(marker.latitude,marker.longtitude),marker.markerType,marker.description);
        }
    }
    function saveMarker(_name){
        lstMarkersSave.cleanMarker();
        for(var i=0; i< listmarker.length; i++){
            var marker = listmarker[i];
            var coord = marker.coordinate;
            var marker_type = "";
            switch(marker.markerType)
            {
                case 0:
                    marker_type = "MARKER_DEFAULT";
                    break;
                case 1:
                    marker_type = "MARKER_TANK";
                    break;
                case 2:
                    marker_type = "MARKER_PLANE";
                    break;
                case 3:
                    marker_type = "MARKER_TARGET";
                    break;
                case 4:
                    marker_type = "MARKER_SHIP";
                    break;
                default:
                    break;
            }
            lstMarkersSave.insertMarker(coord.latitude.toString(),
                                            coord.longitude.toString(),
                                            marker_type,
                                            marker.textMarker);
        }
        lstMarkersSave.saveMarkers(_name);
    }
    //waypoint
    function focusOnWP(id){
        showWaypointId(id)
    }

    function moveWPWithID(id,position){
        changeWPPosition(id,position)
    }

    //ham nay de tao 1 diem wp luc go to here dung cho quad
    function createGotoherePoint(position)
    {
        var component = Qt.createComponent("qrc:/CustomViews/SubComponents/Symbol.qml");
        var pointObject = component.createObject(map, {coordinate: position,missionItemType: UIConstants.waypointType,
                                                  wpId:-1,symbolId:-1});
        if (pointObject === null) {
            // Error Handling
            console.log("Error creating object");
            return;
        }
        else {
            map.addMapItem(pointObject)
            pointObject.z=UIConstants.z_gotohere
            return pointObject
        }
    }

    function changeClickedPosition(position,visible){
        if(gotohereSymbol === null){
            gotohereSymbol = createGotoherePoint(position)
        }
        else
            gotohereSymbol.coordinate = position

        gotohereSymbol.visible = visible
    }

    function setMap(mapData){
        urlMaps = mapFolderPath  + mapData
        createMap()
//        //map.refresh()
//        map.plugin = createPluginMap("esri",Helper.convertUrltoPath(urlMaps))
//        //console.log("include map:" + urlMap)
//        console.log("map: " + Helper.convertUrltoPath(urlMaps));
    }

    function createMap()
    {
        clearListWPLine()
        //var plugin = createPluginMap("esri",Helper.convertUrltoPath(urlMaps))

        var zoomLevel = null
        var tilt = null
        var bearing = null
        var fov = null
        var center = null
        var panelExpanded = null
        if (map) {
            zoomLevel = map.zoomLevel
            tilt = map.tilt
            bearing = map.bearing
            fov = map.fieldOfView
            center = map.center
            //panelExpanded = map.slidersExpanded
            map.destroy()
        }

        map = mapComponent.createObject(itemMap);
        map.z = UIConstants.z_map;
        //map.plugin = plugin;

        if (zoomLevel != null) {
            map.tilt = tilt
            map.bearing = bearing
            map.fieldOfView = fov
            map.zoomLevel = zoomLevel
            map.center = center
            //map.slidersExpanded = panelExpanded
        } else {
            // Use an integer ZL to enable nearest interpolation, if possible.
            map.zoomLevel = Math.floor((map.maximumZoomLevel - map.minimumZoomLevel)/2)
            // defaulting to 45 degrees, if possible.
            map.fieldOfView = Math.min(Math.max(45.0, map.minimumFieldOfView), map.maximumFieldOfView)
        }
        refreshItemMap()
        map.forceActiveFocus()
    }

    function refreshItemMap()
    {
        _markerModel.refreshModel();
        _waypointModel.refreshModel();
        _trajactoryModel.refreshModel();
    }

    function updateMouseOnMap(){

    }


    //target polygon
    function updateTargetPolygon(coord1,coord2,coord3,coord4){
        if(targetPolygon){
            targetPolygon.destroy()

            // targetPolygon.coordinate = position;
        }

        var component = Qt.createComponent("qrc:/CustomViews/SubComponents/TargetPolygon.qml");
        var targetPolygonObject = component.createObject(map,{z:UIConstants.z_targetPolygon,smooth:true,antialiasing:true,visible: true});
        if (targetPolygonObject === null) {
            // Error Handling
            console.log("Error creating object");
        }
        else {
            map.addMapItem(targetPolygonObject)
            targetPolygonObject.z=UIConstants.z_targetPolygon
            targetPolygon =  targetPolygonObject
            targetPolygon.addCoordinate(coord1)
            targetPolygon.addCoordinate(coord2)
            targetPolygon.addCoordinate(coord3)
            targetPolygon.addCoordinate(coord4)
        }
    }

    function updateOpticalLine(coord1,coord2){
        if(opticalLine)
            opticalLine.destroy()

        var component = Qt.createComponent("qrc:/CustomViews/SubComponents/PolyTrajactory.qml");
        var object = component.createObject(map, {z:UIConstants.z_targetPolygon,linewidth: 2,color:"black"});
        if (object === null) {
            // Error Handling
            console.log("Error creating object");
            return;
        }
        else
        {
            map.addMapItem(object)
            opticalLine = object;
            opticalLine.addCoordinate(coord1)
            opticalLine.addCoordinate(coord2)
        }
    }

    function drawTargetLocalization(p1,p2,p3,p4,centerPos,uavPos){
        updateTargetPolygon(p1,p2,p3,p4);
        updateOpticalLine(centerPos,uavPos);
    }

    //end target polygon
}
