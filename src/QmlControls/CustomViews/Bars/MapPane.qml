//------------------ Include QT libs ------------------------------------------
import QtQuick.Window 2.2
import QtQuick 2.6
import QtQuick.Controls 2.1
import QtQuick.Layouts 1.3
import QtGraphicalEffects 1.0
import QtQuick 2.0
import QtPositioning 5.3
//------------------ Include Custom modules/plugins
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0
import CustomViews.SubComponents 1.0
import CustomViews.Dialogs 1.0
//import CustomViews.MapComponents 1.0

//import map
import QtPositioning 5.3
import QtSensors 5.3
import Esri.ArcGISRuntime 100.0
import Esri.ArcGISExtras 1.1
import QtPositioning 5.0
import "qrc:/Maplib/transform.js" as Conv
import io.qdt.dev 1.0
//---------------- Component definition ---------------------------------------
Item {
    id: rootItem
    width: UIConstants.sRect*67
    height: UIConstants.sRect*32
    clip: true
    property bool isMapSync: false
    property bool mousePressed: false
    property int mouseButton: 0
    property bool ctrlPress: false
    property string mapName: "Layers.tpk"
    property string mapHeightFolder: "ElevationData-H1"
    property url dataPath: System.userHomePath + "/ArcGIS/Runtime/Data/"
    property var listVideoTab: []
    property var listDrones: []
    property var listCenterCommander: []
    property var listWaypoint: []
    property var zOrder: [0,1,2,3]
    property var startLine: []
    property int lineID: 0
    property int selectedIndex: -1
    property var selectedWP
    property var clickedLocation: QtPositioning.coordinate(0,0,0)
    property string vehicleType: "MAV_TYPE_GENERIC"
    property var lstWaypointCommand:{
        "MAV_TYPE_GENERIC":{
            "WAYPOINT":{
                "COMMAND": 16,
                "PARAM1":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM2":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM3":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM4":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
            },
            "LOITER":{
                "COMMAND": 19,
                "PARAM1":{
                    "LABEL":"TIME(S)",
                    "EDITABLE":true,
                },
                "PARAM2":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM3":{
                    "LABEL":"RADIUS",
                    "EDITABLE":true,
                },
                "PARAM4":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
            },
            "LAND":{
                "COMMAND": 21,
                "PARAM1":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM2":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM3":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM4":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
            },
            "TAKEOFF":{
                "COMMAND": 22,
                "PARAM1":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM2":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM3":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM4":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
            },
            "DO_JUMP":{
                "COMMAND": 177,
                "PARAM1":{
                    "LABEL":"SEQUENCE",
                    "EDITABLE":true,
                },
                "PARAM2":{
                    "LABEL":"REPEAT",
                    "EDITABLE":true,
                },
                "PARAM3":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM4":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
            }
        },
        "MAV_TYPE_QUADROTOR":{
            "WAYPOINT":{
                "COMMAND": 16,
                "PARAM1":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM2":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM3":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM4":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
            },
            "LOITER":{
                "COMMAND": 19,
                "PARAM1":{
                    "LABEL":"TIME(S)",
                    "EDITABLE":true,
                },
                "PARAM2":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM3":{
                    "LABEL":"RADIUS",
                    "EDITABLE":true,
                },
                "PARAM4":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
            },
            "LAND":{
                "COMMAND": 21,
                "PARAM1":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM2":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM3":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM4":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
            },
            "TAKEOFF":{
                "COMMAND": 22,
                "PARAM1":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM2":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM3":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM4":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
            },
            "DO_JUMP":{
                "COMMAND": 177,
                "PARAM1":{
                    "LABEL":"SEQUENCE",
                    "EDITABLE":true,
                },
                "PARAM2":{
                    "LABEL":"REPEAT",
                    "EDITABLE":true,
                },
                "PARAM3":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM4":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
            }
        },
        "MAV_TYPE_OCTOROTOR":{
            "WAYPOINT":{
                "COMMAND": 16,
                "PARAM1":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM2":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM3":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM4":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
            },
            "LOITER":{
                "COMMAND": 19,
                "PARAM1":{
                    "LABEL":"TIME(S)",
                    "EDITABLE":true,
                },
                "PARAM2":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM3":{
                    "LABEL":"RADIUS",
                    "EDITABLE":true,
                },
                "PARAM4":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
            },
            "LAND":{
                "COMMAND": 21,
                "PARAM1":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM2":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM3":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM4":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
            },
            "TAKEOFF":{
                "COMMAND": 22,
                "PARAM1":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM2":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM3":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM4":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
            },
            "DO_JUMP":{
                "COMMAND": 177,
                "PARAM1":{
                    "LABEL":"SEQUENCE",
                    "EDITABLE":true,
                },
                "PARAM2":{
                    "LABEL":"REPEAT",
                    "EDITABLE":true,
                },
                "PARAM3":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM4":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
            }
        },
        "MAV_TYPE_VTOL_QUADROTOR":{
            "WAYPOINT":{
                "COMMAND": 16,
                "PARAM1":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM2":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM3":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM4":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
            },
            "LOITER":{
                "COMMAND": 19,
                "PARAM1":{
                    "LABEL":"TIME(S)",
                    "EDITABLE":true,
                },
                "PARAM2":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM3":{
                    "LABEL":"RADIUS",
                    "EDITABLE":false,
                },
                "PARAM4":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
            },
            "VTOL_LAND":{
                "COMMAND": 85,
                "PARAM1":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM2":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM3":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM4":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
            },
            "VTOL_TAKEOFF":{
                "COMMAND": 84,
                "PARAM1":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM2":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM3":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM4":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
            },
            "DO_JUMP":{
                "COMMAND": 177,
                "PARAM1":{
                    "LABEL":"SEQUENCE",
                    "EDITABLE":true,
                },
                "PARAM2":{
                    "LABEL":"REPEAT",
                    "EDITABLE":true,
                },
                "PARAM3":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM4":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
            }
        },
        "MAV_TYPE_FIXED_WING":{
            "WAYPOINT":{
                "COMMAND": 16,
                "PARAM1":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM2":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM3":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM4":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
            },
            "LOITER":{
                "COMMAND": 19,
                "PARAM1":{
                    "LABEL":"TIME(S)",
                    "EDITABLE":true,
                },
                "PARAM2":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM3":{
                    "LABEL":"RADIUS",
                    "EDITABLE":false,
                },
                "PARAM4":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
            },
            "VTOL_LAND":{
                "COMMAND": 85,
                "PARAM1":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM2":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM3":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM4":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
            },
            "VTOL_TAKEOFF":{
                "COMMAND": 84,
                "PARAM1":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM2":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM3":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM4":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
            },
            "DO_JUMP":{
                "COMMAND": 177,
                "PARAM1":{
                    "LABEL":"SEQUENCE",
                    "EDITABLE":true,
                },
                "PARAM2":{
                    "LABEL":"REPEAT",
                    "EDITABLE":true,
                },
                "PARAM3":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
                "PARAM4":{
                    "LABEL":"NOT USED",
                    "EDITABLE":false,
                },
            }
        }

    }
    property var lstWaypointMode: ["normal","selected","current"]
    property var lstWaypointAttributeKind:  ["bound","index","altitude"]
    property var lstColor:  {"normal":"orange","selected":"green","current":"blue"}    
    property int wpBoundSize: UIConstants.sRect
    property int wpFontSize: wpBoundSize / 2
    property int numPoinTrailUAV: 1500
    property var listGraphicMarker: []
    property var selectedMarker: undefined
    property var markerSymbolLink: {
        "MARKER_DEFAULT":"qrc:/qmlimages/markers/FlagIcon.png",
        "MARKER_TANK":"qrc:/qmlimages/markers/TankIcon.png",
        "MARKER_PLANE":"qrc:/qmlimages/markers/PlaneIcon.png",
        "MARKER_SHIP":"qrc:/qmlimages/markers/BattleShip.png",
        "MARKER_TARGET":"qrc:/qmlimages/markers/TargetIcon.png"
    }
    property var vehicleSymbolLink: {
        "MAV_TYPE_QUADROTOR":"qrc:/qmlimages/uavIcons/QuadRotorX.png",
        "MAV_TYPE_OCTOROTOR":"qrc:/qmlimages/uavIcons/QuadRotorX.png",
        "MAV_TYPE_VTOL_QUADROTOR":"qrc:/qmlimages/uavIcons/VTOLPlane.png",
        "MAV_TYPE_FIXED_WING":"qrc:/qmlimages/uavIcons/Plane.png",
        "MAV_TYPE_GENERIC":"qrc:/qmlimages/uavIcons/Unknown.png"
    }
    signal clicked(real lat,real lon,real alt)
    signal mapClicked(bool isMap)
    signal mapMoved()
    signal homePositionChanged(real lat, real lon, real alt)
    signal showAdvancedConfigChanged()
    Computer{
        id: cInfo
    }

    Elevation{
        id: elevationFinder
        path: "Elevation"
    }
    MarkerList{
        id: lstMarkers
    }
    MarkerList{
        id: lstMarkersSave
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
            folderPath: cInfo.homeFolder()+"/ArcGIS/Runtime/Data/elevation/"+mapHeightFolder
        }
    }
    function getTotalDistanceWP(){
        return 0;
    }

    function identifyGraphic(listGraphics,x,y){
        var result = undefined;
        for(var i = 0; i< listGraphics.count; i++){
            var graphicItem = listGraphics.get(i);
            if(graphicItem.geometry === null){
                continue;
            }else if(graphicItem.geometry.extent === null){
                continue;
            }else if(graphicItem.geometry.extent.center === null){
                continue;
            }
            var pointOnMap = ArcGISRuntimeEnvironment.createObject("Point", {
                              x: graphicItem.geometry.extent.center.x,
                              y: graphicItem.geometry.extent.center.y,
                              spatialReference: SpatialReference.createWebMercator()
                          });
            var pointScreen = mapView.locationToScreen(pointOnMap);
            if(graphicItem.attributes.containsAttribute("id")){
                var index = graphicItem.attributes.attributeValue("id");
                var type = graphicItem.attributes.attributeValue("type");
                var dx = x - pointScreen.x;
                var dy = y - pointScreen.y;
                if(dx*dx + dy*dy < wpBoundSize*wpBoundSize){
                    console.log("found graphicItem["+index+"]["+type+
                                "]("+pointScreen.x+","+pointScreen.y+")");
                    result = graphicItem;
                    if(type === "WP"){
                        rootItem.selectedIndex = index;
                        rootItem.mapClicked(false);
                    }
                    break;
                }
            }
        }
        if(result !== undefined){
            if(result.attributes.attributeValue("type") === "marker"){
                if(selectedMarker === undefined){
                    selectedMarker = result;
                    selectedMarker.selected = true;
                }else{
                    if(selectedMarker.attributes.attributeValue("id") ===
                            result.attributes.attributeValue("id")){
                        selectedMarker.selected = true;
                    }else{
                        selectedMarker.selected = false;
                        selectedMarker = result;
                        selectedMarker.selected = true;
                    }
                }
            }else if(result.attributes.attributeValue("type") === "WP"){
                if(selectedWP === undefined){
                    selectedWP = result;
                    changeModeWP(selectedWP,selectedWP.attributes.attributeValue("id"),"selected");
                    rootItem.mapClicked(false);
                }else{
                    changeModeWP(selectedWP,selectedWP.attributes.attributeValue("id"),"normal");
                    selectedWP = result;
                    changeModeWP(selectedWP,selectedWP.attributes.attributeValue("id"),"selected");
                }
                updateMouseOnMap();
            }
        }else{
            if(selectedWP !== undefined){
                rootItem.restoreWP();
                changeModeWP(selectedWP,selectedWP.attributes.attributeValue("id"),"normal");
                selectedWP = undefined;
            }
            if(selectedMarker !== undefined){
                rootItem.restoreMarker(mapOverlayWaypoints.graphics,
                                       selectedMarker.attributes.attributeValue("id"));
                selectedMarker.selected = false;
                selectedMarker = undefined;
            }
            rootItem.selectedIndex = -1;
            rootItem.mapClicked(true);
        }
        return result;
    }

    function updateWPDoJump(){
//        console.log("updateWPDoJump");
        for(var i = 0; i< mapOverlayWaypoints.graphics.count; i++){
            var graphicItem = mapOverlayWaypoints.graphics.get(i);
//            console.log("graphicItem["+i+"] type="+graphicItem.attributes.attributeValue("type")+" command="+
//                        graphicItem.attributes.attributeValue("command"));
            if(graphicItem.attributes.containsAttribute("type") &&
                graphicItem.attributes.attributeValue("type") === "WP" &&
                graphicItem.attributes.attributeValue("command") === lstWaypointCommand[vehicleType]["DO_JUMP"]["COMMAND"]){
                var index = graphicItem.attributes.attributeValue("id");
                var listWPPrevID = findListMarker(mapOverlayWaypoints.graphics,"WP",index-1);
                if(listWPPrevID.length > 0){
                    var prevWP = mapOverlayWaypoints.graphics.get(listWPPrevID[0]);
                    var pointPrevOnMap = ArcGISRuntimeEnvironment.createObject("Point", {
                                      x: prevWP.geometry.extent.center.x,
                                      y: prevWP.geometry.extent.center.y,
                                      spatialReference: SpatialReference.createWebMercator()
                                  });
                    var pointPrevOnScreen = mapView.locationToScreen(pointPrevOnMap);
                    var pointWPOnMap = mapView.screenToLocation(pointPrevOnScreen.x + wpBoundSize*3/2,
                                                                   pointPrevOnScreen.y);
                    graphicItem.geometry = pointWPOnMap;
//                        console.log("pointWPOnMapLatLon["+pointWPOnMapLatLon['lat']+","+ pointWPOnMapLatLon['lon']+"]");

                }
            }
        }
    }

    function clearMarkers(){
        selectedMarker = undefined;
        var numGraphic = mapOverlayWaypoints.graphics.count;
        for(var x = numGraphic-1; x >= 0; x--){
            var marker = mapOverlayWaypoints.graphics.get(x);
            if(marker.attributes.containsAttribute("type") &&
                    marker.attributes.attributeValue("type") === "marker"){
                mapOverlayWaypoints.graphics.remove(x);
            }
        }
        listGraphicMarker = [];
    }
    function loadMarker(_name){
        lstMarkers.loadMarkers(_name);
        clearMarkers();
        for(var i=0; i< lstMarkers.numMarker();i++){
            var marker = lstMarkers.getMarker(i);
            addMarkerWGS84(marker.latitude,marker.longtitude,marker.markerType,marker.description);
        }

    }
    function saveMarker(_name){
        lstMarkersSave.cleanMarker();
        for(var i=0; i< mapOverlayWaypoints.graphics.count; i++){
            var marker = mapOverlayWaypoints.graphics.get(i);
            if(marker.attributes.containsAttribute("type") &&
                    marker.attributes.attributeValue("type") === "marker"){
                var latlon = Conv.mercatorToLatLon(marker.geometry.extent.center.x,marker.geometry.extent.center.y);
                lstMarkersSave.insertMarker(Number(latlon['lat']).toString(),
                                            Number(latlon['lon']).toString(),
                                            marker.attributes.attributeValue("kind"),
                                            marker.attributes.attributeValue("description"));
            }
        }
        lstMarkersSave.saveMarkers(_name);
    }
    function changeZOrder(indexClick){
        if(listVideoTab[indexClick].z === zOrder.length - 1){
            return;
        }
//        console.log("before z: ["+
//                    listVideoTab[0].z+","+
//                    listVideoTab[1].z+","+
//                    listVideoTab[2].z+","+
//                    listVideoTab[3].z+","+"]/"+zOrder.length);
        for(var i=0; i< listVideoTab.length; i++){
            var tab = listVideoTab[i];
            if(i !== indexClick && tab.z > listVideoTab[indexClick].z){
                tab.z--;
            }
        }
        listVideoTab[indexClick].z = zOrder.length - 1;
//        console.log("after z: ["+
//                    listVideoTab[0].z+","+
//                    listVideoTab[1].z+","+
//                    listVideoTab[2].z+","+
//                    listVideoTab[3].z+","+"]/"+zOrder.length);
    }
    function showOnMap(type,id,show){
        if(type === "PM"){
            if(id >= 0 && id < listVideoTab.length){
                listVideoTab[id].visible = show;
            }
        }else if(type === "CC"){
            if(id >= 0 && id < listCenterCommander.length){
                listCenterCommander[id].visible = show;
            }
        }else if(type === "DR"){
            if(id >= 0 && id < listDrones.length){
                listDrones[id].visible = show;
            }
        }
    }
    function findListMarker(listGraphicMarker,type,id){
            var markerIndex = [];
            for(var graphicID = 0; graphicID < listGraphicMarker.count ; graphicID ++){
                if( listGraphicMarker.get(graphicID).attributes.containsAttribute("id") &&
                    listGraphicMarker.get(graphicID).attributes.attributeValue("id") === id &&
                    listGraphicMarker.get(graphicID).attributes.attributeValue("type") === type
                        ){
                    markerIndex.push(graphicID);
                }

            }
            return markerIndex;
    }

    function deSelectGraphics(lstGraphics,type){
        for(var id = 0; id < lstGraphics.count ; id ++){
            var graphicItem = lstGraphics.get(id);
            if(graphicItem.attributes.containsAttribute("type") &&
                    graphicItem.attributes.attributeValue("type") === type){
                if(graphicItem.attributes.attributeValue("mode") === "selected"){
                    changeModeWP(graphicItem,graphicItem.attributes.attributeValue("id"),"normal")
                }
            }
        }
    }

    function removeGraphics(lstGraphics,type,markerID){
        var lstMarkerIndex = findListMarker(lstGraphics,type,markerID);
        for(var id = lstMarkerIndex.length-1; id >=0 ; id --){
            lstGraphics.remove(lstMarkerIndex[id]);
        }
    }
    function recalcWPID(lstGraphics,removedID){
//        for(var i=0; i < listWaypoint.length; i++){
//            console.log("listWaypoint["+i+"]"+listWaypoint[i]);
//        }
        var diff = 1;
        if(listWaypoint.length >=2 ){
            diff = listWaypoint[1]-listWaypoint[0];
        }
//        console.log("diff = "+diff);
        for(var id = 0; id < lstGraphics.count ; id ++){
            var wp = lstGraphics.get(id);
            if(wp.attributes.containsAttribute("id") &&
                wp.attributes.attributeValue("id") > removedID &&
                wp.attributes.attributeValue("type") === "WP"){
                var wpID = wp.attributes.attributeValue("id");
//                console.log("wpID = "+wpID);
                var newWpID = wpID;
                if(listWaypoint.length >=2){
                    if(wp.attributes.attributeValue("id") === listWaypoint[1]){
                        newWpID = listWaypoint[0];
                    }else{
                        newWpID = wpID-1;
                    }
                }else{
                    newWpID = wpID-1;
                }

                wp.attributes.replaceAttribute("id",newWpID);
                if(newWpID > 0){
                    var indexSymbol = ArcGISRuntimeEnvironment.createObject("TextSymbol",{
                                                               color: "white",
                                                               fontWidth: Enums.FontWeightBold,
                                                               fontFamily: UIConstants.appFont,
                                                               text: Number(newWpID).toString(),
                                                               size: wpFontSize,
                                                           });

                    var wpIndexID = 1;
                    if(wp.symbol.symbols.count > wpIndexID){
                        wp.symbol.symbols.remove(wpIndexID,1);
                    }
                    wp.symbol.symbols.insert(wpIndexID,indexSymbol);
                }else{
                    var altitudeHome = elevationFinder.getAltitude(
                                            cInfo.homeFolder()+"/ArcGIS/Runtime/Data/elevation/"+mapHeightFolder,
                                            wp.attributes.attributeValue("latitude"),
                                            wp.attributes.attributeValue("longitude"));
                    wp.symbol = selectedWP.symbol = createWPSymbol(wp.attributes.attributeValue("id"),
                                                                   QtPositioning.coordinate(
                                                                       wp.attributes.attributeValue("latitude"),
                                                                       wp.attributes.attributeValue("longitude"),
                                                                       altitudeHome),
                                                                   lstWaypointCommand[vehicleType]["WAYPOINT"]["COMMAND"],
                                                                   wp.attributes.attributeValue("param1"),
                                                                   wp.attributes.attributeValue("param2"),
                                                                   wp.attributes.attributeValue("param3"),
                                                                   wp.attributes.attributeValue("param4"));
                }
            }
        }

    }
    function recalcWPPath(removedID){
        if(waypointPathBuilder.parts.part(0)){
            // remove point at id = removedID-1 cause no line between point[0] to point[1]
            waypointPathBuilder.parts.part(0).removePoint(removedID-1);
            waypointPathGraphic.geometry = waypointPathBuilder.geometry;
        }
    }

    function setMap(_name){
        console.log("Selected file "+_name);
        mapName = _name;
        var map = ArcGISRuntimeEnvironment.createObject("Map");
        var basemap = ArcGISRuntimeEnvironment.createObject("Basemap");
        var arcTileCache = ArcGISRuntimeEnvironment.createObject("TileCache",{
                                                                    path: dataPath + "tpk/"+_name
                                                                 });
        var arcTiledLayer = ArcGISRuntimeEnvironment.createObject("ArcGISTiledLayer",{
                                                                      tileCache: arcTileCache
                                                                   });
        basemap.baseLayers.append(arcTiledLayer);
        map.basemap  = basemap;
        mapView.map = map;
    }
    function setMapOnline(){
        console.log("Set map online");
        var map = ArcGISRuntimeEnvironment.createObject("Map");
        var basemap = ArcGISRuntimeEnvironment.createObject("BasemapImagery");
        map.basemap  = basemap;
        mapView.map = map;
    }
    function createGraphic(geometry, symbol){
        var graphic = ArcGISRuntimeEnvironment.createObject("Graphic");
        graphic.geometry = geometry;
        graphic.symbol = symbol;
        return graphic;
    }
    function createMarkerSymbol(type,description){
        var compositeSymbol = ArcGISRuntimeEnvironment.createObject(
                    "CompositeSymbol",{
                                                                        });
        var opacity = 1;
        var symbol = ArcGISRuntimeEnvironment.createObject("PictureMarkerSymbol",{
                                                           url: markerSymbolLink[type],
                                                           width: wpBoundSize,
                                                           height: wpBoundSize,
                                                           opacity: opacity
                                                       });
        var symbolText = ArcGISRuntimeEnvironment.createObject("TextSymbol",{
                                                               backgroundColor: UIConstants.transparentBlueDarker,
                                                               color: "white",
                                                               fontWidth: Enums.FontWeightBold,
                                                               fontFamily: UIConstants.appFont,
                                                               fontSize: UIConstants.fontSize,
                                                               text: description,
                                                               size: wpFontSize,
                                                               offsetY: - wpBoundSize - wpFontSize,
                                                               verticalAlignment: Enums.VerticalAlignmentBottom
                                                           });
        compositeSymbol.symbols.append(symbol);
        compositeSymbol.symbols.append(symbolText);
        return compositeSymbol;
    }
    function changeVehicleType(vehicleType){
        console.log("changeVehicleType to "+vehicleType);
        console.log("vehicle.MAV_TYPE_FIXED_WING="+Vehicle.MAV_TYPE_FIXED_WING);
        var vehicleSymbolUrl = vehicleSymbolLink["MAV_TYPE_GENERIC"];
        var vehicleHeading = uavGraphic.symbol.angle;
        var opacity = uavGraphic.symbol.opacity;
        switch(vehicleType){
        case 2:
            rootItem.vehicleType = "MAV_TYPE_QUADROTOR";
            waypointEditor.vehicleType = rootItem.vehicleType;
            vehicleSymbolUrl = vehicleSymbolLink["MAV_TYPE_QUADROTOR"];
            break;
        case 14:
            rootItem.vehicleType = "MAV_TYPE_OCTOROTOR";
            waypointEditor.vehicleType = rootItem.vehicleType;
            vehicleSymbolUrl = vehicleSymbolLink["MAV_TYPE_OCTOROTOR"];
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
//        console.log("uavGraphic.symbol.url=["+uavGraphic.symbol.url+"]")
        console.log("vehicleSymbolUrl=["+vehicleSymbolUrl+"]")
        if(uavGraphic.symbol.url !== vehicleSymbolUrl){
            console.log("Change vehicle icon");
            var symbol = ArcGISRuntimeEnvironment.createObject("PictureMarkerSymbol",{
               url: vehicleSymbolUrl,
               opacity: opacity,
               angle: vehicleHeading,
               width: wpBoundSize * 1.5,
               height: wpBoundSize * 1.5
            });
            uavGraphic.symbol = symbol;
        }
    }

    function getMarkerText(marker){
        return marker.symbol.symbols.get(1).text;
    }
    function setMarkerText(marker,text){
        marker.symbol.symbols.get(1).text = text;
    }
    function changeMarkerType(marker,newType){
        var symbol = ArcGISRuntimeEnvironment.createObject("PictureMarkerSymbol",{
                                                           url: markerSymbolLink[newType],
                                                           width: wpBoundSize,
                                                           height: wpBoundSize,
                                                           opacity: opacity
                                                       });
        marker.symbol.symbols.remove(0.0);
        marker.symbol.symbols.insert(0,symbol);
        marker.attributes.replaceAttribute("kind_next",newType);
    }
    function getMarkerType(marker){
        return marker.attributes.attributeValue("kind");
    }
    function createWPSymbol(index,position,command,
                            param1,param2,param3,param4){
        console.log("createWPSymbol ["+index+"]"+"-["+command+"] "+position+" params["+param1+","+param2+","+param3+","+param4+"]");
        var compositeSymbol = ArcGISRuntimeEnvironment.createObject("CompositeSymbol",{
                                                                        });
        var symbol = ArcGISRuntimeEnvironment.createObject("SimpleMarkerSymbol",{
                                                               //antiAlias : true,
                                                               style: Enums.SimpleMarkerSymbolStyleCircle,
                                                               color : lstColor["normal"],
                                                               size : wpBoundSize
                                                           });
        var indexSymbol = ArcGISRuntimeEnvironment.createObject("TextSymbol",{
                                                                       color: "white",
                                                                       fontWidth: Enums.FontWeightBold,
                                                                       fontFamily: UIConstants.appFont,
                                                                       text: Number(index).toString(),
                                                                       size: wpFontSize,
                                                                   });
        var altitudeSymbol = ArcGISRuntimeEnvironment.createObject("TextSymbol",{
                                                                       color: "white",
                                                                       fontWidth: Enums.FontWeightBold,
                                                                       fontFamily: UIConstants.appFont,
                                                                       text: Number(position.altitude).toFixed(0).toString() +"m",
                                                                       size: wpFontSize,
                                                                       offsetX: 0,
                                                                       offsetY: - wpBoundSize / 2 - wpFontSize / 2,
                                                                   });
        compositeSymbol.symbols.append(symbol);
        compositeSymbol.symbols.append(indexSymbol);
        compositeSymbol.symbols.append(altitudeSymbol);
        switch(command){
        case 16:
            if(index === 0){
                var symbolHome = ArcGISRuntimeEnvironment.createObject("PictureMarkerSymbol",{
                                                                       //antiAlias : true,
                                                                       url: "qrc:/qmlimages/home.png",
                                                                       width: wpBoundSize,
                                                                       height: wpBoundSize,
                                                                       opacity: 1
                                                                   });
                compositeSymbol.symbols.remove(1,1);
                compositeSymbol.symbols.append(symbolHome);
            }
            break;
        case 19:
            var symbolLoiter = ArcGISRuntimeEnvironment.createObject("PictureMarkerSymbol",{
                                                                   //antiAlias : true,
                                                                   url: param3>0?
                                                                            "qrc:/qmlimages/markers/ClockwiseCircle.png":
                                                                            "qrc:/qmlimages/markers/AntiClockwiseCircle.png",
                                                                   width: wpBoundSize + 1,
                                                                   height: wpBoundSize + 1,
                                                                   opacity: 1
                                                               });
            compositeSymbol.symbols.append(symbolLoiter);
            break;
        case 21:
        case 85:
            var symbolLand = ArcGISRuntimeEnvironment.createObject("PictureMarkerSymbol",{
                                                                   //antiAlias : true,
                                                                   url: "qrc:/qmlimages/land.png",
                                                                   width: wpBoundSize - 7,
                                                                   height: wpBoundSize - 7,
                                                                   opacity: 1,
                                                                   offsetY: 0
                                                               });
            compositeSymbol.symbols.remove(1,1);
            compositeSymbol.symbols.insert(1,symbolLand);
            break;
        case 22:
        case 84:
            var symbolTakeoff = ArcGISRuntimeEnvironment.createObject("PictureMarkerSymbol",{
                                                                   //antiAlias : true,
                                                                   url: "qrc:/qmlimages/takeoff.png",
                                                                   width: wpBoundSize - 7,
                                                                   height: wpBoundSize - 7,
                                                                   opacity: 1,
                                                                   offsetY: 0
                                                               });
            compositeSymbol.symbols.remove(1,1);
            compositeSymbol.symbols.insert(1,symbolTakeoff);
            break;
        case 177:
            var timeSymbolNext = ArcGISRuntimeEnvironment.createObject("TextSymbol",{
                                                                           color: "white",
                                                                           fontWidth: Enums.FontWeightBold,
                                                                           fontFamily: UIConstants.appFont,
                                                                           text: "->"+Number(param1).toFixed(0).toString(),
                                                                           size: wpFontSize,
                                                                           offsetX: 0,
                                                                           offsetY: - wpBoundSize / 2 - wpFontSize / 2,
                                                                       });
            console.log("create WP symbol do jump ->"+param1)
            compositeSymbol.symbols.remove(2,1);
            compositeSymbol.symbols.append(timeSymbolNext);
            break;
        }
        return compositeSymbol;
    }

    function createWPGraphic(index,position,command,
                            param1,param2,param3,param4){

        var compositeSymbol = createWPSymbol(index,position,command,param1,param2,param3,param4);
        var positionXY = Conv.latlongToMercator(position.latitude,position.longitude);
        var point = ArcGISRuntimeEnvironment.createObject("Point", {
                                                                      x: positionXY['x'],
                                                                      y: positionXY['y'],
                                                                  });
        var wpGraphic = createGraphic(point,compositeSymbol);
        wpGraphic.attributes.insertAttribute("id",index);
        wpGraphic.attributes.insertAttribute("type","WP");
        wpGraphic.attributes.insertAttribute("mode","normal");
        wpGraphic.attributes.insertAttribute("command",command);
        wpGraphic.attributes.insertAttribute("latitude",position.latitude);
        wpGraphic.attributes.insertAttribute("longitude",position.longitude);
        wpGraphic.attributes.insertAttribute("altitude",position.altitude);
        wpGraphic.attributes.insertAttribute("param1",param1);
        wpGraphic.attributes.insertAttribute("param2",param2);
        wpGraphic.attributes.insertAttribute("param3",param3);
        wpGraphic.attributes.insertAttribute("param4",param4);
        wpGraphic.attributes.insertAttribute("command_prev",command);
        wpGraphic.attributes.insertAttribute("latitude_prev",position.latitude);
        wpGraphic.attributes.insertAttribute("longitude_prev",position.longitude);
        wpGraphic.attributes.insertAttribute("altitude_prev",position.altitude);
        wpGraphic.attributes.insertAttribute("param1_prev",param1);
        wpGraphic.attributes.insertAttribute("param2_prev",param2);
        wpGraphic.attributes.insertAttribute("param3_prev",param3);
        wpGraphic.attributes.insertAttribute("param4_prev",param4);
        return wpGraphic;
    }

    function addWPPosition(index,position,command,
                           param1,param2,param3,param4){
        // add Waypoint
        listWaypoint.push(index);
        var positionXY = Conv.latlongToMercator(position.latitude,position.longitude);
        var point = ArcGISRuntimeEnvironment.createObject("Point", {
                                                              x: positionXY['x'],
                                                              y: positionXY['y'],
                                                              spatialReference: SpatialReference.createWebMercator()
                                                          });
        var wpGraphic;


        if(command !== lstWaypointCommand[vehicleType]["DO_JUMP"]["COMMAND"]){
            wpGraphic = createWPGraphic(index,position,command,
                                       param1,param2,param3,param4);
            // add line between last wp to new wp
            if(index > 0){
                waypointPathBuilder.addPoint(point);
                waypointPathGraphic.geometry = waypointPathBuilder.geometry;
            }
        }else{
            var listWPPrevID = findListMarker(mapOverlayWaypoints.graphics,"WP",index-1);
            if(listWPPrevID.length > 0){
                var prevWP = mapOverlayWaypoints.graphics.get(listWPPrevID[0]);
                var pointPrevOnMap = ArcGISRuntimeEnvironment.createObject("Point", {
                                  x: prevWP.geometry.extent.center.x,
                                  y: prevWP.geometry.extent.center.y,
                                  spatialReference: SpatialReference.createWebMercator()
                              });
                var pointPrevOnScreen = mapView.locationToScreen(pointPrevOnMap);
                var pointWPOnMap = mapView.screenToLocation(pointPrevOnScreen.x + wpBoundSize*3/2,
                                                               pointPrevOnScreen.y);
                var pointWPOnMapLatLon = Conv.mercatorToLatLon(pointWPOnMap.x,pointWPOnMap.y);
                wpGraphic = createWPGraphic(index,
                                           QtPositioning.coordinate(pointWPOnMapLatLon['lat'],
                                                                    pointWPOnMapLatLon['lon'],
                                                                    prevWP.attributes.attributeValue('altitude')),
                                           command,
                                           param1,param2,param3,param4);
            }
        }
        console.log("add createWPGraphic["+index+"]");
        mapOverlayWaypoints.graphics.append(wpGraphic);
    }
    function addWP(index){

        console.log("Before add wp["+index+"]mapOverlayWaypoints.graphics.count = "+mapOverlayWaypoints.graphics.count);
//        if(mouseBuilder.geometry.extent === null ||
//           mouseBuilder.geometry.extent.center === null )
//            return;
        isMapSync = false;
        // add Waypoint
        listWaypoint.push(index);
        console.log("mouseBuilder.geometry.extent.center.x = "+mouseBuilder.geometry.extent.center.x);
        var point = ArcGISRuntimeEnvironment.createObject("Point", {
                                                                      x: mouseBuilder.geometry.extent.center.x,
                                                                      y: mouseBuilder.geometry.extent.center.y,
                                                                      spatialReference: SpatialReference.createWebMercator()
                                                                  });
        var alt = 50;
        var pointLatlon = Conv.mercatorToLatLon(mouseBuilder.geometry.extent.center.x,
                                                mouseBuilder.geometry.extent.center.y);
        console.log("pointLatlon("+pointLatlon['lat']+","+pointLatlon['lon']+")");
        var wpIcon = createWPGraphic(index,
                                    QtPositioning.coordinate(pointLatlon['lat'],
                                                             pointLatlon['lon'],
                                                             alt),
                                    16,
                                    0,0,0,0);
        console.log("add createWPGraphic["+index+"]");
        mapOverlayWaypoints.graphics.append(wpIcon);
        console.log("listWaypoint.length = "+listWaypoint.length);
        console.log("mapOverlayWaypoints.graphics.count = "+mapOverlayWaypoints.graphics.count);
        // add line between last wp to new wp
        if(index > 0){
            waypointPathBuilder.addPoint(point);
            waypointPathGraphic.geometry = waypointPathBuilder.geometry;
        }
    }
    function lastWPIndex(){
        var result = -1;
        var numWP = listWaypoint.length;
        if(numWP>0) result = listWaypoint[numWP-1];
        return result;
    }

    function indexInListWaypoint(sequence){

//        console.log("listWaypoint="+listWaypoint);
        var index = -1;
        for(var i=0; i< listWaypoint.length; i++){
            if(listWaypoint[i] === sequence){
                index = i;
                break;
            }
        }
        var countWPDoJumpInLoop = 0;
        for(var i=0; i< mapOverlayWaypoints.graphics.count; i++){
            var graphicItem = mapOverlayWaypoints.graphics.get(i);
            if(graphicItem.attributes.containsAttribute("type") &&
                graphicItem.attributes.attributeValue("type") === "WP" &&
                graphicItem.attributes.attributeValue("command") === lstWaypointCommand[vehicleType]["DO_JUMP"]["COMMAND"]){
                if(graphicItem.attributes.attributeValue("id") < sequence){
                    countWPDoJumpInLoop++;
                }
            }
        }
        index -=  countWPDoJumpInLoop;
        console.log("indexInListWaypoint ["+sequence+"] = "+index);
        return index;
    }

    function removeWP(sequence){
        var lstWaypointRemove = findListMarker(mapOverlayWaypoints.graphics,"WP",sequence);
        if(lstWaypointRemove.length <=0 ) return;
        var wpRemove = mapOverlayWaypoints.graphics.get(lstWaypointRemove[0]);
        console.log("removeWP["+sequence+"]"+wpRemove.attributes.attributeValue("command"));
        removeGraphics(mapOverlayWaypoints.graphics,"WP",sequence);
        // recalculate waypoint id
        recalcWPID(mapOverlayWaypoints.graphics,sequence);
        // recalculate waypoints path
        if(wpRemove.attributes.attributeValue("command") !== lstWaypointCommand[vehicleType]["DO_JUMP"]["COMMAND"])
            recalcWPPath(indexInListWaypoint(sequence));
        // remove element from list
        listWaypoint.pop();
        if(selectedWP !== undefined){
            updateWPDoJump();
            selectedIndex = -1;
            selectedWP = undefined;
            isMapSync = false;
        }
    }

    function clearWPs(){
        clickedGraphic.visible = false;
        for(var i=listWaypoint.length - 1; i >=0 ; i--){
            // clear waypoints
            removeWP(listWaypoint[i]);
        }
        listWaypoint = [];
        if(selectedWP !== undefined){
            selectedIndex = -1;
            selectedWP = undefined;
        }
        if(waypointPathBuilder.parts.part(0)){
            waypointPathBuilder.parts.part(0).removeAll();
            waypointPathGraphic.geometry = waypointPathBuilder.geometry;
        }
    }
    function moveMarkerPosition(marker,position){
        // move wp
        marker.geometry = position;
    }
    function moveWPWithID(waypointID,positionWGS84){
        var lstWPID = findListMarker(mapOverlayWaypoints.graphics,"WP",waypointID);
        if(lstWPID.length > 0){
            var pointMer = Conv.latlongToMercator(positionWGS84.latitude,positionWGS84.longitude);

            var point = ArcGISRuntimeEnvironment.createObject("Point", {
                                                                  x: pointMer['x'],
                                                                  y: pointMer['y']
                                                              });
            moveWPPosition(mapOverlayWaypoints.graphics.get(lstWPID[0]),point);
        }
    }
    function wpPathID(index){
        var pointInPath = -1;
        if(waypointID === 0)
        {
            pointInPath = 0;
        }else if(waypointID === 1){
            pointInPath = waypointID;
        }else if(waypointID >=2){
            if(listWaypoint[1] === 2){
                pointInPath = waypointID -1;
            }else{
                pointInPath = waypointID;
            }
        }
        var countWPDoJumpInLoop = 0;
        for(var i=0; i< mapOverlayWaypoints.graphics.count; i++){
            var graphicItem = mapOverlayWaypoints.graphics.get(i);
            if(graphicItem.attributes.containsAttribute("type") &&
                graphicItem.attributes.attributeValue("type") === "WP" &&
                graphicItem.attributes.attributeValue("command") === lstWaypointCommand[vehicleType]["DO_JUMP"]["COMMAND"]){
                if(graphicItem.attributes.attributeValue("id") < waypointID){
                    countWPDoJumpInLoop++;
                }
            }
        }
        return pointInPath-countWPDoJumpInLoop;
    }

    function recalcWPPathInAfterAdd(selectedWP){
        var waypointID = selectedWP.attributes.attributeValue("id");
        var point = ArcGISRuntimeEnvironment.createObject("Point", {
                                                              x: selectedWP.geometry.extent.center.x,
                                                              y: selectedWP.geometry.extent.center.y
                                                          });
        // update path
        var pointInPath = -1;
        if(waypointID === 0)
        {
            pointInPath = 0;
        }else if(waypointID === 1){
            pointInPath = waypointID;
        }else if(waypointID >=2){
            if(listWaypoint[1] === 2){
                pointInPath = waypointID -1;
            }else{
                pointInPath = waypointID;
            }
        }
        var countWPDoJumpInLoop = 0;
        for(var i=0; i< mapOverlayWaypoints.graphics.count; i++){
            var graphicItem = mapOverlayWaypoints.graphics.get(i);
            if(graphicItem.attributes.containsAttribute("type") &&
                graphicItem.attributes.attributeValue("type") === "WP" &&
                graphicItem.attributes.attributeValue("command") === lstWaypointCommand[vehicleType]["DO_JUMP"]["COMMAND"]){
                if(graphicItem.attributes.attributeValue("id") < waypointID){
                    countWPDoJumpInLoop++;
                }
            }
        }
        if(pointInPath-countWPDoJumpInLoop-1 >= 0){
            waypointPathBuilder.parts.part(0).insertPoint(pointInPath-countWPDoJumpInLoop-1,point);
            waypointPathGraphic.geometry = waypointPathBuilder.geometry;
        }
    }

    function recalcWPPathInAfterRemove(selectedWP){
        var waypointID = selectedWP.attributes.attributeValue("id");
        var point = ArcGISRuntimeEnvironment.createObject("Point", {
                                                              x: selectedWP.geometry.extent.center.x,
                                                              y: selectedWP.geometry.extent.center.y
                                                          });
        // update path
        var pointInPath = -1;
        if(waypointID === 0)
        {
            pointInPath = 0;
        }else if(waypointID === 1){
            pointInPath = waypointID;
        }else if(waypointID >=2){
            if(listWaypoint[1] === 2){
                pointInPath = waypointID -1;
            }else{
                pointInPath = waypointID;
            }
        }
        var countWPDoJumpInLoop = 0;
        for(var i=0; i< mapOverlayWaypoints.graphics.count; i++){
            var graphicItem = mapOverlayWaypoints.graphics.get(i);
            if(graphicItem.attributes.containsAttribute("type") &&
                graphicItem.attributes.attributeValue("type") === "WP" &&
                graphicItem.attributes.attributeValue("command") === lstWaypointCommand[vehicleType]["DO_JUMP"]["COMMAND"]){
                if(graphicItem.attributes.attributeValue("id") < waypointID){
                    countWPDoJumpInLoop++;
                }
            }
        }
        if(pointInPath-countWPDoJumpInLoop-1 >= 0){
            waypointPathBuilder.parts.part(0).removePoint(pointInPath-countWPDoJumpInLoop-1);
            waypointPathGraphic.geometry = waypointPathBuilder.geometry;
        }
    }

    function moveWPPosition(selectedWP,position){
        var waypointID = selectedWP.attributes.attributeValue("id");
        // move wp
//        console.log("move waypoint "+waypointID);
//        console.log("lstGraphics.count = "+lstGraphics.count);
        var point = ArcGISRuntimeEnvironment.createObject("Point", {
                                                              x: position.x,
                                                              y: position.y
                                                          });
        selectedWP.geometry = point;
        // update path
        var pointInPath = -1;
        if(waypointID === 0)
        {
            pointInPath = 0;
        }else if(waypointID === 1){
            pointInPath = waypointID;
        }else if(waypointID >=2){
            if(listWaypoint[1] === 2){
                pointInPath = waypointID -1;
            }else{
                pointInPath = waypointID;
            }
        }
        var countWPDoJumpInLoop = 0;
        for(var i=0; i< mapOverlayWaypoints.graphics.count; i++){
            var graphicItem = mapOverlayWaypoints.graphics.get(i);
            if(graphicItem.attributes.containsAttribute("type") &&
                graphicItem.attributes.attributeValue("type") === "WP" &&
                graphicItem.attributes.attributeValue("command") === lstWaypointCommand[vehicleType]["DO_JUMP"]["COMMAND"]){
                if(graphicItem.attributes.attributeValue("id") < waypointID){
                    countWPDoJumpInLoop++;
                }
            }
        }

        if(pointInPath-countWPDoJumpInLoop-1 >= 0){
            waypointPathBuilder.parts.part(0).setPoint(pointInPath-countWPDoJumpInLoop-1,position);
            waypointPathGraphic.geometry = waypointPathBuilder.geometry;
        }
    }

    function focusAllObject(){
//        var maxScale = mapView.map.maxScale;
//        if(mapView.map.maxScale === 0) {
//            maxScale = 8000;
//        }
//        mapView.setViewpointCenter(uavBuilder.geometry.extent.center);
    }
    function focusOnUAV(){
        var maxScale = mapView.map.maxScale;
        if(mapView.map.maxScale === 0) {
            maxScale = 8000;
        }
        mapView.setViewpointCenter(uavBuilder.geometry.extent.center);
    }
    function focusOnTracker(){
        var maxScale = mapView.map.maxScale;
        if(mapView.map.maxScale === 0){
            maxScale = 8000
        }
        mapView.setViewpointCenter(homeBuilder.geometry.extent.center);
    }
    function zoomIn(){
        var maxScale = mapView.map.maxScale;
        if(mapView.map.maxScale === 0) {
            maxScale = 8000;
        }
        var pointFocus;
        if(selectedWP !== undefined){
            pointFocus = selectedWP;
        }else{
            pointFocus = mouseGraphic;
        }

        if(mapView.mapScale / 2 >  maxScale)
            mapView.setViewpointCenterAndScale(pointFocus.geometry.extent.center,mapView.mapScale / 2)
        else
            mapView.setViewpointCenterAndScale(pointFocus.geometry.extent.center,maxScale)
    }
    function zoomOut(){
        var minScale = mapView.map.minScale;
        if(mapView.map.minScale === 0) {
            minScale = 14539821;
        }
        var pointFocus;
        if(selectedWP !== undefined){
            pointFocus = selectedWP;
        }else{
            pointFocus = mouseGraphic;
        }
        if(mapView.mapScale * 2 <  minScale)
            mapView.setViewpointCenterAndScale(pointFocus.geometry.extent.center,mapView.mapScale * 2)
        else
            mapView.setViewpointCenterAndScale(pointFocus.geometry.extent.center,minScale)
    }
    function focusOnWP(waypointID){
        console.log("focusOnWP["+waypointID+"]")
        var foundWP = false;
        for(var i =0; i< mapOverlayWaypoints.graphics.count; i++){
            var wp = mapOverlayWaypoints.graphics.get(i);
            if(wp.attributes.attributeValue("type") === "WP"){
                if(wp.attributes.attributeValue("id") === waypointID){

                    selectedWP = wp;
                    if(wp.attributes.attributeValue("mode") !== "current"){
                        changeModeWP(wp,waypointID,"selected");
                    }
                    var point = ArcGISRuntimeEnvironment.createObject("Point", {
                                                          x: selectedWP.geometry.extent.center.x,
                                                          y: selectedWP.geometry.extent.center.y,
                                                          spatialReference: SpatialReference.createWebMercator()
                                                      });
                    var mouseOnMap = mapView.locationToScreen(point);
                    rectDragWP.x = mouseOnMap.x - rectDragWP.width/2;
                    rectDragWP.y = mouseOnMap.y - rectDragWP.height/2;
                    mapView.setViewpointCenter(selectedWP.geometry.extent.center);
                    foundWP = true;
                }else if(wp.attributes.attributeValue("id") !== waypointID){
                    if(wp.attributes.attributeValue("mode") !== "current")
                        changeModeWP(wp,wp.attributes.attributeValue("id"),"normal");
                }
            }
        }
        if(!foundWP){
            if(selectedWP !== undefined){
                selectedIndex = -1;
                selectedWP = undefined;
            }
        }
    }
    function focusOnPosition(latitude,longitude){
        var pointMer = Conv.latlongToMercator(latitude,longitude);
        var point2Geo = ArcGISRuntimeEnvironment.createObject("Point",
                                                                {
                                                                    x: pointMer['x'],
                                                                    y: pointMer['y'],
                                                                    spatialReference: SpatialReference.createWebMercator()
                                                                });
        mapView.setViewpointCenter(point2Geo);
        mouseBuilder.setXY(pointMer['x'],pointMer['y']);
        mouseGraphic.geometry = mouseBuilder.geometry;
        if(selectedWP !== undefined){
            selectedWP = undefined;
            selectedIndex = -1;
        }
    }

    function updatePlane(position){
        // update plane position
//        console.log("plane("+position.latitude+","+position.longitude+")")
        var mer = Conv.latlongToMercator(position.latitude,position.longitude);
        uavBuilder.setXY(mer['x'],mer['y']);
        uavGraphic.geometry = uavBuilder.geometry;
        // update trajectory
        trajectory.addPointXY(mer['x'], mer['y']);
        trajectoryGraphic.geometry = trajectory.geometry;
        var partTrajectory = trajectory.parts.part(0);
        if(partTrajectory === null){
            //console.log("part not exist");
        }else{
            //console.log("trajectory has "+partTrajectory.pointCount+" point");
            if(partTrajectory.pointCount > numPoinTrailUAV){
                partTrajectory.removePoint(0);
            }
        }
    }
    function updateHeadingPlane(angle){
        uavGraphic.symbol.angle = angle;
    }
    function drawTargetLocalization(point1,point2,point3,point4,point5,plane){
        planeFOVBuilder.parts.removeAll();
        planeFOVBuilder.addPointXY(point1.y,point1.x);
        planeFOVBuilder.addPointXY(point2.y,point2.x);
        planeFOVBuilder.addPointXY(point3.y,point3.x);
        planeFOVBuilder.addPointXY(point4.y,point4.x);
        planeFOVGraphic.geometry = planeFOVBuilder.geometry;

        centralFOVBuilder.parts.removeAll();
        centralFOVBuilder.addPointXY(plane.y,plane.x);
        centralFOVBuilder.addPointXY(point5.y,point5.x);
        centralFOVGraphicLine.geometry = centralFOVBuilder.geometry;

        var headPoint = ArcGISRuntimeEnvironment.createObject("Point", {
                                                                   x: point5.y,
                                                                   y: point5.x,
                                                                   spatialReference: SpatialReference.createWgs84()
                                                               });
        centralFOVGraphic.geometry = headPoint;
    }
    function updateTracker(position){
        // update plane position
//        console.log("Tracker("+position.latitude+","+position.longitude+")")
        var mer = Conv.latlongToMercator(position.latitude,position.longitude);
        homeBuilder.setXY(mer['x'],mer['y']);
        homeGraphic.geometry = homeBuilder.geometry;

    }
    function updateHeadingTracker(angle){
        homeGraphic.symbol.symbols.get(1).angle = angle;
    }

    function setMapFocus(enable){
        mapView.focus = enable;
    }

    function changeModeWP(wp,wpID,mode){
        if(wp.attributes.attributeValue("id") === wpID){
//            console.log("change WP["+wpID+"] from "+wp.attributes.attributeValue("mode")+" to "+mode);
            wp.attributes.replaceAttribute("mode",mode);
            wp.symbol.symbols.get(0).color = lstColor[mode];
        }
    }

    function changeCurrentWP(wpID){
        for(var i =0; i< mapOverlayWaypoints.graphics.count; i++){
            var wp = mapOverlayWaypoints.graphics.get(i);
            if(wp.attributes.attributeValue("type") === "WP"){
                if(wp.attributes.attributeValue("id") === wpID){
                    changeModeWP(wp,wpID,"current")
                }else if(wp.attributes.attributeValue("id") !== wpID
                         ){
                    if(wp.attributes.attributeValue("id") === rootItem.selectedIndex){
                        changeModeWP(wp,rootItem.selectedIndex,"selected");
                    }else {
                        changeModeWP(wp,wp.attributes.attributeValue("id"),"normal");
                    }
                }
            }
        }
    }

    function changeWPCommand(command,
                             param1,param2,param3,param4){
        if(selectedWP !== undefined &&
                selectedWP.attributes.attributeValue("id") !== 0){
            var index = selectedWP.attributes.attributeValue("id");
            console.log("current command "+selectedWP.attributes.attributeValue("command") +" to "+ command);

            // update command to attributes
            if(selectedWP.attributes.attributeValue("command") !== command){
                if(selectedWP.attributes.attributeValue("command") !== lstWaypointCommand[vehicleType]["DO_JUMP"]["COMMAND"]){
                    console.log("Line 990");
                    var newWPSymbol = createWPSymbol(index,QtPositioning.coordinate(
                                                         selectedWP.attributes.attributeValue("latitude"),
                                                         selectedWP.attributes.attributeValue("longitude"),
                                                         selectedWP.attributes.attributeValue("altitude")),
                                                     command,
                                                     param1,param2,param3,param4);
                    selectedWP.symbol = newWPSymbol;
                    if(command === lstWaypointCommand[vehicleType]["DO_JUMP"]["COMMAND"]){
                        // update for waypoint != do jump
                        // change symbol only
                        recalcWPPathInAfterRemove(selectedWP);
                        // update location on map
                        var listWPPrevID = findListMarker(mapOverlayWaypoints.graphics,"WP",index-1);
                        if(listWPPrevID.length > 0){
                            var prevWP = mapOverlayWaypoints.graphics.get(listWPPrevID[0]);
                            var pointPrevOnMap = ArcGISRuntimeEnvironment.createObject("Point", {
                                              x: prevWP.geometry.extent.center.x,
                                              y: prevWP.geometry.extent.center.y,
                                              spatialReference: SpatialReference.createWebMercator()
                                          });
                            var pointPrevOnScreen = mapView.locationToScreen(pointPrevOnMap);
                            rectDragWP.x = pointPrevOnScreen.x + wpBoundSize*3/2 - rectDragWP.width/2;
                            rectDragWP.y = pointPrevOnScreen.y - rectDragWP.height/2;
                            var pointWPOnMap = mapView.screenToLocation(pointPrevOnScreen.x + wpBoundSize*3/2,
                                                                           pointPrevOnScreen.y);
                            selectedWP.geometry = pointWPOnMap;
                        }
                        updateWPDoJump();
                    }else{
                        console.log("Line 1020");
                    }
                }else{
                    console.log("Line 1023");
                    var newWPSymbol = createWPSymbol(index,QtPositioning.coordinate(
                                                         selectedWP.attributes.attributeValue("latitude"),
                                                         selectedWP.attributes.attributeValue("longitude"),
                                                         selectedWP.attributes.attributeValue("altitude")),
                                                     command,
                                                     param1,param2,param3,param4);
                    selectedWP.symbol = newWPSymbol;
                    recalcWPPathInAfterAdd(selectedWP);
                }
                selectedWP.attributes.replaceAttribute("command",command);
            }else{
                var newWPSymbol = createWPSymbol(index,QtPositioning.coordinate(
                                                     selectedWP.attributes.attributeValue("latitude"),
                                                     selectedWP.attributes.attributeValue("longitude"),
                                                     selectedWP.attributes.attributeValue("altitude")),
                                                 command,
                                                 param1,param2,param3,param4);
                selectedWP.symbol = newWPSymbol;
            }

            selectedWP.symbol.symbols.get(0).color = lstColor["selected"];
            console.log("now command "+selectedWP.attributes.attributeValue("command"));

            // update params
            selectedWP.attributes.replaceAttribute("param1",param1);
            selectedWP.attributes.replaceAttribute("param2",param2);
            selectedWP.attributes.replaceAttribute("param3",param3);
            selectedWP.attributes.replaceAttribute("param4",param4);

        }
    }
    function changeClickedPosition(position,visible){
        if(visible){
            var pointMer = Conv.latlongToMercator(position.latitude,
                                                  position.longitude);
            clickedBuilder.setXY(pointMer['x'],pointMer['y']);
            clickedGraphic.geometry = clickedBuilder.geometry;
        }
        clickedGraphic.visible = visible;
    }
    function changeWPPosition(selectedWP,command,position){
        if(selectedWP.attributes.attributeValue("command") !== lstWaypointCommand[vehicleType]["DO_JUMP"]["COMMAND"]){
            console.log("changeWPPosition to "+ position);
            console.log("selectedWP = "+selectedWP);
            selectedWP.attributes.replaceAttribute("latitude",position.latitude);
            selectedWP.attributes.replaceAttribute("longitude",position.longitude);
            selectedWP.attributes.replaceAttribute("altitude",position.altitude);
            var currentColor = selectedWP.symbol.symbols.get(0).color;
            var altitudeSymbol = ArcGISRuntimeEnvironment.createObject("TextSymbol",{
                                                                           color: "white",
                                                                           fontWidth: Enums.FontWeightBold,
                                                                           fontFamily: UIConstants.appFont,
                                                                           text: Number(position.altitude).toFixed(0).toString() +"m",
                                                                           size: wpFontSize,
                                                                           offsetX: 0,
                                                                           offsetY: - wpBoundSize /2 - wpFontSize / 2,
                                                                       });
            selectedWP.symbol = createWPSymbol(selectedWP.attributes.attributeValue("id"),
                                               position,command,
                                               selectedWP.attributes.attributeValue("param1"),
                                               selectedWP.attributes.attributeValue("param2"),
                                               selectedWP.attributes.attributeValue("param3"),
                                               selectedWP.attributes.attributeValue("param4"));

            var pointMer = Conv.latlongToMercator(position.latitude,
                                                  position.longitude);
            var point = ArcGISRuntimeEnvironment.createObject
                    ("Point",{
                         x: pointMer['x'],
                         y: pointMer['y'],
                         spatialReference: SpatialReference.createWebMercator()
                     }
                     );
            moveWPPosition(selectedWP,point);
            selectedWP.symbol.symbols.get(0).color = currentColor;
        }
    }
    function changeHomePosition(homePosition){
        console.log("Mappane changeHomePosition");
        for(var i = 0; i< mapOverlayWaypoints.graphics.count; i++){
            var graphicItem = mapOverlayWaypoints.graphics.get(i);
            if(graphicItem.attributes.containsAttribute("type") &&
                graphicItem.attributes.attributeValue("type") === "WP"){
                if(graphicItem.attributes.containsAttribute("id") &&
                        graphicItem.attributes.attributeValue("id") === 0){
                    if(selectedWP !== undefined &&
                            selectedWP.attributes.attributeValue("id") === 0){
                        confirmWP(homePosition,0,0,0,0);
                    }else{
                        changeWPPosition(graphicItem,
                                         graphicItem.attributes.attributeValue("command"),
                                         homePosition);
                    }
                }
            }
        }
    }

    function confirmWP(newPosition,param1,param2,param3,param4){
        console.log("confirmWP to "+newPosition);
        if(selectedWP !== undefined){
            var command = selectedWP.attributes.attributeValue("command");
            var id = selectedWP.attributes.attributeValue("id");
            console.log("WP["+id+"]"+command);
            if(selectedWP.attributes.attributeValue("id") !== 0){
                changeWPCommand(selectedWP.attributes.attributeValue("command"),
                            param1,param2,param3,param4);
            }
            if(selectedWP.attributes.attributeValue("id") === 0){
                if(newPosition.altitude !== selectedWP.attributes.attributeValue("altitude")){
                    homePositionChanged(selectedWP.attributes.attributeValue("latitude"),
                                    selectedWP.attributes.attributeValue("longitude"),
                                    newPosition.altitude);
                }
            }else{
                if(Number(selectedWP.attributes.attributeValue("longitude")).toFixed(6) !== Number(newPosition.longitude).toFixed(6) ||
                   Number(selectedWP.attributes.attributeValue("latitude")).toFixed(6) !== Number(newPosition.latitude).toFixed(6) ||
                   Number(selectedWP.attributes.attributeValue("altitude")).toFixed(0) !== Number(newPosition.altitude).toFixed(0) ){
                    isMapSync = false;
                }
            }

            changeWPPosition(selectedWP,command,newPosition);
            focusOnWP(id);
            updateWPDoJump();
            selectedWP.attributes.replaceAttribute("command_prev",
                            selectedWP.attributes.attributeValue("command"));
            selectedWP.attributes.replaceAttribute("latitude_prev",
                            selectedWP.attributes.attributeValue("latitude"));
            selectedWP.attributes.replaceAttribute("longitude_prev",
                            selectedWP.attributes.attributeValue("longitude"));
            selectedWP.attributes.replaceAttribute("altitude_prev",
                            selectedWP.attributes.attributeValue("altitude"));
            selectedWP.attributes.replaceAttribute("param1_prev",
                            selectedWP.attributes.attributeValue("param1"));
            selectedWP.attributes.replaceAttribute("param2_prev",
                            selectedWP.attributes.attributeValue("param2"));
            selectedWP.attributes.replaceAttribute("param3_prev",
                            selectedWP.attributes.attributeValue("param3"));
            selectedWP.attributes.replaceAttribute("param4_prev",
                            selectedWP.attributes.attributeValue("param4"));

        }
    }

    function restoreWP(){
        if(selectedWP !== undefined){
            var command = rootItem.selectedWP.attributes.attributeValue("command_prev");
            var id = selectedWP.attributes.attributeValue("id");
            console.log("WP["+id+"]"+command);
            if(selectedWP.attributes.attributeValue("id") !== 0){
                changeWPCommand(selectedWP.attributes.attributeValue("command_prev"),
                            selectedWP.attributes.attributeValue("param1_prev"),
                            selectedWP.attributes.attributeValue("param2_prev"),
                            selectedWP.attributes.attributeValue("param3_prev"),
                            selectedWP.attributes.attributeValue("param4_prev"));
            }
            changeWPPosition(selectedWP,command,QtPositioning.coordinate(
                                 selectedWP.attributes.attributeValue("latitude_prev"),
                                 selectedWP.attributes.attributeValue("longitude_prev"),
                                 selectedWP.attributes.attributeValue("altitude_prev")));
            focusOnWP(id);
            updateWPDoJump();

            selectedWP.attributes.replaceAttribute("command",
                            selectedWP.attributes.attributeValue("command_prev"));
            selectedWP.attributes.replaceAttribute("latitude",
                            selectedWP.attributes.attributeValue("latitude_prev"));
            selectedWP.attributes.replaceAttribute("longitude",
                            selectedWP.attributes.attributeValue("longitude_prev"));
            selectedWP.attributes.replaceAttribute("altitude",
                            selectedWP.attributes.attributeValue("altitude_prev"));
            selectedWP.attributes.replaceAttribute("param1",
                            selectedWP.attributes.attributeValue("param1_prev"));
            selectedWP.attributes.replaceAttribute("param2",
                            selectedWP.attributes.attributeValue("param2_prev"));
            selectedWP.attributes.replaceAttribute("param3",
                            selectedWP.attributes.attributeValue("param3_prev"));
            selectedWP.attributes.replaceAttribute("param4",
                            selectedWP.attributes.attributeValue("param4_prev"));
        }
    }

    function getCurrentListWaypoint(){
        var currentListWaypoint = [];
        var startIndex = 0;
        var tmpListWaypoint = listWaypoint;
        for(var id = 0; id < mapOverlayWaypoints.graphics.count ; id ++){
            var graphicItem = mapOverlayWaypoints.graphics.get(id);
            if(graphicItem.attributes.containsAttribute("type") &&
                    graphicItem.attributes.attributeValue("type") === "WP"){
//                console.log("graphic["+graphicItem.attributes.attributeValue("id")+"] command "+
//                            graphicItem.attributes.attributeValue("command"));
                if(graphicItem.geometry.extent!== null){
                    var wpLatLon =  Conv.mercatorToLatLon(graphicItem.geometry.extent.center.x,
                                                          graphicItem.geometry.extent.center.y)
                    var missionItem = myComponent.createObject(rootItem);

                    missionItem.frame = graphicItem.attributes.attributeValue("id") === 0?0:3;
    //                console.log("getCurrentListWaypoint ["+graphicItem.attributes.attributeValue("id")+"]command "+graphicItem.attributes.attributeValue("command"));
                    missionItem.command = graphicItem.attributes.attributeValue("command");
                    missionItem.param1 = graphicItem.attributes.attributeValue("param1");
                    missionItem.param2 = graphicItem.attributes.attributeValue("param2");
                    missionItem.param3 = graphicItem.attributes.attributeValue("param3");
                    missionItem.param4 = graphicItem.attributes.attributeValue("param4");
                    missionItem.param5 = graphicItem.attributes.attributeValue("latitude");
                    missionItem.param6 = graphicItem.attributes.attributeValue("longitude");
                    missionItem.param7 = graphicItem.attributes.attributeValue("altitude");
                    missionItem.sequence = graphicItem.attributes.attributeValue("id");
                    currentListWaypoint.push(missionItem)
                }
            }
        }

        return currentListWaypoint;
    }
    function nextWaypoint(wpID){

    }

    function convertLocationToScreen(latitude,longitude){
        var pointMer =  Conv.latlongToMercator(latitude,
                                              longitude)
        var point = ArcGISRuntimeEnvironment.createObject("Point", {
                                  x: pointMer['x'],
                                  y: pointMer['y'],
                                  spatialReference: SpatialReference.createWebMercator()
                              });
        var pointOnScreen = mapView.locationToScreen(point);
        return pointOnScreen;
    }
    function hideProfilePath()
    {
        rectProfilePath.visible = false;
    }
    function updateMouseOnMap(){

        if(UIConstants.mouseOnMapMode === UIConstants.mouseOnMapModeWaypoint){
            if(selectedWP !== undefined){
                var point = ArcGISRuntimeEnvironment.createObject("Point", {
                                                                              x: rootItem.selectedWP.geometry.extent.center.x,
                                                                              y: rootItem.selectedWP.geometry.extent.center.y,
                                                                              spatialReference: SpatialReference.createWebMercator()
                                                                          });
                var mouseOnMap = mapView.locationToScreen(point);
                rectDragWP.x = mouseOnMap.x - rectDragWP.width/2;
                rectDragWP.y = mouseOnMap.y - rectDragWP.height/2;
                waypointEditor.changeState()
            }
        }else if(UIConstants.mouseOnMapMode === UIConstants.mouseOnMapModeMeasure){
            var point = ArcGISRuntimeEnvironment.createObject("Point", {
                                                                          x: mouseBuilder.geometry.extent.center.x,
                                                                          y: mouseBuilder.geometry.extent.center.y,
                                                                          spatialReference: SpatialReference.createWebMercator()
                                                                      });
            var mouseOnMap = mapView.locationToScreen(point);
            rectDragDistance.x = mouseOnMap.x - rectDragDistance.width/2;
            rectDragDistance.y = mouseOnMap.y - rectDragDistance.height/2;
        }
        // update marker location
        if(selectedMarker !== undefined){
            var point = ArcGISRuntimeEnvironment.createObject("Point", {
                                                                          x: rootItem.selectedMarker.geometry.extent.center.x,
                                                                          y: rootItem.selectedMarker.geometry.extent.center.y,
                                                                          spatialReference: SpatialReference.createWebMercator()
                                                                      });
            var mouseOnMap = mapView.locationToScreen(point);
            rectDragMarker.x = mouseOnMap.x - rectDragMarker.width/2;
            rectDragMarker.y = mouseOnMap.y - rectDragMarker.height/2;
            markerEditor.changeState()
        }
    }
    // distance line
    function createDistanceLine(point1,point2){
        var listGraphics = [];
        var angle = -Math.atan((point1.y-point2.y)/(point1.x-point2.x))*180/Math.PI;
        var distanceResult = GeometryEngine.distanceGeodetic(point1,point2,Enums.LinearUnitIdMeters,
                                                             Enums.AngularUnitIdDegrees     ,
                                                             Enums.GeodeticCurveTypeNormalSection);
        console.log("distanceResult.distance = "+distanceResult.distance)
        console.log("distanceResult.azimuth1 = "+distanceResult.azimuth1)
        console.log("distanceResult.azimuth2 = "+distanceResult.azimuth2)
        var headSymbol = ArcGISRuntimeEnvironment.createObject("SimpleMarkerSymbol",{
                                                                   //antiAlias : true,
                                                                   style: Enums.SimpleMarkerSymbolStyleCircle,
                                                                   color : "red",
                                                                   size : 4
                                                               });
        var lineSymbol = ArcGISRuntimeEnvironment.createObject("SimpleLineSymbol",{
                                                                   antiAlias : true,
                                                                   style: Enums.SimpleLineSymbolStyleSolid,
                                                                   color : "red",
                                                                   width : 2
                                                               });
        var lineJson = {"paths":
                [[
                    [point1.x,point1.y],
                    [point2.x,point2.y]
                ]]
        };

        var lineGeometry = ArcGISRuntimeEnvironment.createObject("Polyline", {
                                                                     "json": lineJson,
                                                                     spatialReference: SpatialReference.createWebMercator()
                                                                 });
        var distanceSymbol = ArcGISRuntimeEnvironment.createObject("TextSymbol",{
                                                                       angle: angle,
                                                                       color: "black",
                                                                       fontWidth: Enums.FontWeightBold,
                                                                       fontFamily: UIConstants.appFont,
                                                                       text: Number(distanceResult.azimuth1 > 0?
                                                                                        distanceResult.azimuth1:
                                                                                        distanceResult.azimuth1 + 360).toFixed(2) + UIConstants.degreeSymbol+" "+parseInt(GeometryEngine.length(lineGeometry))+"m",
                                                                       size: wpFontSize,
                                                                       haloColor: "white",
                                                                       haloWidth: 2,
                                                                       verticalAlignment: Enums.VerticalAlignmentBottom
                                                                   });
        var distancePoint = ArcGISRuntimeEnvironment.createObject("Point", {
                                                                      x: (point1.x + point2.x)/2 ,
                                                                      y: (point1.y + point2.y)/2,
                                                                      spatialReference: SpatialReference.createWebMercator()
                                                                  });
        var point1Geo = ArcGISRuntimeEnvironment.createObject("Point", {
                                                                  x: point1.x,
                                                                  y: point1.y,
                                                                  spatialReference: SpatialReference.createWebMercator()
                                                              });
        var point2Geo = ArcGISRuntimeEnvironment.createObject("Point", {
                                                                  x: point2.x,
                                                                  y: point2.y,
                                                                  spatialReference: SpatialReference.createWebMercator()
                                                              });
        listGraphics.push(createGraphic(lineGeometry,lineSymbol));
        listGraphics.push(createGraphic(point1Geo,headSymbol));
        listGraphics.push(createGraphic(point2Geo,headSymbol));
        listGraphics.push(createGraphic(distancePoint,distanceSymbol));
        return listGraphics;
    }
    function removeLastMeasurementLine(){
        var numGraphicObjects = mapOverlayDistance.graphics.count;
        for(var x = numGraphicObjects -1; x >=numGraphicObjects -4  ;x--){
            mapOverlayDistance.graphics.remove(x);
        }
    }
    function removeAllMeasurementLine(){
        for(var x = mapOverlayDistance.graphics.count -1; x >=0  ; x--){
             mapOverlayDistance.graphics.remove(x);
        }
    }

    function findSelectedMarkerID(listGraphicMarker){
        var selectedMarkerID = -1;
        for(var id = 0; id < listGraphicMarker.length ; id ++){
            if(listGraphicMarker[id].selected){
                selectedMarkerID = id;
                break;
            }
        }
        return selectedMarkerID;
    }
    function deleteMarker(listGraphicMarker,type,markerID){
        var lstMarkerIndex = findListMarker(listGraphicMarker,type,markerID);
        for(var id = lstMarkerIndex.length-1; id >=0 ; id --)
            listGraphicMarker.remove(lstMarkerIndex[id]);
    }
    function addMarker(){
        var point = ArcGISRuntimeEnvironment.createObject("Point", {
                                                              x: mouseBuilder.geometry.extent.center.x,
                                                              y: mouseBuilder.geometry.extent.center.y
                                                          });
        var markerIcon = createMarker(point,"MARKER_DEFAULT","Default marker");

        //console.log(markerIcon+":"+markerIcon.attributes.attributeValue("type")+":["+point.x+","+point.y+"]");
        mapOverlayWaypoints.graphics.append(markerIcon);

        // add marker to list marker control
        listGraphicMarker.push(markerIcon);
    }
    function removeMarker(){
        console.log("Remove marker");
        if(selectedMarker === undefined){
            return;
        }
        for(var markerID = listGraphicMarker.length-1; markerID >= 0; markerID --){
            if(listGraphicMarker[markerID].selected === true){
                listGraphicMarker.splice(markerID,1)
            }
        }
        deleteMarker(mapOverlayWaypoints.graphics,"marker",selectedMarker.attributes.attributeValue("id"));
        selectedMarker = undefined;
    }

    function addMarkerWGS84(latitude,longitude,type,description){
        var pointMer = Conv.latlongToMercator(latitude,longitude);
        var point2Geo = ArcGISRuntimeEnvironment.createObject("Point",
                                                                {
                                                                    x: pointMer['x'],
                                                                    y: pointMer['y'],
                                                                    spatialReference: SpatialReference.createWebMercator()
                                                                });
        var markerIcon = createMarker(point2Geo,type,description);
        mapOverlayWaypoints.graphics.append(markerIcon);
    }

    function createMarker(point,type,description){
        var id = (new Date()).getTime();
        var symbol = createMarkerSymbol(type,description);
        var markerIcon = ArcGISRuntimeEnvironment.createObject
                ("Graphic",
                 {geometry: point, symbol: symbol}
                 );
        var locationWGS84 = Conv.mercatorToLatLon(
                    point.x,
                    point.y);
        markerIcon.attributes.insertAttribute("id",id);
        markerIcon.attributes.insertAttribute("type","marker");
        markerIcon.attributes.insertAttribute("kind",type);
        markerIcon.attributes.insertAttribute("kind_next",markerIcon.attributes.attributeValue("kind"));
        markerIcon.attributes.insertAttribute("latitude",locationWGS84['lat']);
        markerIcon.attributes.insertAttribute("longitude",locationWGS84['lon']);
        markerIcon.attributes.insertAttribute("description",description);
        return markerIcon;
    }
    function modifyMarker(listGraphicMarker,newType,markerID){
        var lstMarkerIndex = findListMarker(listGraphicMarker,"marker",markerID);
        for(var id = lstMarkerIndex.length-1; id >=0 ; id --){
            var oldMarker = listGraphicMarker.get(lstMarkerIndex[id]);
            console.log("oldMarker id "+lstMarkerIndex[id]);
            changeMarkerType(oldMarker,newType);
        }
    }
    function confirmMarker(listGraphicMarker,markerID,coordinate){
        var lstMarkerIndex = findListMarker(listGraphicMarker,"marker",markerID);
        for(var id = lstMarkerIndex.length-1; id >=0 ; id --){
            var oldMarker = listGraphicMarker.get(lstMarkerIndex[id]);
            var oldMarkerKind = oldMarker.attributes.attributeValue("kind");
            var nextMarkerKind = oldMarker.attributes.attributeValue("kind_next");
            var oldMarkerID = oldMarker.attributes.attributeValue("id");
            console.log("Confirm marker "+coordinate.latitude+","+coordinate.longitude);
            var nextPointMer = Conv.latlongToMercator(
                        coordinate.latitude,
                        coordinate.longitude);
            var point = ArcGISRuntimeEnvironment.createObject("Point",{
                        x: nextPointMer['x'],
                        y: nextPointMer['y'],
                        spatialReference: SpatialReference.createWebMercator()
                     });
            oldMarker.geometry = point;
            oldMarker.attributes.replaceAttribute("kind",nextMarkerKind);
            oldMarker.attributes.replaceAttribute("kind_next",nextMarkerKind);
            oldMarker.attributes.replaceAttribute("latitude",coordinate.latitude);
            oldMarker.attributes.replaceAttribute("longitude",coordinate.longitude);
            oldMarker.attributes.replaceAttribute("description",oldMarker.symbol.symbols.get(1).text);
        }
    }
    function restoreMarker(listGraphicMarker,markerID){
        console.log("restore marker");
        var lstMarkerIndex = findListMarker(listGraphicMarker,"marker",markerID);
        for(var id = lstMarkerIndex.length-1; id >=0 ; id --){
            var oldMarker = listGraphicMarker.get(lstMarkerIndex[id]);
            var oldMarkerKind = oldMarker.attributes.attributeValue("kind");
            oldMarker.attributes.replaceAttribute("kind",oldMarkerKind);
            oldMarker.attributes.replaceAttribute("kind_next",oldMarkerKind);

            var pointMer =  Conv.latlongToMercator(oldMarker.attributes.attributeValue("latitude"),
                                                  oldMarker.attributes.attributeValue("longitude"))
            var point = ArcGISRuntimeEnvironment.createObject("Point", {
                                      x: pointMer['x'],
                                      y: pointMer['y'],
                                      spatialReference: SpatialReference.createWebMercator()
                                  });
            oldMarker.geometry = point;
            changeMarkerType(oldMarker,oldMarkerKind);
            selectedMarker.symbol.symbols.get(1).text = selectedMarker.attributes.attributeValue("description")
        }
    }
    function setFocus(enable){
        mapView.focus = enable;
    }
    function clearFlightPath(){
        var partTrajectory = trajectory.parts.part(0);
        if(partTrajectory === null){
            console.log("part not exist");
        }else{
            console.log("trajectory has "+partTrajectory.pointCount+" point");
            if(partTrajectory.pointCount > 0){
                partTrajectory.removeAll();
            }
        }
    }

    function focusOnMarker(direction){
        var selectedMarkerID = findSelectedMarkerID(listGraphicMarker);
        if(selectedMarker !== undefined){
            selectedMarker.selected = false;
        }
        console.log("lstGraphicMarker.length="+listGraphicMarker.length)
        var nextSelectedMarkerID = selectedMarkerID;
        if(nextSelectedMarkerID < 0){
            if(listGraphicMarker.length <= 0){
                return;
            }else{
                nextSelectedMarkerID = 0;
            }
        }else{
            if(direction === "next"){
                nextSelectedMarkerID++;
                if(nextSelectedMarkerID >= listGraphicMarker.length){
                    nextSelectedMarkerID = 0;
                }
            }else if(direction === "prev"){
                nextSelectedMarkerID -- ;
                if(nextSelectedMarkerID < 0){
                    nextSelectedMarkerID = listGraphicMarker.length-1;
                }
            }

        }
        console.log("focusOnMarker "+nextSelectedMarkerID);
        var marker = listGraphicMarker[nextSelectedMarkerID];
        marker.selected = true;
        selectedMarker = marker;
        var pointLatLon = Conv.mercatorToLatLon(marker.geometry.extent.center.x,
                                                 marker.geometry.extent.center.y);

        focusOnPosition(pointLatLon['lat'],
                        pointLatLon['lon']);
    }
    Component {
        id: myComponent
        MissionItem{
            id: missionSample
        }
    }

    MapView {
        id: mapView
        anchors.fill: parent
        wrapAroundMode: Enums.WrapAroundModeEnabledWhenSupported        
        rotationByPinchingEnabled: false
        Keys.onPressed: {
            if(event.key === Qt.Key_A || event.key === Qt.Key_D){
//                console.log("event.key === Qt.Key_A || event.key === Qt.Key_D");
                mapView.setViewpointRotation(0);
            }else if(event.key === Qt.Key_S){
                focusOnUAV();
            }else if(event.key === Qt.Key_M){
                focusAllObject();
            }else if(event.key === Qt.Key_Plus || event.key === Qt.Key_Equal){
                zoomIn();
            }else if(event.key === Qt.Key_Minus || event.key === Qt.Key_Underscore){
                zoomOut();
            }else if(event.key === Qt.Key_Backspace){
                if(rootItem.ctrlPress === true){
                    console.log("removeAllMeasurementLine");
                    rootItem.removeAllMeasurementLine();
                    rootItem.ctrlPress = false;
                }else{
                    console.log("removeLastMeasurementLine");
                    rootItem.removeLastMeasurementLine();
                }
            }else if(event.key === Qt.Key_Control){
                rootItem.ctrlPress = true;
            }else if(event.key === Qt.Key_Equal || event.key === Qt.Key_Minus ||
                     event.key === Qt.Key_Up || event.key === Qt.Key_Down ||
                     event.key === Qt.Key_Right || event.key === Qt.Key_left ){
                rootItem.setFocus(true);
            }else if(event.key >= Qt.Key_0 && event.key <= Qt.Key_9){
                rootItem.selectedIndex = event.key - Qt.Key_0
                focusOnWP(event.key - Qt.Key_0);
            }else if(event.key === Qt.Key_Q){
                focusOnMarker("next");
                rootItem.setFocus(true);
            }else if(event.key === Qt.Key_C){
                clearFlightPath();
            }else if(rootItem.ctrlPress && event.key === Qt.Key_F6){
                rootItem.showAdvancedConfigChanged();
                console.log("showAdvancedConfigChanged");
                rootItem.ctrlPress = false;
            }

        }
        Map {
            id: map
            BasemapImagery {
               id: basemap
            }
            spatialReference: SpatialReference.createWebMercator()
        }
        GraphicsOverlay {
            id: mapOverlayFOV
            renderingMode: Enums.GraphicsRenderingModeDynamic
            PolygonBuilder {
                id: planeFOVBuilder
                spatialReference: SpatialReference.createWgs84()
            }

            // symbol for nesting ground
            Graphic{
                id: planeFOVGraphic
                symbol: SimpleFillSymbol {
                    id: nestingGroundSymbol
                    style: Enums.SimpleFillSymbolStyleSolid
                    color: Qt.rgba(67.0/255.0, 223.0/255.0, 101.0/255.0, 0.75)
                    // default property: ouline
                    /**/
                    SimpleLineSymbol {
                        style: Enums.SimpleLineSymbolStyleSolid
                        color: "black"
                        width: 1
                        antiAlias: true
                    }

                }
            }

            PolylineBuilder{
                id: centralFOVBuilder
                spatialReference: SpatialReference.createWgs84()
            }
            Graphic {
                id: centralFOVGraphic
                symbol: SimpleMarkerSymbol{
                    style: Enums.SimpleMarkerSymbolStyleCross
                    color: "black"
                    size: 1
                    angle: 45
                }
            }

            Graphic {
                id: centralFOVGraphicLine
                symbol: SimpleLineSymbol{
                    style: Enums.SimpleLineSymbolStyleSolid
                    color: "black"
                    width: 1
                    //antiAlias: true
                }
            }
        }
        GraphicsOverlay {
            id: mapOverlayMouse
            PointBuilder{
                id: mouseBuilder
                spatialReference: SpatialReference.createWebMercator()
            }
            Graphic {
                id: mouseGraphic
                symbol: SimpleMarkerSymbol{
                    style: Enums.SimpleFillSymbolStyleCross
                    size: 10
                    color: "red"
                    angle: 45
                }
            }
            PointBuilder{
                id: homeBuilder
                spatialReference: SpatialReference.createWebMercator()
            }
            Graphic {
                id: homeGraphic
                symbol: CompositeSymbol{
                    SimpleMarkerSymbol{
                        style: Enums.SimpleMarkerSymbolStyleCircle
                        size: wpBoundSize*2
                        color: "transparent"
                        outline: SimpleLineSymbol{
                            width: 1
                            color: "blue"
                        }
                    }
                    PictureMarkerSymbol{
                        url: vehicleSymbolLink["MAV_TYPE_GENERIC"]
                        angle: 20
                        width: wpBoundSize * 2
                        height: wpBoundSize * 2
                    }
                }
            }
        }
        GraphicsOverlay {
            id: mapOverlayDistance
            renderingMode: Enums.GraphicsRenderingModeDynamic
            selectionColor: "yellow"
        }
        GraphicsOverlay {
            id: mapOverlayWaypointsPath
            renderingMode: Enums.GraphicsRenderingModeDynamic
            PolygonBuilder{
                id: waypointPathBuilder
                spatialReference: SpatialReference.createWebMercator()
            }
            Graphic {
                id: waypointPathGraphic
                symbol: SimpleLineSymbol{
                    style: Enums.SimpleLineSymbolStyleSolid
                    antiAlias : true
                    color : lstColor["normal"]
                    width : 1
                }
            }
        }
        GraphicsOverlay {
            id: mapOverlayWaypoints
            renderingMode: Enums.GraphicsRenderingModeDynamic
            selectionColor: "yellow"
        }
        GraphicsOverlay{
            id: mapOverlayGotoLocation
            PointBuilder{
                id: clickedBuilder
                spatialReference: SpatialReference.createWebMercator()
            }
            Graphic {
                id: clickedGraphic
                symbol: CompositeSymbol{
                    SimpleMarkerSymbol{
                        style: Enums.SimpleMarkerSymbolStyleCircle
                        color: lstColor["normal"]
                        size : wpBoundSize
                    }
                    TextSymbol{
                        color: "white"
                        fontWeight: Enums.FontWeightBold
                        fontFamily: UIConstants.appFont
                        text: "G"
                        size: wpFontSize
                    }
                }
            }
        }

        GraphicsOverlay {
            id: mapOverlayVehicle

            PolylineBuilder{
                id: trajectory
                spatialReference: SpatialReference.createWebMercator()
            }

            Graphic {
                id: trajectoryGraphic
                SimpleLineSymbol {
                    style: Enums.SimpleLineSymbolStyleSolid
                    color: "red"
                    width: 1
                }

            }

            PointBuilder{
                id: uavBuilder
            }

            Graphic {
                id: uavGraphic
                symbol: PictureMarkerSymbol{
                    url: vehicleSymbolLink["MAV_TYPE_GENERIC"]
                    opacity: 1
                    width: wpBoundSize * 1.5
                    height: wpBoundSize * 1.5
                }
            }
            Graphic{
                symbol: CompositeSymbol{
                }
            }
            Graphic{
                symbol: TextSymbol{

                }
            }

        }

        onViewpointChanged: {
            rootItem.mapMoved();
            updateMouseOnMap()
        }

        onMousePressed: {
            var point = Qt.point(mouse.mapPoint.x,
                                mouse.mapPoint.y);
            var latlon = Conv.mercatorToLatLon(mouse.mapPoint.x,
                                          mouse.mapPoint.y);
            var lat = latlon['lat'];
            var lon = latlon['lon'];
            clickedLocation.latitude = lat;
            clickedLocation.longitude = lon;
            var amsl = elevationFinder.getAltitude(
                        cInfo.homeFolder()+"/ArcGIS/Runtime/Data/elevation/"+mapHeightFolder,
                        latlon['lat'],latlon['lon']);
//            console.log("amsl = "+amsl);
            setFocus(true);
            if(!rootItem.mousePressed) rootItem.mousePressed = true;
            mouseBuilder.setXY(mouse.mapX,mouse.mapY);
            if(UIConstants.mouseOnMapMode === UIConstants.mouseOnMapModeWaypoint){
                mouseGraphic.geometry = mouseBuilder.geometry;
//                mapView.identifyGraphicsOverlay(mapOverlayWaypoints, mouse.x, mouse.y, 1, false,1);
                console.log("identifyGraphicsOverlay at ("+mouse.x+","+mouse.y+")");
                identifyGraphic(mapOverlayWaypoints.graphics,mouse.x,mouse.y);
            }else if(UIConstants.mouseOnMapMode === UIConstants.mouseOnMapModeMeasure){

            }
            updateMouseOnMap();
            console.log("Mouse pressed changed "+ mouse.button);
            mouseButton = mouse.button;
        }
        onMouseReleased: {
            if(mouseButton === Qt.RightButton && startLine.length > 0){
                startLine = [];
            }
            mouseButton = 0;
        }

        onMousePositionChanged: {
            if(UIConstants.mouseOnMapMode === UIConstants.mouseOnMapModeWaypoint){
                rootItem.mapMoved();
                updateMouseOnMap()
            }else if(UIConstants.mouseOnMapMode === UIConstants.mouseOnMapModeMeasure){

            }
            if(mouseButton === Qt.RightButton){
                var pointEnd = ArcGISRuntimeEnvironment.createObject("Point", {
                                                                              x: mouse.mapPoint.x,
                                                                              y: mouse.mapPoint.y,
                                                                              spatialReference: SpatialReference.createWebMercator()
                                                                          });
                if(startLine.length > 0){
                    var pointStart = startLine[0];
                    var lineGraphics = createDistanceLine(pointStart,pointEnd);
                    var numGraphic = mapOverlayDistance.graphics.count;
                    deleteMarker(mapOverlayDistance.graphics,"distance",lineID);
                    for(var i = 0; i < lineGraphics.length; i++){
                        lineGraphics[i].attributes.insertAttribute("id",lineID);
                        lineGraphics[i].attributes.insertAttribute("type","distance");
                        mapOverlayDistance.graphics.append(lineGraphics[i]);
                    }
                }else{
                    if(startLine.length == 0){
                        startLine.push(pointEnd);
                        lineID ++;
                    }
                }
            }
        }
        onMapScaleChanged: {

            rootItem.mapMoved();
            lblMapScaleValue.text = "1:"+(isNaN(mapView.mapScale)?"1":Number(mapView.mapScale).toFixed(0).toString());
            updateMouseOnMap();
            // update do jump wp
            updateWPDoJump();
        }
//        onMousePressedAndHeld: {
//            timerPressAndHold.start()
//        }
        Rectangle{
            id: rectDragMarker
            x: 100
            y: 100
            height: wpBoundSize + 20
            width: height
            radius: height/2
            color: UIConstants.transparentColor
            visible: selectedMarker !== undefined
//            visible: rootItem.mousePressed && (UIConstants.mouseOnMapMode === UIConstants.mouseOnMapModeMeasure)

            MouseArea{
                anchors.fill: parent
                drag.target: parent
                onPositionChanged: {
                    var center = Qt.point(parent.x + parent.width/2,
                                          parent.y + parent.height/2);
                    var newLocation = mapView.screenToLocation(center.x,center.y);
                    moveMarkerPosition(selectedMarker,newLocation);
                    var altitude = 0;
                    var locationWGS84 = Conv.mercatorToLatLon(
                                selectedMarker.geometry.extent.center.x,
                                selectedMarker.geometry.extent.center.y);
                    var markerCoordinate = QtPositioning.coordinate(
                                locationWGS84['lat'],
                                locationWGS84['lon'],
                                altitude);
                    selectedMarker.attributes.replaceAttribute("latitude",locationWGS84['lat']);
                    selectedMarker.attributes.replaceAttribute("longitude",locationWGS84['lon']);
                    selectedMarker.attributes.replaceAttribute("latitude_prev",locationWGS84['lat']);
                    selectedMarker.attributes.replaceAttribute("longitude_prev",locationWGS84['lon']);
                    var asl = elevationFinder.getAltitude(
                                cInfo.homeFolder()+"/ArcGIS/Runtime/Data/elevation/"+mapHeightFolder,
                                locationWGS84['lat'],locationWGS84['lon']);
                    markerEditor.asl = asl;
                    markerEditor.changeCoordinate(markerCoordinate);
                }
                onPressAndHold: {
                    if(!markerEditor.visible)
                        timerPressAndHold.start()
                }
                onReleased: {
                    timerPressAndHold.stop();
                    cvsMarker.angle = 0;
                    cvsMarker.requestPaint();
                }
            }
            Canvas {
                id: cvsMarker
                anchors.fill: parent
                contextType: "2d"
                antialiasing: true
                property real angle: 0
                onPaint: {
                    var ctx = getContext('2d');
                    ctx.reset();

                    var x = width / 2
                    var y = height / 2
                    var losSize = width /2 - 4;
                    ctx.strokeStyle = "red";
                    ctx.beginPath();
                    ctx.lineWidth = 4;
                    ctx.arc(x, y, losSize,
                            0,
                            angle/180*Math.PI, false)
                    context.stroke();
                }
                Timer{
                    id: timerPressAndHold
                    interval: 30
                    repeat: true
                    running: false
                    onTriggered: {
                        cvsMarker.angle +=12;
                        cvsMarker.requestPaint();
                        if(cvsMarker.angle > 360){
                            cvsMarker.angle = 0;
                            stop();
                        }
                    }
                    onRunningChanged: {
                        if(running == false){
                            markerEditor.visible = true;
                            markerEditor.changeState();
                        }
                    }
                }
            }

            MarkerEditor{
                id: markerEditor
                x: parent.height + 10
                visible: false
                function changeState(){
                    if(selectedMarker!= undefined){
                        var altitude = 0;

                        var locationWGS84 = Conv.mercatorToLatLon(
                                    selectedMarker.geometry.extent.center.x,
                                    selectedMarker.geometry.extent.center.y);
                        var markerCoordinate = QtPositioning.coordinate(
                                    locationWGS84['lat'],
                                    locationWGS84['lon'],
                                    altitude);
                        var asl = elevationFinder.getAltitude(
                                    cInfo.homeFolder()+"/ArcGIS/Runtime/Data/elevation/"+mapHeightFolder,
                                    locationWGS84['lat'],locationWGS84['lon']);
                        markerEditor.asl = asl;
                        markerEditor.loadInfo(
                                    markerCoordinate,
                                    getMarkerType(selectedMarker),
                                    getMarkerText(selectedMarker)
                                    );
                    }
                }

                onMarkerIDChanged: {
                    changeMarkerType(selectedMarker,markerType);
                }
                onConfirmClicked: {
                    if(rootItem.selectedMarker!=undefined){
                        var id = selectedMarker.attributes.attributeValue("id");
                        var nextPointMer = Conv.latlongToMercator(
                                    coordinate.latitude,
                                    coordinate.longitude);
                        var point = ArcGISRuntimeEnvironment.createObject("Point",{
                                     x: nextPointMer['x'],
                                     y: nextPointMer['y'],
                                     spatialReference: SpatialReference.createWebMercator()
                                 });

                        confirmMarker(mapOverlayWaypoints.graphics,id,coordinate);

//                        var listMarkerID = findListMarker(mapOverlayWaypoints.graphics,"marker",id);
                        selectedMarker.selected = false;
                        selectedMarker = undefined;
                    }
                    markerEditor.visible = false;
                }
                onCancelClicked: {
                    restoreMarker(mapOverlayWaypoints.graphics,
                                  selectedMarker.attributes.attributeValue("id"));
                    markerEditor.visible = false;
                }
                onTextChanged: {
                    setMarkerText(selectedMarker,newText);
                }
            }
        }


        Rectangle{
            id: rectDragDistance
            x: 100
            y: 100
            height: wpBoundSize + 10
            width: height
            radius: height/2
            color: UIConstants.transparentRed
            visible: rootItem.mousePressed && (UIConstants.mouseOnMapMode === UIConstants.mouseOnMapModeMeasure)

            MouseArea{
                anchors.fill: parent
                drag.target: parent
                onPositionChanged: {
                    var center = Qt.point(parent.x + parent.width/2,
                                          parent.y + parent.height/2);
                    var pointEnd = mapView.screenToLocation(center.x,center.y);
                    if(startLine.length > 0){
                        var pointStart = startLine[0];
                        var lineGraphics = createDistanceLine(pointStart,pointEnd);
                        var numGraphic = mapOverlayDistance.graphics.count;
                        deleteMarker(mapOverlayDistance.graphics,"distance",lineID);
                        for(var i = 0; i < lineGraphics.length; i++){
                            lineGraphics[i].attributes.insertAttribute("id",lineID);
                            lineGraphics[i].attributes.insertAttribute("type","distance");
                            mapOverlayDistance.graphics.append(lineGraphics[i]);
                        }
                    }else{
                        if(startLine.length == 0){
                            startLine.push(pointEnd);
                            lineID ++;
                        }
                    }
                }
                onReleased: {
                    if(startLine.length > 0){
                        var center = Qt.point(parent.x + parent.width/2,
                                              parent.y + parent.height/2);
                        var pointStart = startLine[0];
                        var pointEnd = mapView.screenToLocation(center.x,center.y);
                        var startLineLatLon = Conv.mercatorToLatLon(pointStart.x,pointStart.y);
                        var endLineLatLon = Conv.mercatorToLatLon(pointEnd.x,pointEnd.y);

                        var rulercoord1 = QtPositioning.coordinate(
                                    startLineLatLon['lat'],startLineLatLon['lon'],0);
                        var rulercoord2 = QtPositioning.coordinate(
                                    endLineLatLon['lat'],endLineLatLon['lon'],0);
                        profilePath.addElevation(
                                    rulercoord1,rulercoord2);
                        rectProfilePath.visible = true;
                        startLine = [];
                    }
                }
            }
        }

        Rectangle{
            id: rectDragWP
            x: 100
            y: 100
            height: wpBoundSize + 20
            width: height
            radius: height/2
            color: UIConstants.transparentBlue
            visible: selectedWP !== undefined && (UIConstants.mouseOnMapMode === UIConstants.mouseOnMapModeWaypoint)

            MouseArea{
                anchors.fill: parent
                drag.target: parent
                onPressed: {
                    if(selectedWP !== undefined &&
                        selectedWP.attributes.attributeValue("command") !== lstWaypointCommand[vehicleType]["DO_JUMP"]["COMMAND"]){
                        console.log("set drag target to parent");
                        drag.target = parent;
                    }else{
                        console.log("undefined drag target");
                        drag.target = undefined;
                    }
                }

                onClicked: {
                    var point = mapView.screenToLocation(mouse.x, mouse.y);
                    var mouseOnMap = mapView.locationToScreen(point);
                    if(selectedWP !== undefined){
                        waypointEditor.changeState()
                    }

                }
                onPositionChanged: {
                    if(selectedWP !== undefined &&
                        selectedWP.attributes.attributeValue("command") !== lstWaypointCommand[vehicleType]["DO_JUMP"]["COMMAND"]){
                        var center = Qt.point(parent.x + parent.width/2,
                                              parent.y + parent.height/2);
                        var newWPPosition = mapView.screenToLocation(center.x,center.y);
                        moveWPPosition(selectedWP,newWPPosition);
                        var altitude = selectedWP.attributes.attributeValue("altitude");
                        var locationWGS84 = Conv.mercatorToLatLon(
                                    newWPPosition.x,
                                    newWPPosition.y);
                        var waypointCoordinate = QtPositioning.coordinate(
                                    locationWGS84['lat'],
                                    locationWGS84['lon'],
                                    altitude);
                        selectedWP.attributes.replaceAttribute("latitude",locationWGS84['lat']);
                        selectedWP.attributes.replaceAttribute("longitude",locationWGS84['lon']);
                        selectedWP.attributes.replaceAttribute("latitude_prev",locationWGS84['lat']);
                        selectedWP.attributes.replaceAttribute("longitude_prev",locationWGS84['lon']);
                        var asl = elevationFinder.getAltitude(
                                    cInfo.homeFolder()+"/ArcGIS/Runtime/Data/elevation/"+mapHeightFolder,
                                    locationWGS84['lat'],locationWGS84['lon']);
                        waypointEditor.changeASL(asl);
                        waypointEditor.changeCoordinate(waypointCoordinate);
                        updateWPDoJump();
                        if(selectedWP.attributes.attributeValue("id") > 0)
                            isMapSync = false;
                    }
                }
                onPressAndHold: {
                    if(!waypointEditor.visible)
                        timerPressAndHoldWaypoint.start()
                }
                onReleased: {
                    timerPressAndHoldWaypoint.stop();
                    cvsWaypoint.angle = 0;
                    cvsWaypoint.requestPaint();
                    if(selectedWP !== undefined &&
                        selectedWP.attributes.attributeValue("id") === 0){
                        homePositionChanged(selectedWP.attributes.attributeValue("latitude"),
                                            selectedWP.attributes.attributeValue("longitude"),
                                            selectedWP.attributes.attributeValue("altitude"));
                    }
                }
            }
            Canvas {
                id: cvsWaypoint
                anchors.fill: parent
                contextType: "2d"
                antialiasing: true
                property real angle: 0
                onPaint: {
                    var ctx = cvsWaypoint.getContext('2d');
                    ctx.reset();

                    var x = width / 2
                    var y = height / 2
                    var losSize = width /2 - 4;
                    ctx.strokeStyle = "red";
                    ctx.beginPath();
                    ctx.lineWidth = 4;
                    ctx.arc(x, y, losSize,
                            0,
                            angle/180*Math.PI, false)
                    context.stroke();
                }
                Timer{
                    id: timerPressAndHoldWaypoint
                    interval: 30
                    repeat: true
                    running: false
                    onTriggered: {
                        cvsWaypoint.angle +=12;
                        cvsWaypoint.requestPaint();
                        if(cvsWaypoint.angle > 360){
                            cvsWaypoint.angle = 0;
                            stop();
                        }
                    }
                    onRunningChanged: {
                        if(running == false){
                            waypointEditor.visible = true;
                            waypointEditor.changeState();
                        }
                    }
                }
            }
            WaypointEditor{
                id: waypointEditor
                x: parent.height + 10
                anchors.verticalCenter: parent.verticalCenter
                anchors.verticalCenterOffset: (height / 2 + parent.y+ parent.height/2) > rootItem.height ?
                                                        rootItem.height - (height / 2 + parent.y+parent.height/2):
                                                  ((-height / 2 + parent.y+ parent.height/2) < 0 ?
                                                        - (-height / 2 + parent.y+parent.height/2):0)
                visible: false
                function changeState(){
                    var lat = selectedWP.attributes.attributeValue("latitude_prev");
                    var lon = selectedWP.attributes.attributeValue("longitude_prev")
                    var asl = elevationFinder.getAltitude(
                                cInfo.homeFolder()+"/ArcGIS/Runtime/Data/elevation/"+mapHeightFolder,
                                lat,lon);
                    waypointEditor.changeASL(asl);
                    var waypointCoordinate = QtPositioning.coordinate(
                                selectedWP.attributes.attributeValue("latitude_prev"),
                                selectedWP.attributes.attributeValue("longitude_prev"),
                                selectedWP.attributes.attributeValue("altitude_prev"));
                    waypointEditor.loadInfo(waypointCoordinate,
                                            selectedWP.attributes.attributeValue("command_prev"),
                                            selectedWP.attributes.attributeValue("param1_prev"),
                                            selectedWP.attributes.attributeValue("param2_prev"),
                                            selectedWP.attributes.attributeValue("param3_prev"),
                                            selectedWP.attributes.attributeValue("param4_prev"));
                    waypointModeEnabled = (selectedWP.attributes.attributeValue("id") !== 0);
                }

                onConfirmClicked: {
                    rootItem.confirmWP(QtPositioning.coordinate(latitude,longitude,agl),
                                       param1,param2,param3,param4);
                    waypointEditor.visible = false;

                }
                onCancelClicked: {
                    rootItem.restoreWP();
                    waypointEditor.visible = false;
                }
                onWaypointModeChanged: {
                    if(selectedWP !== undefined){
                        console.log("waypointMode = "+waypointMode);
                        if(selectedWP.attributes.attributeValue("command") === lstWaypointCommand[vehicleType]["DO_JUMP"]["COMMAND"] &&
                           waypointMode !== "DO_JUMP"){
                            var wpDoJumpLatLon = Conv.mercatorToLatLon(
                                        selectedWP.geometry.extent.center.x,
                                        selectedWP.geometry.extent.center.y)
                            changeCoordinate(QtPositioning.coordinate(
                                                 wpDoJumpLatLon['lat'],
                                                 wpDoJumpLatLon['lon'],
                                                 selectedWP.attributes.attributeValue("altitude")));
                        }
                        changeWPCommand(lstWaypointCommand[vehicleType][waypointMode]["COMMAND"],param1,param2,param3,param4);
                    }

                }
            }
        }

        Label {
            id: lblMapScale
            text: qsTr("Scale:")
            verticalAlignment: Text.AlignVCenter
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 30
            anchors.right: parent.right
            anchors.rightMargin: 99
            color: UIConstants.textColor
            font.family: UIConstants.appFont
            font.pixelSize: UIConstants.fontSize
        }

        Label {
            id: lblMapScaleValue
            text: qsTr("1:1")
            anchors.left: lblMapScale.right
            anchors.leftMargin: 10
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 30
            color: UIConstants.textColor
        }
    }
    Component.onCompleted: {
        setMapFocus(true);
    }
}

/*##^## Designer {
    D{i:25;anchors_x:1247}
}
 ##^##*/
