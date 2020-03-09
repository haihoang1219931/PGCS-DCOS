import QtQuick                  2.11
import QtQuick.Controls         2.4
import QtQuick.Layouts          1.3
import QtPositioning 5.2
import CustomViews.Components   1.0
import CustomViews.Bars         1.0

import CustomViews.UIConstants  1.0
import CustomViews.Pages        1.0
import CustomViews.Configs      1.0
import CustomViews.Advanced     1.0
import CustomViews.Dialogs      1.0
// Flight Controller & Payload Controller
import io.qdt.dev               1.0
import UC 1.0
ApplicationWindow {
    id: mainWindow
    visible: true
    width: 1920
    height: 1080
    title: qsTr("DCOS - PGCSv0.1")
    function switchVideoMap(onResize){
        if(onResize){
            if(videoPane.z < mapPane.z){
                videoPane.x = rectMap.x;
                videoPane.y = rectMap.y;
                videoPane.width = rectMap.width;
                videoPane.height = rectMap.height;
                mapPane.x = paneControl.x;
                mapPane.y = paneControl.y;
                mapPane.width = paneControl.width;
                mapPane.height = paneControl.height;
            }else{
                videoPane.x = paneControl.x;
                videoPane.y = paneControl.y;
                videoPane.width = paneControl.width;
                videoPane.height = paneControl.height;
                mapPane.x = rectMap.x;
                mapPane.y = rectMap.y;
                mapPane.width = rectMap.width;
                mapPane.height = rectMap.height;
            }
        }else{
            if(videoPane.z > mapPane.z){
                videoPane.x = rectMap.x;
                videoPane.y = rectMap.y;
                videoPane.width = rectMap.width;
                videoPane.height = rectMap.height;
                videoPane.z = 1;
                mapPane.x = paneControl.x;
                mapPane.y = paneControl.y;
                mapPane.width = paneControl.width;
                mapPane.height = paneControl.height;
                mapPane.z = 2;
                UIConstants.layoutMaxPane = UIConstants.layoutMaxPaneVideo;
            }else{
                videoPane.x = paneControl.x;
                videoPane.y = paneControl.y;
                videoPane.width = paneControl.width;
                videoPane.height = paneControl.height;
                videoPane.z = 2;
                mapPane.x = rectMap.x;
                mapPane.y = rectMap.y;
                mapPane.width = rectMap.width;
                mapPane.height = rectMap.height;
                mapPane.z = 1;
                UIConstants.layoutMaxPane = UIConstants.layoutMaxPaneMap;
            }
        }
    }
    function updatePanelsSize(){
        if(UIConstants.layoutMaxPane === UIConstants.layoutMaxPaneVideo){
            mapPane.x =  paneControl.x
            mapPane.y = paneControl.y
            mapPane.width = paneControl.width;
            mapPane.height = paneControl.height;
        }else if(UIConstants.layoutMaxPane === UIConstants.layoutMaxPaneMap){
            videoPane.x =  paneControl.x
            videoPane.y = paneControl.y
            videoPane.width = paneControl.width;
            videoPane.height = paneControl.height;
        }
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //                  Components
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    GimbalNetwork{
        id: gimbalNetwork
    }

    Joystick{
        id: joystick
    }

    Computer{
        id: computer
    }
    IOFlightController{
        id: comTracker
    }

    Vehicle{
        id: tracker
        onCoordinateChanged: {
            mapPane.updateTracker(tracker.coordinate);
        }
        onHeadingChanged: {
            mapPane.updateHeadingTracker(tracker.heading);
        }
    }
    IOFlightController{
        id: comTest
    }

    Vehicle{
        id: vehicle
//        communication: comTest
        onCoordinateChanged: {
            mapPane.updatePlane(position);
        }
        onHeadingChanged: {
            mapPane.updateHeadingPlane(vehicle.heading);
        }

        onVehicleTypeChanged: {
            mapPane.changeVehicleType(vehicle.vehicleType);
        }

        onHomePositionChanged:{
            mapPane.changeHomePosition(vehicle.homePosition);
            if(!timerSendHome.running){
                timerSendHome.start();
            }
        }
    }
    Timer{
        id: timerSendHome
        interval: 1000
        repeat: true
        running: false
        onTriggered: {
//            console.log("Send home postion to tracker");
            tracker.sendHomePosition(vehicle.homePosition);
        }
    }

    PlanController{
        id: planController
        vehicle: vehicle
        onRequestMissionDone: {
            if(valid){
                mapPane.clearWPs();
                console.log("Mission count received "+planController.missionItems.length);
                for(var i=0; i< planController.missionItems.length; i++){
                    var missionItem = planController.missionItems[i];
                    console.log("missionItem["+missionItem.sequence+"]"+missionItem.frame+":["+missionItem.command+"]"
                                +missionItem.param1+":"+missionItem.param2+":"+missionItem.param3+":"+missionItem.param4+":"
                                +missionItem.param5+":"+missionItem.param6+":"+missionItem.param7
                                );
                    mapPane.addWPPosition(missionItem.sequence,missionItem.position,missionItem.command,
                                          missionItem.param1,missionItem.param2,missionItem.param3,missionItem.param4);

                }
                toastFlightControler.callActionAppearance = false;
                toastFlightControler.rejectButtonAppearance = false;
                toastFlightControler.toastContent = "Plan request success";
                toastFlightControler.show();
                mapPane.isMapSync = true;
            }else{
                console.log("Mission request failed\r\n");
                toastFlightControler.callActionAppearance = false;
                toastFlightControler.rejectButtonAppearance = false;
                toastFlightControler.toastContent = "Plan request failed";
                toastFlightControler.show();
            }
        }
        onUploadMissionDone: {
            if(!valid){
                mapPane.clearWPs();
                console.log("Mission count received "+planController.missionItems.length);
                for(var i=0; i< planController.missionItems.length; i++){
                    var missionItem = planController.missionItems[i];
                    console.log("missionItem["+missionItem.sequence+"]"+missionItem.frame+":["+missionItem.command+"]"
                                +missionItem.param1+":"+missionItem.param2+":"+missionItem.param3+":"+missionItem.param4+":"
                                +missionItem.param5+":"+missionItem.param6+":"+missionItem.param7
                                );
//                    if( missionItem.command!== 22){
//                        mapPane.addWPPosition(missionItem.sequence,missionItem.position,missionItem.command,
//                                              missionItem.param1,missionItem.param2,missionItem.param3,missionItem.param4);
//                    }else{
//                        mapPane.isContainWP1 = false;
//                    }
                    mapPane.addWPPosition(missionItem.sequence,missionItem.position,missionItem.command,
                                          missionItem.param1,missionItem.param2,missionItem.param3,missionItem.param4);
                }
            }else{
                console.log("onUploadMissionDone\r\n");
//                planController.missionItems = mapPane.getCurrentListWaypoint();
                toastFlightControler.callActionAppearance = false;
                toastFlightControler.rejectButtonAppearance = false;
                toastFlightControler.toastContent = "Plan upload success";
                toastFlightControler.show();
                var listCurrentWaypoint = mapPane.getCurrentListWaypoint();
                planController.missionItems = listCurrentWaypoint;
                mapPane.isMapSync = true;
            }
        }
    }
    MissionController{
        id: missionController
        vehicle: vehicle
        onCurrentIndexChanged: {
//            console.log("changeCurrentWP to "+sequence);
            mapPane.changeCurrentWP(sequence);
        }
    }

    //------------ Toastr notification
    Toast {
        id: toast
        anchors { bottom: parent.bottom; right: parent.right }
    }


    //------------ UC EventListener signals
    Connections {
        target: UC_API?UCEventListener:undefined
        onInvalidOpenPcdVideoFired: {
            switch( invalidCase ) {
                case UCEventEnums.PCD_VIDEO_DUPLICATE:
                    toast.callActionAppearance = false;
                    toast.rejectButtonAppearance = false;
                    toast.toastContent = "Chỉ mở được video của 1 PCD tại 1 thời điểm.! \nĐóng PCD video trước khi mở video của PCD khác";
                    toast.show();
                    break;
                case UCEventEnums.USER_NOT_IN_ROOM:
                    toast.callActionAppearance = false;
                    toast.rejectButtonAppearance = false;
                    toast.toastContent = "Ko hợp lệ ! \n Chỉ xem được hình ảnh của PCD ở trong phòng !";
                    toast.show();
                    break;
                default:
                    console.log(" Invalid case fired.!!!");
                    break;
            }
        }
    }
    Connections{
        target: UC_API?UCDataModel:undefined
        onSelectedIDChanged:{
            console.log("onSelectedIDChanged ["+selectedID+"]"+
                        UCDataModel.listUsers[selectedID].latitude+","+
                        UCDataModel.listUsers[selectedID].longitude);
            mapPane.focusOnPosition(UCDataModel.listUsers[selectedID].latitude,
                                    UCDataModel.listUsers[selectedID].longitude);
        }
    }

    Timer {
        id: timer
    }

    //------------ UC Signals
    Connections {
        target: UC_API?UcApi:undefined

        //--- Signal inform that client socket connected to server
        onConnectedToServer: {
            toast.toastContent = "UC module connected to server";
            toast.callActionAppearance =  false;
            toast.rejectButtonAppearance = false;
            toast.show();
        }

        //--- Signal notify that just received list rooms
        onReceivedListRoomsSignal: {
            console.log("\nHanlde signal ReceivedListRoomsSignal:");
            console.log("--------------------------------------> Data: " + listRoomMsg);
            UCDataModel.cleanRoom();
            JSON.parse(listRoomMsg).forEach(function(eachRoom, i) {
                var room = {};
                room.roomName =  eachRoom.roomName;
                room.onlineUsers = eachRoom.participants;
                console.log("New room added to UCDataModel: " + JSON.stringify(eachRoom));
                UCDataModel.addRoom(room);
            });
        }

        //--- Signal notify that just received list active users
        onReceivedListActiveUsers: {
            // UCDataModel.clean();

            //- Data type:
            //          @listUsers : [ {ipAddress: string, user_id: string,
            //                          room_name: string, available: bool,
            //                          role: string, shared: bool, uid: string,
            //                          name: string},
            //                         {...},
            //                         {...}]

            console.log("\nHanlde signal receivedListActiveUsers:");
            console.log("--------------------------------------> Data: " + listUsers);

            JSON.parse(listUsers).forEach(function(eachUser, i) {
                if(!UCDataModel.isUserExist(eachUser.name)){
                    var user = {};
                    user.ipAddress = eachUser.ipAddress;
                    user.userId = eachUser.user_id;
                    user.roomName =  eachUser.room_name;
                    user.available = eachUser.available;
                    user.role =  eachUser.role;
                    user.shared = eachUser.shared;
                    user.uid = eachUser.uid;
                    user.name = eachUser.name;
                    user.connectionState = eachUser.connection;
                    user.latitude = eachUser.lat;
                    user.longitude = eachUser.lng;
                    UCDataModel.addUser(user);
                }
            });
            console.log("--------------------------------------> Done handle received list users\n");
        }

        //--- Signal notify that there are new user online
        onNewUserOnline: {
            console.log("\nHanlde signal onNewUserOnline:");
            console.log("--------------------------------------> Data: {newUserName: " + newUserName + ", newUserId: " + newUserUid);
            toast.toastContent = "Người dùng " + newUserName + " online. !";
            toast.actionType = "add_to_room";
            toast.injectData =  JSON.parse(newUserUid);
            toast.callActionAppearance =  true;
            toast.rejectButtonAppearance = false;
            toast.show();
            console.log("--------------------------------------> Done handle new user online event\n");
        }

        //--- Signal to notify that there are one online user want to join room
        onPcdRequestJoinRoom: {
            console.log("\nHanlde signal pcdRequestJoinRoom: \n");
            console.log("--------------------------------------> Data: pcdUid = " + pcdUid);
            toast.toastContent  = "PCD có số hiệu " + pcdUid + " muốn tham gia phòng. !" ;
            toast.actionType = "request_join_room";
            toast.injectData = JSON.parse(pcdUid);
            toast.pauseAnimationTime = 5000;
            toast.callActionAppearance = true;
            toast.rejectButtonAppearance = true;
            toast.show();
            console.log("--------------------------------------> Done handle pcd request join room event\n");
        }

        //--- Signal to notify that there are a pcd want to share his video
        onPcdRequestShareVideo: {
            console.log("\nHanlde signal onPcdRequestShareVideo:");
            console.log("--------------------------------------> Data: pcdUid = " + pcdUid);
            for (var i = 0; i < userOnMap.listUsersOnMap.count; i++) {
                if( userOnMap.listUsersOnMap.itemAt(i).uid_ == JSON.parse(pcdUid) ) {
                    userOnMap.listUsersOnMap.itemAt(i).spreadColor = "#fc5c65";
                    userOnMap.listUsersOnMap.itemAt(i).iconFlag = "\uf256";
                    if( userOnMap.listUsersOnMap.itemAt(i).active !== true ) {
                        userOnMap.listUsersOnMap.itemAt(i).active = true;
                    }
                }
            }
            console.log("--------------------------------------> Done handle pcd request share video\n");
        }

        //--- Signal to inform the user acceptance to request add to room
        onPcdReplyRequestAddToRoom: {
            console.log("\nHanlde signal pcdReplyRequestAddToRoom:");
            console.log("--------------------------------------> Data: pcdUid - "  + pcdUid + ", room - " + room + ", accepted - " + accepted ? "true" : "false");
            if( accepted ) {
                toast.callActionAppearance = false;
                toast.rejectButtonAppearance = false;
                toast.toastContent = "PCD số hiệu " + pcdUid + " đã vào phòng. !";

                //UCDataModel.updateUser(JSON.parse(pcdUid), UserAttribute.ROOM_NAME, room);
            } else {
                toast.toastContent = "PCD số hiệu " + pcdUid + " đã từ chối. !";
            }
            toast.show();
            console.log("--------------------------------------> Done handle pcd reply request add to room\n");
        }

        //--- Signal to notify that room just have something changed
        onUpdateRoom: {
            console.log("\nHanlde signal update room:");
            console.log("--------------------------------------> Data: " + dataObjectStr);

            var dataObject =  JSON.parse(dataObjectStr);

            if( dataObject.action === "join") {
                UCDataModel.updateUser(dataObject.participant.uid, UserAttribute.ROOM_NAME, dataObject.participant.room_name);
                UCDataModel.updateUser(dataObject.participant.uid, UserAttribute.AVAILABLE, dataObject.participant.available);
                UCDataModel.newUserJoinedRoom(UcApi.getRoomName());
            }else if( dataObject.action ===  "leave" ) {
                UCDataModel.updateUser(dataObject.participant.uid, UserAttribute.ROOM_NAME, "");
                UCDataModel.updateUser(dataObject.participant.uid, UserAttribute.AVAILABLE, dataObject.participant.available);
                toast.callActionAppearance =  false;
                toast.rejectButtonAppearance = false;
                toast.toastContent = "PCD" + dataObject.participant.name + " đã rời phòng. !";
                toast.show();
                UCDataModel.userLeftRoom(UcApi.getRoomName());
            }

            console.log("--------------------------------------> Done handle update room signal\n");
        }

        //--- Signal to notify that there is one user change his available state
        onSpecUserChangeAvailable: {
            console.log("\nHanlde signal specific user change available:");
            console.log("--------------------------------------> Data: " + userAttrObj);
            UCDataModel.updateUser(JSON.parse(userAttrObj).uid, UserAttribute.AVAILABLE, JSON.parse(userAttrObj).available);
            console.log("--------------------------------------> Done handle user change available signal\n");
//            mapPane.updateUserOnMap();
        }

        //--- Signal to notify that there is one user just change hist role
        onUserUpdateRole: {
            console.log("\nHanlde signal specific user update role: ");
            console.log("--------------------------------------> Data: userUip - " + userUid + ", role - " + role);
            UCDataModel.updateUser(JSON.parse(userUid), UserAttribute.ROLE, JSON.parse(role));
            console.log("--------------------------------------> Done handle user update role\n");

        }

        //--- Signal to notify that there is one user just close the connection
        onUserInactive: {
            console.log("\nHanlde signal specific user inactive:");
            console.log("--------------------------------------> Data: userUid - " + userUid);
            UCDataModel.removeUser(JSON.parse(userUid));
            console.log("--------------------------------------> Done handle spec user inactive\n");
        }


        //--- Signal to inform about state of sharing pcd video
        onPcdSharingVideoStatus: {
            console.log("\nHanlde signal received specific user video sharing state: ");
            console.log("--------------------------------------> Data: status - " + status ? "true" : "false" + ", pcdUid - " + pcdUid);
            if( status == true)  {
                //- Show notification
                toast.toastContent = "Video của PCD có số hiệu " + pcdUid + " đã được chia sẻ đến các thành viên trong phòng !";
                toast.show();

                //- Update view
                userOnMap.pcdVideoSharing = true;
            }else {
                //- Show notification
                toast.toastContent = "Đã ngắt chia sẻ video của PCD có số hiệu " + pcdUid + " đến các thành viên trong phòng!";
                toast.show();

                //- Update view
                userOnMap.pcdVideoSharing = false;
            }
            UCDataModel.updateUser(JSON.parse(pcdUid), UserAttribute.SHARED, status);
            console.log("--------------------------------------> Done handle spec user video sharing state \n");
        }

        //--- Signal to notify that you should reload web engine page view
        onShouldReloadWebEngineView: {
            console.log("\nHanlde signal notify that should reload web engine view: ");
            console.log("--------------------------------------> Data: (redo_action: " + redoAction + ", redo_data: " + redoData + ")");
            userOnMap.reloadPcdVideo();
            timer.interval = 1000;
            timer.repeat = true;
            timer.running = true;
            timer.triggered.connect(function () {
                var redoData_ = JSON.parse(redoData);
                if( userOnMap.pcdVideoView.pcdVideo.loading === false ) {
                    switch( redoAction ) {
                        case RedoActionAfterReloadWebView.OPEN_SINGLE_PCD_VIDEO:
                            UcApi.requestOpenParticularPcdVideo(redoData_.pcdUid);
                            break;
                        case RedoActionAfterReloadWebView.ADD_PCD_TO_ROOM:
                            UcApi.addPcdToRoom(redoData_.pcd_uid);
                            break;
                    }
                    timer.running = false;
                    //-- update web engine view position follow along with x, y of pcd
                    //                    listActiveUsers.forEach(function(eachUser) {
                    //                        if( eachUser.ipAddress === JSON.parse(pcdSessionIpAddress) ) {
                    //                            pcdVideoView.x = eachUser.x_ + 40;
                    //                            pcdVideoView.y = eachUser.y_;
                    //                        }
                    //                    })
                }
            })
            console.log("--------------------------------------> Done handle reload web engine view\n");
        }

        //--- Signal to notify that you should update user location on map
        onSpecUserLocationUpdated: {
            console.log("\nHanlde signal notify that there is user updated his location: ");
            console.log("--------------------------------------> Data: (usreUid: " + userUid +
                        ", lat["+UserAttribute.LATITUDE+"]: " + lat +
                        ", lng["+UserAttribute.LONGITUDE+"]: " + lng + " ) ");
            UCDataModel.updateUser(userUid, UserAttribute.LATITUDE, lat);
            UCDataModel.updateUser(userUid, UserAttribute.LONGITUDE, lng);
            console.log("--------------------------------------> Done handle update user location\n");
        }

        //--- Signal to notify that you should change user connection status icon at sidebar
        onUserUpdateConnectionState: {
            console.log("\nHanlde signal notify that there is user updated his connection status: ");
            console.log("--------------------------------------> Data: (usreUid: " + userUid + ", connectionState: " + connectionState + " ) ");
            UCDataModel.updateUser(userUid, UserAttribute.CONNECTION_STATE, connectionState);
            console.log("--------------------------------------> Done handle update user connection\n");
        }
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //                  App Skeleten
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //------------ Header
    NavBar{
        id: navbar
        anchors {top: parent.top; left: parent.left; right: parent.right}
        height: UIConstants.sRect*2
        z: 8
        onItemNavChoosed:{
            console.log("seq clicked = "+seq);
            if(seq>=0 && seq<=2){
                footerBar.footerBarCurrent = seq;
                if(seq === 0){
                    UIConstants.monitorMode = UIConstants.monitorModeMission;
                    stkMainRect.currentIndex = 1;
                }else if(seq === 2){
                    UIConstants.monitorMode = UIConstants.monitorModeFlight;
                    stkMainRect.currentIndex = 1;
                }else {
                    stkMainRect.currentIndex = 0;
                }

            }else if(seq === 3){
                if(!navbar._isPayloadActive){
                    rightBar.hideRightBar();
                }else{
                    if(rightBar.currentIndex === 2){
                        rightBar.currentIndex = 0;
                    }
                }
            }else if(seq === 4){
                stkMainRect.currentIndex = 2;
            }
        }
        onDoShowParamsSelect:{
            if(show){
                var compo = Qt.createComponent("qrc:/CustomViews/Dialogs/ParamsSelectDialog.qml");
                var confirmDialogObj = compo.createObject(parent,{
                    "title":"Select minor params to show",
                    "type": "CONFIRM",
                    "vehicle": vehicle,
                    "x":parent.width / 2 - UIConstants.sRect * 50 / 2,
                    "y":parent.height / 2 - UIConstants.sRect * 24 / 2,
                    "z":200});
                confirmDialogObj.clicked.connect(function (type,func){
                    navbar.dialogShow = "";
                    confirmDialogObj.destroy();
                    compo.destroy();
                });
            }
        }
        onDoSwitchPlaneMode: {
            if(!footerBar.isShowConfirm){
                footerBar.isShowConfirm = true;
                footerBar.compo = Qt.createComponent("qrc:/CustomViews/Dialogs/ConfirmDialog.qml");
                footerBar.confirmDialogObj = footerBar.compo.createObject(parent,{
                    "title":"Are you sure to change next flight mode to:\n"+currentMode,
                    "type": "CONFIRM",
                    "x":parent.width / 2 - UIConstants.sRect * 13 / 2,
                    "y":parent.height / 2 - UIConstants.sRect * 6 / 2,
                    "z":200});
                footerBar.confirmDialogObj.clicked.connect(function (type,func){
                    if(func === "DIALOG_OK"){
                        setFlightMode(currentMode);
                        vehicle.flightMode = currentMode;
                    }else if(func === "DIALOG_CANCEL"){
                        setFlightMode(previousMode);
                    }
                    footerBar.isShowConfirm = false;
                    dialogShow = "";
                    mapPane.setFocus(true);
                    footerBar.confirmDialogObj.setFocus(false);
                    footerBar.confirmDialogObj.destroy();
                    footerBar.compo.destroy();
                });
            }
        }

        Component.onCompleted: {
            navbar.setFlightModes(vehicle.flightModes)
        }
    }

    //------------ Body
    StackLayout{
        id: stkMainRect
        anchors {bottom: footerBar.top; left: parent.left; top: navbar.bottom; right: parent.right }
        currentIndex: 1
        z: 7
        Item {
            id: rectPreFlightCheck
            Layout.alignment: Qt.AlignCenter
            PreflightCheck{
                id: preflightCheck
                anchors.fill: parent

            }
        }
        Item {
            id: rectFlight
            Layout.alignment: Qt.AlignCenter
            RightBar{
                id: rightBar
                width: UIConstants.sRect*3
                anchors {
                    bottom: parent.bottom;
                    top: parent.top ;
                    topMargin: 30
                    right: parent.right;
                    rightMargin: navbar._isPayloadActive?0:-width;}
                z: 6
                visible: UIConstants.monitorMode === UIConstants.monitorModeFlight
            }
            ParamsShowDialog{
                id: paramsShowPanel
                title: "Minor params"
                z:6
                vehicle: vehicle
                anchors {
                    bottom: parent.bottom;
                    bottomMargin: 5;
                    right: rightBar.left;
                    rightMargin: UIConstants.sRect
                }
            }
            VideoPane{
                id: videoPane
                visible: false
                width: paneControl.width
                height: paneControl.height
                x: paneControl.x
                y: paneControl.y
                z: 2
            }
            PaneControl{
                id: paneControl
                visible: false
                anchors {bottom: parent.bottom; bottomMargin: UIConstants.sRect; left: parent.left; leftMargin: UIConstants.sRect}
                z: 5
//                visible: UIConstants.monitorMode === UIConstants.monitorModeFlight
                layoutMax: UIConstants.layoutMaxPane
                onSwitchClicked: {
                    mainWindow.switchVideoMap(false)
                }
                onMinimizeClicked: {
                    if(videoPane.z > mapPane.z){
                        videoPane.visible = state === "show";
                    }else{
                        mapPane.visible = state === "show";
                    }
                }
                onFocusAll: {
                    mapPane.focusAllObject();
                }
                onZoomIn: {
                    mapPane.zoomIn();
                }
                onZoomOut: {
                    mapPane.zoomOut();
                }
                onSensorClicked: {
                    console.log("Change sensor ID from ["+camState.sensorID+"]");
                    if(camState.sensorID === camState.sensorIDEO){
                        camState.sensorID = camState.sensorIDIR;
                    }else{
                        camState.sensorID = camState.sensorIDEO;
                    }
                    console.log("Change sensor ID to ["+camState.sensorID+"]");
                    if(VIDEO_DECODER){
                        if(camState.sensorID === camState.sensorIDEO){
                            videoPane.player.setVideo(
                                        "rtspsrc location=rtsp://192.168.0.103/z3-1.sdp latency=150 ! rtph265depay ! h265parse ! avdec_h265 ! "+
                                        "appsink name=mysink sync=false async=false");
                            videoPane.player.start()
                        }else{
                            videoPane.player.setVideo(
                                        "rtspsrc location=rtsp://192.168.0.103/z3-2.sdp latency=150 ! rtph265depay ! h265parse ! avdec_h265 ! "+
                                        "appsink name=mysink sync=false async=false");
                            videoPane.player.start()
                        }
                    }
//                    if(CAMERA_CONTROL){
//                        if(camState.isConnected && camState.isPingOk && gimbalNetwork.isGimbalConnected){
//                            gimbalNetwork.ipcCommands.changeSensorID(camState.sensorID);
//                        }
//                    }
                }
                onGcsSnapshotClicked: {
                    if(VIDEO_DECODER){
                        videoPane.player.capture();
                    }
                }
                onGcsStabClicked: {
                    camState.gcsStab =! camState.gcsStab;
                    if(VIDEO_DECODER){
                        videoPane.player.setStab(camState.gcsStab)
                    }
                }

                onGcsRecordClicked: {
                    camState.gcsRecord=!camState.gcsRecord;
//                    console.log("setVideoSavingState to "+camState.gcsRecord)
                    if(VIDEO_DECODER){
                        videoPane.player.setRecord(camState.gcsRecord);
                    }
                }
                onGcsShareClicked: {
                    camState.gcsShare=!camState.gcsShare;
//                    console.log("setVideoSavingState to "+camState.gcsRecord)
                    if(VIDEO_DECODER){
                        videoPane.player.setShare(camState.gcsShare);
                    }
                }

                onWidthChanged: {
                    mainWindow.updatePanelsSize();
                }
                Component.onCompleted: {
                    mainWindow.updatePanelsSize();
                }
            }

            MapPane{
                id: mapPane
                width: rectMap.width
                height: rectMap.height
                x: rectMap.x
                y: rectMap.y
                z: 1
                function updateUsersOnMap(){
                    if(UC_API){
                        for (var id = 0; id < UCDataModel.listUsers.length; id++){
                            var user = UCDataModel.listUsers[id];
                            var pointMapOnScreen =
                                    mapPane.convertLocationToScreen(user.latitude,user.longitude);
                            console.log("updateUserOnMap["+id+"] from ["+user.latitude+","+user.longitude+"] to ["
                                        +pointMapOnScreen.x+","+pointMapOnScreen.y+"]" );
                            userOnMap.updateUCClientPosition(id,pointMapOnScreen.x,pointMapOnScreen.y);
                        }
                        userOnMap.updatePcdVideo(userOnMap.currentPcdId);
                    }else{
                        return;
                    }
                }

                onMapClicked: {
                    footerBar.flightView = !isMap?"WP":"MAP";
                    if(!isMap)
                    {
                        var listWaypoint = mapPane.getCurrentListWaypoint();
                        for(var i=0; i< listWaypoint.length; i++){
                            var missionItem = listWaypoint[i];
                            if( missionItem.sequence === selectedIndex){
                                footerBar.loiterSelected = (missionItem.command === 19);
                                break;
                            }else{

                            }
                        }
                    }else{
                        if(footerBar.addWPSelected){
                            mapPane.addWP(mapPane.lastWPIndex()+1);
                            // update mission items
                            var listCurrentWaypoint = mapPane.getCurrentListWaypoint();
                            planController.writeMissionItems = listCurrentWaypoint;
                        }
                    }
                }
                onMapMoved: {
                    updateUsersOnMap()
                }
                onHomePositionChanged: {
                    console.log("Home change to "+lat+","+lon);
                    vehicle.setHomeLocation(lat,lon);
                    vehicle.setAltitudeRTL(alt);
                }
                Connections{
                    target: vehicle
                    onFlightModeChanged:{
                        if(vehicle.flightMode !== "Guided"){
                            mapPane.changeClickedPosition(mapPane.clickedLocation,false);
                        }
                    }
                }
                UserOnMap {
                    id: userOnMap
                    anchors.fill: parent
                }
            }
            Rectangle{
                id: rectMap
                color: "transparent"
                border.color: UIConstants.grayColor
                border.width: 1
                radius: UIConstants.rectRadius
                anchors {bottom: parent.bottom; left: parent.left; top: parent.top; right: parent.right }
                z: 5
                visible: UIConstants.monitorMode === UIConstants.monitorModeFlight
            }

            HUD{
                id: hud
                anchors {right: rightBar.left; top: parent.top;topMargin: UIConstants.sRect * 2; rightMargin: UIConstants.sRect }
                z: 5
                visible: UIConstants.monitorMode === UIConstants.monitorModeFlight
            }
            StackLayout{
                id: popUpInfo
                clip: true
                visible: UIConstants.monitorMode === UIConstants.monitorModeFlight && UC_API
                width: UIConstants.sRect * 12
                height: UIConstants.sRect * 18
                anchors.top: hud.bottom
                anchors.topMargin: UIConstants.sRect/2
                anchors.right: rightBar.left
                anchors.rightMargin: UIConstants.sRect
                property bool show: true
                currentIndex: 0
                z: 5
                Page{
                    background: Rectangle{
                        color: UIConstants.transparentBlue
                    }
                    PopupInfo{
                        id: ucInfo
                        anchors.fill: parent
                        onOpenChatBoxClicked:{
                            console.log("onOpenChatBoxClicked "+popUpInfo.currentIndex);
                            popUpInfo.currentIndex = 1;

                        }
                    }
                }
                Page{
                    background: Rectangle{
                        color: UIConstants.transparentBlue
                    }
                    ChatBox{
                        id: chatBox
                        anchors.fill: parent
                        onCloseClicked: {
                            console.log("onCloseClicked "+popUpInfo.currentIndex);
                            popUpInfo.currentIndex = 0;
                        }
                    }
                }
            }
            PropertyAnimation{
                id: animPopup
                target: popUpInfo
                properties: "anchors.rightMargin"
                to: popUpInfo.show ? UIConstants.sRect :-UIConstants.sRect * 12
                duration: 800
                easing.type: Easing.InOutBack
                running: false
            }
            PropertyAnimation{
                id: animBtnShowUp
                target: btnShowPopup
                properties: "iconRotate"
                to: popUpInfo.show ? 0 : 180
                duration: 800
                easing.type: Easing.InExpo
                running: false
                onStopped: {
                    animPopup.start()
                }
            }
            Canvas{
                y: 70
                width: 20
                height: 60
                z: 5
                visible: UIConstants.monitorMode === UIConstants.monitorModeFlight && popUpInfo.visible
                anchors.verticalCenter: popUpInfo.verticalCenter
                anchors.right: popUpInfo.left
                anchors.rightMargin: 0
                onPaint: {
                    var ctx = getContext("2d");
                    var drawColor = UIConstants.bgAppColor;
                    ctx.strokeStyle = drawColor;
                    ctx.fillStyle = drawColor;
                    ctx.beginPath();
                    ctx.moveTo(width,0);
                    ctx.lineTo(width,height);
                    ctx.lineTo(0,height*5/6);
                    ctx.lineTo(0,height*1/6);
                    ctx.closePath();
                    ctx.fill()
                }
                FlatButtonIcon{
                    id: btnShowPopup
                    y: 70
                    width: 20
                    height: 60
                    z: 5
                    anchors.verticalCenter: parent.verticalCenter
                    anchors.horizontalCenter: parent.horizontalCenter
                    iconSize: 20
                    icon: UIConstants.iChevronRight
                    isAutoReturn: true
                    isShowRect: false
                    isSolid: true

                    onClicked: {
                        popUpInfo.show = !popUpInfo.show;
                        animBtnShowUp.start();
                    }
                }
            }
            MapControl {
                id: mapControl
                anchors.left: parent.left
                anchors.top: parent.top
                anchors.leftMargin: UIConstants.sRect
                anchors.topMargin: UIConstants.sRect*2
                width: UIConstants.sRect*2.5
                height: UIConstants.sRect*2.5*3
                visible: UIConstants.layoutMaxPane === UIConstants.layoutMaxPaneMap
                z: 5
                onFocusAll: {
                    mapPane.focusAllObject();
                }
                onZoomIn: {
                    mapPane.zoomIn();
                }
                onZoomOut: {
                    mapPane.zoomOut();
                }
            }
        }
        Item {
            id: rectSystem
            Layout.alignment: Qt.AlignCenter
            PageConfig{
                id: pageConfig
                anchors.fill: parent
                anchors.bottomMargin: -footerBar.height
                vehicle: vehicle
            }
        }
    }

    //------------ Footer
    FooterBar{
        id: footerBar
        anchors {bottom: parent.bottom; left: parent.left; right: parent.right }
        height: UIConstants.sRect*3
        visible: stkMainRect.currentIndex != 2
        z: 100
        property bool isShowConfirm: false
        property var compo
        property var confirmDialogObj
        Toast {
            id: toastFlightControler
            height: UIConstants.sRect*2
            width: UIConstants.sRect*20
            color: UIConstants.transparentBlue
            border.width: 1
            border.color: UIConstants.grayColor
            radius: UIConstants.rectRadius
            anchors {
                bottom: parent.top;
                bottomMargin: UIConstants.rectRadius
                horizontalCenter: parent.horizontalCenter; }
        }
        onDoLoadMap: {
            console.log("mapPane.dataPath = "+mapPane.dataPath);
            if(!isShowConfirm){
                isShowConfirm = true;
                var x = mainWindow.width/2-UIConstants.sRect*10/2;
                var y = mainWindow.height/2-UIConstants.sRect*15/2;
                var compo = Qt.createComponent("qrc:/CustomViews/Dialogs/MapDialog.qml");
                var fileDialogObj = compo.createObject(parent,{"title":"Please choose a file!",
                                                            "folder":mapPane.dataPath+"tpk",
//                                                            "nameFilters":["*.tpk"],
                                                            "x": x,
                                                            "y": y,
                                                            "z": 200
                                                       });
                fileDialogObj.fileSelected.connect(function (file){
                    mapPane.setMap(file);
                });

                fileDialogObj.modeSelected.connect(function (mode){
                    if(mode === "ONLINE"){
                        mapPane.setMapOnline();
                    }
                });
                fileDialogObj.clicked.connect(function (type,func){
                    fileDialogObj.destroy();
                    compo.destroy();
                    isShowConfirm = false;
                });
            }
        }

        onDoLoadMission: {
            if(!isShowConfirm){
                isShowConfirm = true;
                var x = mainWindow.width/2-UIConstants.sRect*10/2;
                var y = mainWindow.height/2-UIConstants.sRect*15/2;
                var compo = Qt.createComponent("qrc:/CustomViews/Dialogs/FileDialog.qml");
                var fileDialogObj = compo.createObject(parent,{"title":"Please choose a map file!",
                                                            "fileMode": "FILE_OPEN",
                                                            "folder":applicationDirPath+"/"+"missions",
                                                            "nameFilters":["*.waypoints"],
                                                            "x": x,
                                                            "y": y,
                                                            "z": 200
                                                       });
                fileDialogObj.clicked.connect(function (type,func){
                    if(func === "DIALOG_OK"){
                        var path = fileDialogObj.folder + "/" + fileDialogObj.currentFile;
                        // remove prefixed "file://"
                        path= path.replace(/^(file:\/{2})|(qrc:\/{2})|(http:\/{2})/,"");
                        // unescape html codes like '%23' for '#'
                        var cleanPath = decodeURIComponent(path);
                        console.log("Load file "+cleanPath);
                        planController.readWaypointFile(cleanPath);
                        mapPane.loadMarker(cleanPath+".markers");
                    }else if (func === "DIALOG_CANCEL"){

                    }
                    fileDialogObj.destroy();
                    compo.destroy();
                    isShowConfirm = false;
                });
            }
        }

        onDoSaveMission: {
            if(!isShowConfirm){
                isShowConfirm = true;
                var x = mainWindow.width/2-UIConstants.sRect*10/2;
                var y = mainWindow.height/2-UIConstants.sRect*15/2;
                var compo = Qt.createComponent("qrc:/CustomViews/Dialogs/FileDialog.qml");
                var fileDialogObj = compo.createObject(parent,{ "title":"Please insert file's name!",
                                                                "fileMode": "FILE_SAVE",
                                                                "folder":applicationDirPath+"/"+"missions",
                                                                "nameFilters":["*.waypoints"],
                                                                "x": x,
                                                                "y": y,
                                                                "z": 200});
                fileDialogObj.clicked.connect(function (type,func){
                    if(func === "DIALOG_OK"){
                        var path = fileDialogObj.folder + "/" + fileDialogObj.currentFile;
                        // remove prefixed "file://"
                        path = path.replace(/^(file:\/{2})|(qrc:\/{2})|(http:\/{2})/,"");
                        // unescape html codes like '%23' for '#'
                        var cleanPath = decodeURIComponent(path);
                        console.log("Save file "+cleanPath);
                        if(!cleanPath.endsWith(".waypoints")){
                            cleanPath = cleanPath + ".waypoints";
                        }else{

                        }

                        // update list waypoint to plancontroller
                        var listCurrentWaypoint = mapPane.getCurrentListWaypoint();
    //                                console.log("listCurrentWaypoint = "+listCurrentWaypoint);
    //                                for(var i =0; i< listCurrentWaypoint.length; i++){
    //                                    var missionItem = listCurrentWaypoint[i];
    //                                    console.log("missionItem["+missionItem.sequence+"]"+missionItem.frame+":["+missionItem.command+"]"+
    //                                                missionItem.param1+":"+missionItem.param2+":"+missionItem.param3+":"+missionItem.param4+
    //                                                missionItem.param5+":"+missionItem.param6+":"+missionItem.param7+":");
    //                                }
                        planController.missionItems = listCurrentWaypoint;
                        planController.writeWaypointFile(cleanPath);
                        mapPane.saveMarker(cleanPath+".markers");
                    }else if(func === "DIALOG_CANCEL"){

                    }
                    fileDialogObj.destroy();
                    compo.destroy();
                    isShowConfirm = false;
                });
            }
        }

        onPreflightCheckNext: {
            preflightCheck.next();
        }
        onPreflightCheckPrev: {
            preflightCheck.prev();
        }

        onDoPreflightItemCheck: {
            preflightCheck.doCheck();
        }
        onDoFlyAction: {
            switch(actionIndex){
            case 25:
                if(navbar._isPayloadActive) {
                    if(rightBar.currentIndex != 2){
                        rightBar.currentIndex = 2;
                        console.log("rightBar.currentIndex = 2");
                    }else{
                        console.log("hide right bar");
                        navbar._isPayloadActive = false;
                        rightBar.hideRightBar();
                    }
                }else{
                    navbar._isPayloadActive = true;
                    rightBar.currentIndex = 2;
                }
                break;
            case 1:
                if(!isShowConfirm){
                    isShowConfirm = true;
                    console.log("Do Auto");
                    var compo = Qt.createComponent("qrc:/CustomViews/Dialogs/ConfirmDialog.qml");
                    var confirmDialogObj = compo.createObject(parent,{
                        "title":"Are you sure to want to \n change flight mode to AUTO",
                        "type": "CONFIRM",
                        "x":parent.width / 2 - UIConstants.sRect * 13 / 2,
                        "y":parent.height / 2 - UIConstants.sRect * 6 / 2,
                        "z":200});
                    confirmDialogObj.clicked.connect(function (type,func){
                        if(func === "DIALOG_OK"){
                            vehicle.flightMode = "Auto";
                        }else if(func === "DIALOG_CANCEL"){

                        }
                        confirmDialogObj.destroy();
                        compo.destroy();
                        footerBar.isShowConfirm = false;
                    });

                }
                break;
            case 2:
                if(!isShowConfirm){
                    isShowConfirm = true;
                    console.log("Do Guided");
                    var compo = Qt.createComponent("qrc:/CustomViews/Dialogs/ConfirmDialog.qml");
                    var confirmDialogObj = compo.createObject(parent,{
                        "title":"Are you sure to want to \n change flight mode to Guided",
                        "type": "CONFIRM",
                        "x":parent.width / 2 - UIConstants.sRect * 13 / 2,
                        "y":parent.height / 2 - UIConstants.sRect * 6 / 2,
                        "z":200});
                    confirmDialogObj.clicked.connect(function (type,func){
                        if(func === "DIALOG_OK"){
                            vehicle.flightMode = "Guided";
                        }else if(func === "DIALOG_CANCEL"){

                        }
                        confirmDialogObj.destroy();
                        compo.destroy();
                        footerBar.isShowConfirm = false;
                    });

                }
                break;
            case 3:
                if(!isShowConfirm){
                    isShowConfirm = true;
                    console.log("Do Takeoff");
                    var compo = Qt.createComponent("qrc:/CustomViews/Dialogs/ConfirmDialog.qml");
                    var confirmDialogObj = compo.createObject(parent,{
                        "title":"Are you sure to want to \n"+(!vehicle.armed?"ARM and ":"")+"TAKE OFF ?",
                        "type": "CONFIRM",
                        "x":parent.width / 2 - UIConstants.sRect * 13 / 2,
                        "y":parent.height / 2 - UIConstants.sRect * 6 / 2,
                        "z":200});
                    confirmDialogObj.clicked.connect(function (type,func){
                        if(func === "DIALOG_OK"){
                            if(!vehicle.armed)
                                vehicle.setArmed(true);
                            vehicle.commandTakeoff(100);
                            vehicle.startMission();
//                            navbar.startFlightTimer();
                        }else if(func === "DIALOG_CANCEL"){

                        }
                        confirmDialogObj.destroy();
                        compo.destroy();
                        footerBar.isShowConfirm = false;
                    });
                }
                break;

            case 4:
                if(!isShowConfirm){
                    isShowConfirm = true;
                    console.log("Do Altitude changed");
                    var minValue = 10;
                    var maxValue = 400;
                    var currentValue = 0;
                    if(vehicle.vehicleType === 2 || vehicle.vehicleType == 14){
                        minValue = 10;
                        maxValue = 400;
                    }else {
                        minValue = 150;
                        maxValue = 3000;
                    }

                    var compo = Qt.createComponent("qrc:/CustomViews/Dialogs/AltitudeEditor.qml");
                    var confirmDialogObj = compo.createObject(parent,{
                        "x":parent.width / 2 - UIConstants.sRect * 14 / 2,
                        "y":parent.height / 2 - UIConstants.sRect * 15 / 4,
                        "z":200,
                        "minValue": minValue,
                        "maxValue": maxValue,
                        "currentValue": footerBar.getFlightAltitudeTarget()});
                    confirmDialogObj.confirmClicked.connect(function (){
                        console.log("vehicle.currentWaypoint = "+vehicle.currentWaypoint);
                        footerBar.isShowConfirm = false;
                        footerBar.setFlightAltitudeTarget(vehicle.currentWaypoint,confirmDialogObj.currentValue);
                        vehicle.commandSetAltitude(confirmDialogObj.currentValue);
                        confirmDialogObj.destroy();
                        compo.destroy();
                    });
                    confirmDialogObj.cancelClicked.connect(function (){
                        footerBar.isShowConfirm = false;
                        confirmDialogObj.destroy();
                        compo.destroy();
                    });

                }
                break;
            case 5:
                if(!isShowConfirm){
                    isShowConfirm = true;
                    console.log("Do Speed changed");
                    var minValue = 0;
                    var maxValue = 36;
                    if(vehicle.vehicleType === 2 || vehicle.vehicleType == 14){
                        minValue = 0;
                        maxValue = 36;
                    }else {
                        minValue = 85;
                        maxValue = 110;
                    }
                    var compo = Qt.createComponent("qrc:/CustomViews/Dialogs/SpeedEditor.qml");
                    var confirmDialogObj = compo.createObject(parent,{
                        "x":parent.width / 2 - UIConstants.sRect * 14 / 2,
                        "y":parent.height / 2 - UIConstants.sRect * 15 / 4,
                        "z":200,
                        "minValue": minValue,
                        "maxValue": maxValue,
                        "currentValue": Math.round(footerBar.getFlightSpeedTarget())});
                    confirmDialogObj.confirmClicked.connect(function (){
                        vehicle.commandChangeSpeed(confirmDialogObj.currentValue);
                        confirmDialogObj.destroy();
                        compo.destroy();
//                        footerBar.setFlightSpeedTarget(confirmDialogObj.currentValue);
                        footerBar.isShowConfirm = false;
                    });
                    confirmDialogObj.cancelClicked.connect(function (){

                        confirmDialogObj.destroy();
                        compo.destroy();
                        footerBar.isShowConfirm = false;
                    });

                }
                break;
            case 6:
                if(!isShowConfirm){
                    isShowConfirm = true;
                    console.log("Do Loiter Radius changed");
                    var compo = Qt.createComponent("qrc:/CustomViews/Dialogs/LoiterRadiusEditor.qml");
                    var confirmDialogObj = compo.createObject(parent,{
                        "x":parent.width / 2 - UIConstants.sRect * 14 / 2,
                        "y":parent.height / 2 - UIConstants.sRect * 15 / 4,
                        "z":200,
                        "currentValue": vehicle.paramLoiterRadius});
                    confirmDialogObj.confirmClicked.connect(function (){
                        vehicle.commandLoiterRadius(confirmDialogObj.currentValue);
                        confirmDialogObj.destroy();
                        compo.destroy();
                        footerBar.isShowConfirm = false;
                    });
                    confirmDialogObj.cancelClicked.connect(function (){

                        confirmDialogObj.destroy();
                        compo.destroy();
                        footerBar.isShowConfirm = false;
                    });

                }
                break;
            }

        }
        onDoNewMission: {
            mapPane.clearWPs();
//            mapPane.clearMarkers();
        }

        onDeleteWP: {
            mapPane.removeWP(mapPane.selectedIndex);
            // update mission items
            var listCurrentWaypoint = mapPane.getCurrentListWaypoint();
            planController.writeMissionItems = listCurrentWaypoint;
//            planController.missionItems = planController.writeMissionItems;

        }
        onDoNextWP: {
            if(mapPane.selectedIndex >= planController.missionItems.length-1){
                mapPane.selectedIndex = -1;
            }
            mapPane.selectedIndex ++;
            mapPane.focusOnWP(mapPane.selectedIndex);
        }

        onDoDownloadPlan: {
            planController.loadFromVehicle();
        }
        onDoUploadPlan: {
            var listCurrentWaypoint = mapPane.getCurrentListWaypoint();
            console.log("listCurrentWaypoint = "+listCurrentWaypoint);
            for(var i =0; i< listCurrentWaypoint.length; i++){
                var missionItem = listCurrentWaypoint[i];
                console.log("missionItem["+missionItem.sequence+"]"+missionItem.frame+":["+missionItem.command+"]"+
                            missionItem.param1+":"+missionItem.param2+":"+missionItem.param3+":"+missionItem.param4+
                            missionItem.param5+":"+missionItem.param6+":"+missionItem.param7+":");
            }
            planController.writeMissionItems = listCurrentWaypoint;
            planController.sendToVehicle();
        }

        onDoGoWP: {
            if(mapPane.selectedIndex > 0){
                if(vehicle.flightMode === "Guided"){
                    vehicle.flightMode = "Auto";
                }
                vehicle.setCurrentMissionSequence(mapPane.selectedIndex);
            }else if(mapPane.selectedIndex === 0){
                if(!isShowConfirm){
                    isShowConfirm = true;
                    console.log("Do RTL");
                    var compo = Qt.createComponent("qrc:/CustomViews/Dialogs/ConfirmDialog.qml");
                    var confirmDialogObj = compo.createObject(parent,{
                        "title":"Are you sure to want to \n change flight mode to RTL",
                        "type": "CONFIRM",
                        "x":parent.width / 2 - UIConstants.sRect * 13 / 2,
                        "y":parent.height / 2 - UIConstants.sRect * 6 / 2,
                        "z":200});
                    confirmDialogObj.clicked.connect(function (type,func){
                        if(func === "DIALOG_OK"){
                            vehicle.flightMode = "RTL";
                        }else if(func === "DIALOG_CANCEL"){

                        }
                        confirmDialogObj.destroy();
                        compo.destroy();
                        footerBar.isShowConfirm = false;
                    });
                }
            }
        }
        onDoArm: {
            if(!isShowConfirm){
                isShowConfirm = true;
                console.log("Do Arm");
                var compo = Qt.createComponent("qrc:/CustomViews/Dialogs/ConfirmDialog.qml");
                var confirmDialogObj = compo.createObject(parent,{
                    "title":"Are you sure to want to \n"+(!vehicle.armed?"ARM":"DISARM")+"?",
                    "type": "CONFIRM",
                    "x":parent.width / 2 - UIConstants.sRect * 13 / 2,
                    "y":parent.height / 2 - UIConstants.sRect * 6 / 2,
                    "z":200});
                confirmDialogObj.clicked.connect(function (type,func){
                    if(func === "DIALOG_OK"){
                        vehicle.setArmed(!vehicle.armed);
                        if(arm){
                            vehicle.setHomeLocation(vehicle.coordinate.latitude,
                                                vehicle.coordinate.longitude);
                            mapPane.moveWPWithID(0,vehicle.coordinate);
                        }
                    }else if(func === "DIALOG_CANCEL"){

                    }
                    confirmDialogObj.destroy();
                    compo.destroy();
                    footerBar.isShowConfirm = false;
                });
            }
        }
        onDoCircle:{
            console.log("closeWise = "+closeWise);
            if(mapPane.selectedIndex >= 0){
                mapPane.changeWPCommand(mapPane.lstWaypointCommand[mapPane.vehicleType]["LOITER"]["COMMAND"],
                                        3600,0,closeWise > 0 ? "1":"-1",0);
                mapPane.selectedWP.attributes.replaceAttribute("command_prev",
                            mapPane.selectedWP.attributes.attributeValue("command"));
                mapPane.selectedWP.attributes.replaceAttribute("param3_prev",
                            mapPane.selectedWP.attributes.attributeValue("param3"));
                mapPane.selectedWP.attributes.replaceAttribute("param1_prev",3600);
                mapPane.selectedWP.attributes.replaceAttribute("param1",3600);
                // update mission items
                var listCurrentWaypoint = mapPane.getCurrentListWaypoint();
                planController.writeMissionItems = listCurrentWaypoint;
                planController.sendToVehicle();
            }
        }
        onDoWaypoint: {
            if(mapPane.selectedIndex >= 0){
                mapPane.changeWPCommand(mapPane.lstWaypointCommand[mapPane.vehicleType]["WAYPOINT"]["COMMAND"],
                                        0,0,0,0);
                mapPane.selectedWP.attributes.replaceAttribute("command_prev",
                            mapPane.selectedWP.attributes.attributeValue("command"));
                mapPane.selectedWP.attributes.replaceAttribute("param1_prev",0);
                mapPane.selectedWP.attributes.replaceAttribute("param1",0);
                // update mission items
                var listCurrentWaypoint = mapPane.getCurrentListWaypoint();
                planController.writeMissionItems = listCurrentWaypoint;
                planController.sendToVehicle();
            }
        }
        onDoGoPosition:{
            if(!isShowConfirm){
                isShowConfirm = true;
                console.log("Do Go Position");
                if(vehicle.flightMode !== "Guided"){
                    var compo = Qt.createComponent("qrc:/CustomViews/Dialogs/ConfirmDialog.qml");
                    var confirmDialogObj = compo.createObject(parent,{
                        "title":"Vehicle is not in Guided mode\nDo you want to change and go?",
                        "type": "CONFIRM",
                        "x":parent.width / 2 - UIConstants.sRect * 13 / 2,
                        "y":parent.height / 2 - UIConstants.sRect * 6 / 2,
                        "z":200});
                    confirmDialogObj.clicked.connect(function (type,func){
                        if(func === "DIALOG_OK"){
                            vehicle.flightMode = "Guided";
                            mapPane.changeClickedPosition(mapPane.clickedLocation,true);
                            vehicle.commandGotoLocation(mapPane.clickedLocation);
                        }else if(func === "DIALOG_CANCEL"){

                        }
                        confirmDialogObj.destroy();
                        compo.destroy();
                        footerBar.isShowConfirm = false;
                    });
                }else{
                    var compo = Qt.createComponent("qrc:/CustomViews/Dialogs/ConfirmDialog.qml");
                    var confirmDialogObj = compo.createObject(parent,{
                        "title":"Are you sure to go to \n selected location"+"?",
                        "type": "CONFIRM",
                        "x":parent.width / 2 - UIConstants.sRect * 13 / 2,
                        "y":parent.height / 2 - UIConstants.sRect * 6 / 2,
                        "z":200});
                    confirmDialogObj.clicked.connect(function (type,func){
                        if(func === "DIALOG_OK"){
                            mapPane.changeClickedPosition(mapPane.clickedLocation,true);
                            vehicle.commandGotoLocation(mapPane.clickedLocation);
                        }else if(func === "DIALOG_CANCEL"){

                        }
                        confirmDialogObj.destroy();
                        compo.destroy();
                        footerBar.isShowConfirm = false;
                    });
                }
            }
        }
        onDoAddMarker:{
            mapPane.addMarker();
        }
        onDoDeleteMarker: {
            mapPane.removeMarker();
        }
    }

    onWidthChanged: {
        mainWindow.switchVideoMap(true)
        mapPane.updateMouseOnMap()
    }
    onHeightChanged: {
        mainWindow.switchVideoMap(true)
    }
    ListView{
        id: listPlate
        width: 100
        anchors.top: navbar.bottom
        anchors.topMargin: 40
        anchors.left: parent.left
        anchors.leftMargin: 100
        anchors.bottom: footerBar.top
        anchors.bottomMargin: 8
        visible: false
        z:8
        function add(src,plate){
            model.append({"src":"file://"+src,"plate":plate});
        }
        function clear(){
            model.clear();
        }
        function remove(index){
            model.remove(index);
        }
        model: ListModel{
        }

        spacing: 5
        delegate: Item{
            id: item1
            width: 100
            height: 100

            Image{
                id: img
                anchors.fill: parent
                source: src
            }
            Rectangle{
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.top: parent.top
                height: 20
                color: "green"
                Label{
                    id: lbl
                    verticalAlignment: Text.AlignVCenter
                    horizontalAlignment: Text.AlignHCenter
                    anchors.horizontalCenter: parent.horizontalCenter
                    text: plate
                    color: "white"
                }
            }
        }
    }
    CameraStateManager{
        id: camState
    }
    Timer{
        id: timerRequestData
        interval: 100; repeat: true;
//        running: camState.isPingOk && camState.isConnected
        running: false
        property int countGetData: 0
        onTriggered: {
//            console.log("Get gimbal data");
            var frameID = 0;
            if(VIDEO_DECODER){
//                frameID = videoPane.player.frameID;
            }

            var data = gimbalNetwork.gimbalModel.gimbal.getData(frameID);
            // === hainh added 2019-03-28
//            camState.panPos = Number(data["panPos"]);
//            camState.tiltPos = Number(data["tiltPos"]);
            if(camState.sensorID === camState.sensorIDEO){
                camState.hfov = Number(data["EO"]["HFOV"]);
            }else{
                camState.hfov = Number(data["IR"]["HFOV"]);
            }
//            camState.hfov = Number(data["EO"]["HFOV"]);
//            camState.vfov = Number(data["hfov"])/2;
            // 'pn', 'pe', 'pd', 'roll', 'pitch', 'yaw',
//            camState.latitude = Number(data["pn"]);
//            camState.longitude = Number(data["pe"]);
//            camState.altitude = Number(data["pd"]);
//            camState.roll = Number(data["roll"]);
//            camState.pitch = Number(data["pitch"]);
//            camState.yaw = Number(data["yaw"]);
            // hainh added 2019-03-28 ===
//            camState.sensorID = data["SENSOR"];
//            camState.updateTrackSize(data["TRACK_SIZE"]);
//            camState.changeLockMode(data["LOCK_MODE"]);
//            camState.gimbalMode = data["GIMBAL_MODE"];
//            camState.gimbalRecord = data["GIMBAL_RECORD"];
//            camState.gimbalStab = data["STAB_GIMBAL"];
//            camState.digitalStab = data["STAB_DIGITAL"];
//            camState.trackSize = data["TRACK_SIZE"];
            if(gimbalNetwork.isSensorConnected)
                gimbalNetwork.ipcCommands.treronGetZoomPos();
        }
    }
    Rectangle{
        id: configPane
        x: parent.width / 2 - width / 2
        y: parent.height / 2 - height / 2
        width: 650
        height:500
        visible: false
        z:8
        MouseArea{
            x: 1
            y: 1
            width: parent.width
            height: 50
            drag.target: configPane
            drag.axis: Drag.XAndYAxis
            drag.minimumX: 0
            drag.minimumY: 0
            onPressed: {
                console.log("Pressed");
            }
        }
        StackLayout{
            id: stkConfig
            anchors.fill: parent
            AdvancedConfig{
                id: advancedConfig
                anchors.fill: parent
            }
        }
    }
    Rectangle {
        id: rectCameraControl
        width: UIConstants.sRect * 13
        height: UIConstants.sRect * 6
        color: UIConstants.transparentBlue
        visible: stkMainRect.currentIndex == 1 && false
        radius: UIConstants.rectRadius
        anchors.bottom: footerBar.top
        anchors.bottomMargin: UIConstants.sRect
        anchors.horizontalCenter: parent.horizontalCenter
        property bool show: true
        z:7
        MouseArea{
            anchors.fill: parent
            hoverEnabled: true
        }
        PropertyAnimation{
            id: animCameraControl
            target: rectCameraControl
            properties: "anchors.bottomMargin"
            to: rectCameraControl.show ? UIConstants.sRect : - UIConstants.sRect * 6
            duration: 800
            easing.type: Easing.InOutBack
            running: false
        }
        PropertyAnimation{
            id: animShowCameraControl
            target: btnShowCameraControl
            properties: "iconRotate"
            to: rectCameraControl.show ? 0 : 180
            duration: 800
            easing.type: Easing.InExpo
            running: false
            onStopped: {
                animCameraControl.start()
            }
        }
        Canvas{
            width: 60
            height: 20
            anchors.horizontalCenter: rectCameraControl.horizontalCenter
            anchors.bottom: rectCameraControl.top
            anchors.bottomMargin: 0
            onPaint: {
                var ctx = getContext("2d");
                var drawColor = UIConstants.transparentBlue;
                ctx.strokeStyle = drawColor;
                ctx.fillStyle = drawColor;
                ctx.beginPath();
                ctx.moveTo(0,height);
                ctx.lineTo(width,height);
                ctx.lineTo(width*5/6,0);
                ctx.lineTo(width*1/6,0);
                ctx.closePath();
                ctx.fill();
            }
            FlatButtonIcon{
                id: btnShowCameraControl
                y: 70
                width: 20
                height: 60
                anchors.verticalCenter: parent.verticalCenter
                anchors.horizontalCenter: parent.horizontalCenter
                iconSize: 20
                icon: UIConstants.iChevronDown
                isAutoReturn: true
                isShowRect: false
                isSolid: true
                onClicked: {
                    rectCameraControl.show = !rectCameraControl.show;
                    animShowCameraControl.start();
                }
            }
        }


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
            isSolid: true
            isAutoReturn: true
            isShowRect: false
            anchors.horizontalCenter: parent.horizontalCenter
            onPressed: {
//                console.log("Pressed");
                if(gimbalNetwork.isGimbalConnected)
                    gimbalNetwork.ipcCommands.gimbalControl(0, 0, camState.invertTilt*(-1023)*camState.hfov/Math.PI*camState.alphaSpeed)
            }
            onReleased: {
//                console.log("Released");
                if(gimbalNetwork.isGimbalConnected)
                    gimbalNetwork.ipcCommands.gimbalControl(0, 0, 0)
            }
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
            onPressed: {
//                console.log("Pressed");
                if(gimbalNetwork.isGimbalConnected)
                    gimbalNetwork.ipcCommands.gimbalControl(0,0, camState.invertTilt*1023*camState.hfov/Math.PI*camState.alphaSpeed)
            }
            onReleased: {
//                console.log("Released");
                if(gimbalNetwork.isGimbalConnected)
                    gimbalNetwork.ipcCommands.gimbalControl(0, 0, 0)
            }
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
            onPressed: {
//                console.log("Pressed");
                if(gimbalNetwork.isGimbalConnected)
                    gimbalNetwork.ipcCommands.gimbalControl(0, camState.invertPan*(-1023)*camState.hfov/Math.PI*camState.alphaSpeed, 0)
            }
            onReleased: {
//                console.log("Released");
                if(gimbalNetwork.isGimbalConnected)
                    gimbalNetwork.ipcCommands.gimbalControl(0, 0, 0)
            }
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
            onPressed: {
//                console.log("Pressed");
                if(gimbalNetwork.isGimbalConnected)
                    gimbalNetwork.ipcCommands.gimbalControl(0, camState.invertPan*1023*camState.hfov/Math.PI*camState.alphaSpeed, 0)
            }
            onReleased: {
//                console.log("Released");
                if(gimbalNetwork.isGimbalConnected)
                    gimbalNetwork.ipcCommands.gimbalControl(0, 0, 0)
            }
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
            isSolid: true
            iconColor: camState.sensorID === camState.sensorIDEO?UIConstants.textColor:UIConstants.grayColor
            isEnable: camState.sensorID === camState.sensorIDEO
            anchors.verticalCenter: parent.verticalCenter
            onPressed: {
                if(gimbalNetwork.isSensorConnected)
                    gimbalNetwork.ipcCommands.treronZoomIn()
            }
            onReleased: {
                if(gimbalNetwork.isSensorConnected)
                    gimbalNetwork.ipcCommands.treronZoomStop()
            }
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
            iconColor: camState.sensorID === camState.sensorIDEO?UIConstants.textColor:UIConstants.grayColor
            isEnable: camState.sensorID === camState.sensorIDEO
            anchors.verticalCenter: parent.verticalCenter
            onPressed: {
                if(gimbalNetwork.isSensorConnected)
                    gimbalNetwork.ipcCommands.treronZoomOut()
            }
            onReleased: {
                if(gimbalNetwork.isSensorConnected)
                    gimbalNetwork.ipcCommands.treronZoomStop()
            }
        }
    }

    Timer{
        id: timerStart
        repeat: false
        interval: 2000
        onTriggered: {
            // --- Flight
            comTest.loadConfig(FCSConfig);
            comTest.connectLink();
            vehicle.communication = comTest;
            vehicle.planController = planController;

            comTracker.loadConfig(TRKConfig);
            comTracker.connectLink();
            tracker.communication = comTracker;

            tracker.uav = vehicle;
            console.log("CAMERA_CONTROL = "+CAMERA_CONTROL)
            // --- Payload
            if(CAMERA_CONTROL){
                var config = PCSConfig.getData();
                console.log("config = "+config);
                console.log("Connect to camera "+config["CAM_CONTROL_IP"]+":"+config["CAM_CONTROL_IN"]+":"+config["CAM_CONTROL_REP"])
                gimbalNetwork.newConnect(config["CAM_CONTROL_IP"],
                                         config["CAM_CONTROL_IN"],
                                         config["CAM_CONTROL_REP"]);
                timerRequestData.start();
            }
            if(UC_API)
            {
                UcApi.notifyQmlReady();
            }
        }
    }

    Component.onCompleted: {
        UIConstants.changeTheme(UIConstants.themeNormal);
        if(FCSConfig.value("Settings:MapDefault:Value:data") !== "")
            mapPane.setMap(FCSConfig.value("Settings:MapDefault:Value:data"));
        timerStart.start();

    }
}

/*##^## Designer {
    D{i:0;autoSize:true;height:480;width:640}
}
 ##^##*/
