import QtQuick                  2.11
import QtQuick.Controls         2.4
import QtQuick.Layouts          1.3
import QtQuick.Controls         1.2 as OldControl
import QtQuick.Controls.Styles 1.4
import QtQuick.Window 2.11
import QtPositioning 5.2
import CustomViews.Components   1.0
import CustomViews.Bars         1.0

import CustomViews.UIConstants  1.0
import CustomViews.Pages        1.0
import CustomViews.Configs      1.0
import CustomViews.HUD          1.0
import CustomViews.Advanced     1.0
import CustomViews.Dialogs      1.0
import CustomViews.SubComponents 1.0
// Flight Controller & Payload Controller
import io.qdt.dev               1.0
import UC 1.0

ApplicationWindow {
    id: mainWindow
    visible: true
//    visibility: ApplicationWindow.FullScreen
    width: 1366
    height: 768
    title: qsTr("DCOS - PGCSv0.1")
    flags: Qt.WindowMinMaxButtonsHint
    property int seqTab: 2
    property var itemListName:
        UIConstants.itemTextMultilanguages
    function switchMapFull(){
        Pane.x = paneControl.x;
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

    CameraController{
        id: cameraController
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
        id: comVehicle
    }

    Vehicle{
        id: vehicle
        //        communication: comVehicle
        onCoordinateChanged: {
            mapPane.updatePlane(position);
        }
        onHeadingChanged: {
            mapPane.updateHeadingPlane(vehicle.heading);
        }

        onVehicleTypeChanged: {
            mapPane.changeVehicleType(vehicle.vehicleType);
            preflightCheck.changeVehicleType(vehicle.vehicleType);
            FCSConfig.changeData("VehicleType",vehicle.vehicleType);
        }

        onHomePositionChanged:{
            mapPane.changeHomePosition(vehicle.homePosition);
            if(!timerSendHome.running){
                timerSendHome.start();
            }
        }
        //        Component.onCompleted: {
        //            console.log("vehicle----------------->:"+vehicle)
        //            cameraController.gimbal.setVehicle(vehicle);
        //        }
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
            console.log("changeCurrentWP to "+sequence);
            mapPane.changeCurrentWP(sequence);
        }
    }

    //------------ Toastr notification
    Toast {
        id: toast
        anchors { bottom: popUpInfo.bottom; right: popUpInfo.right }
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
            userOnMap.reloadPcdVideo();
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
                    user.ipAddress = eachUser.ip_address;
                    user.userId = eachUser.user_id;
                    user.roomName =  eachUser.room_name;
                    user.available = eachUser.available;
                    user.role =  eachUser.role;
                    user.shared = eachUser.shared;
                    user.uid = eachUser.uid;
                    user.name = eachUser.name;
                    user.connectionState = eachUser.connection;
                    user.latitude = 21.039140;
                    user.longitude = 105.544545;
                    UCDataModel.addUser(user);
                } else {
                    UCDataModel.updateUser(eachUser.uid, UserAttribute.IP_ADDRESS, eachUser.ip_address);
                    UCDataModel.updateUser(eachUser.uid, UserAttribute.USER_ID, eachUser.user_id);
                    UCDataModel.updateUser(eachUser.uid, UserAttribute.ROOM_NAME, eachUser.room_name);
                    UCDataModel.updateUser(eachUser.uid, UserAttribute.AVAILABLE, eachUser.available);
                    UCDataModel.updateUser(eachUser.uid, UserAttribute.SHARED, eachUser.shared);
                    UCDataModel.updateUser(eachUser.uid, UserAttribute.NAME, eachUser.name);
                    UCDataModel.updateUser(eachUser.uid, UserAttribute.CONNECTION_STATE, eachUser.connection);
                }
            });
            mapPane.updateUsersOnMap();
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
            toast.toastContent = "Warning from pcd. Open pcd video to view detail";
            toast.callActionAppearance = false;
            toast.rejectButtonAppearance = false;
            toast.show();
            for (var i = 0; i < Object.keys(UCDataModel.listUsers).length; i++) {
                if( UCDataModel.listUsers[i].uid == JSON.parse(pcdUid) ) {
                    //userOnMap.listUsersOnMap.itemAt(i).spreadColor = "#fc5c65";
                    //userOnMap.listUsersOnMap.itemAt(i).iconFlag = "\uf256";
                    mapPane.focusOnPosition(UCDataModel.listUsers[i].latitude,
                                            UCDataModel.listUsers[i].longitude);
                    // UCDataModel.updateUser(JSON.parse(pcdUid), 10, !UCDataModel.listUsers[i].isSelected);
                    UCDataModel.updateUser(JSON.parse(pcdUid), 12, !UCDataModel.listUsers[i].warning);
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
                toast.toastContent = dataObject.participant.name + " đã rời phòng. !";
                toast.show();
                UCDataModel.userLeftRoom(UcApi.getRoomName());
                // hide PCDVIDEO
                if(userOnMap.pcdVideoView.visible)
                    userOnMap.pcdVideoView.visible = false;
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

        //--- Signal to notify you that the target media connection has failed
        onMediaError: {
            console.log("\nHandle signal notify you that the target media connection has failed");
            console.log("-------------------------------------> Data: (sourceUid: " + sourceUid + ", errorType: " + errorType + ")");
            toast.toastContent = "Không thể truy cập camera của người dùng";
            toast.callActionAppearance = false;
            toast.rejectButtonAppearance = false;
            toast.show();
            userOnMap.pcdVideoView.visible = false;
            console.log("--------------------------------------> Done handle media error\n");
        }

        //--- Signal to notify you that losed connection to socket server
        onNetworkCrash: {
            console.log("\nHandle signal notify you that losed connection to socket server");
            console.log("-------------------------------------> Data: ()");
            toast.toastContent = "Mất kết nối đến server";
            toast.callActionAppearance = false;
            toast.rejectButtonAppearance = false;
            toast.show();
            for (var i = 0; i < Object.keys(UCDataModel.listUsers).length; i++) {
                UCDataModel.updateUser(UCDataModel.listUsers[i].uid, 11, false);
            }
            console.log("--------------------------------------> Done handle connection error\n");
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
            seqTab = seq;
            if(seq>=0 && seq<=2){
                footerBar.footerBarCurrent = seq;
                if(seq === 0){
                    UIConstants.monitorMode = UIConstants.monitorModeMission;
                    stkMainRect.currentIndex = 1;
                    mapPane.showWPScroll(true)
                }else if(seq === 2){
                    UIConstants.monitorMode = UIConstants.monitorModeFlight;
                    stkMainRect.currentIndex = 1;
                    mapPane.showWPScroll(false)
                }else {
                    stkMainRect.currentIndex = 0;
                }
                if(seq === 0){
                    mainWindow.switchMapFull();
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
                pageConfig.showAdvancedConfig(false);
            }
        }
        onDoShowParamsSelect:{
            if(show){
                paramsSelectPanel.visible = true;
            }
        }
        onDoSwitchPlaneMode: {
            if(!footerBar.isShowConfirm){
                footerBar.isShowConfirm = true;
                footerBar.compo = Qt.createComponent("qrc:/CustomViews/Dialogs/ConfirmDialog.qml");
                footerBar.confirmDialogObj = footerBar.compo.createObject(parent,{
                              "title":mainWindow.itemListName["DIALOG"]["CONFIRM"]["FLIGHT_MODE"]
                                [UIConstants.language[UIConstants.languageID]]+":\n"+currentMode,
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
        onDoShowJoystickConfirm: {
            if(!footerBar.isShowConfirm){
                footerBar.isShowConfirm = true;
                footerBar.compo = Qt.createComponent("qrc:/CustomViews/Dialogs/ConfirmDialog.qml");
                footerBar.confirmDialogObj = footerBar.compo.createObject(parent,{
                              "title":mainWindow.itemListName["DIALOG"]["CONFIRM"]["JOYSTICK_"+(!joystick.useJoystick?"ACTIVE":"DEACTIVE")]
                                [UIConstants.language[UIConstants.languageID]],
                              "type": "CONFIRM",
                              "x":parent.width / 2 - UIConstants.sRect * 13 / 2,
                              "y":parent.height / 2 - UIConstants.sRect * 6 / 2,
                              "z":200});
                footerBar.confirmDialogObj.clicked.connect(function (type,func){
                    if(func === "DIALOG_OK"){
                        joystick.useJoystick = !joystick.useJoystick;
                        joystick.saveConfig();
                    }else if(func === "DIALOG_CANCEL"){

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
                    topMargin: UIConstants.sRect
                    right: parent.right;
                    rightMargin: navbar._isPayloadActive?0:-width;}
                z: 6
                visible: UIConstants.monitorMode === UIConstants.monitorModeFlight
            }
            ParamsShowDialog{
                id: paramsShowPanel
                title: itemListName["PARAMS"]["SHOW"]
                       [UIConstants.language[UIConstants.languageID]]
                z:6
                vehicle: vehicle
                width: hud.width
                anchors {
                    bottom: parent.bottom;
                    bottomMargin: 5;
                    right: rightBar.left;
                    rightMargin: 8
                }
            }
            ParamsSelectDialog{
                id: paramsSelectPanel
                visible: false
                title:itemListName["PARAMS"]["SELECT"]
                      [UIConstants.language[UIConstants.languageID]]
                type: "CONFIRM"
                x:parent.width / 2 - width / 2
                y:parent.height / 2 - height / 2
                z:9
                onClicked: {
                    paramsSelectPanel.visible = false;
                    navbar.dialogShow = "";
                }
            }
            ListPlatesLog{
                id: listPlateLog
                x: parent.width/2-width/2
                y: parent.height/2-height/2
                visible: paneControl.visible && chkLog.checked && (USE_VIDEO_CPU || USE_VIDEO_GPU)
                z: 8
            }

            VideoPane{
                id: videoPane
                visible: paneControl.visible
                width: paneControl.width
                height: paneControl.height
                x: paneControl.x
                y: paneControl.y
                z: 2
                Overlay{
                    id: videoOverlay
                    anchors.fill: parent
                    Connections{
                        target: cameraController.gimbal
                        onZoomCalculatedChanged:{
                            console.log("zoom_end")
                            //                            if(viewIndex === 0)
                            {
                                videoOverlay.zoomCalculate = zoomCalculated;
                            }
                        }
                    }
                }

                ObjectsOnScreen{
                    anchors.fill: parent
//                    visible: camState.gcsTargetLocalization
                    visible: false
                    player: cameraController.videoEngine
                }
            }
            PaneControl{
                id: paneControl
                anchors {bottom: parent.bottom; bottomMargin: 8; left: parent.left; leftMargin: 8}
                z: 5
                visible: UIConstants.monitorMode === UIConstants.monitorModeFlight && CAMERA_CONTROL
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
                    if(USE_VIDEO_CPU || USE_VIDEO_GPU){
                        if(camState.sensorID === camState.sensorIDIR){
                            cameraController.gimbal.changeSensor("EO");
                        }else{
                            cameraController.gimbal.changeSensor("IR");
                        }
                    }
                }
                onGcsSnapshotClicked: {
                    if(USE_VIDEO_CPU || USE_VIDEO_GPU){
                        cameraController.gimbal.snapShot();
                    }
                }
                onGcsStabClicked: {
                    if(USE_VIDEO_CPU || USE_VIDEO_GPU){
                        cameraController.gimbal.setDigitalStab(!camState.digitalStab)
                    }
                }

                onGcsRecordClicked: {
                    if(USE_VIDEO_CPU || USE_VIDEO_GPU){
                        cameraController.gimbal.setRecord(!camState.record);
                    }
                }
                onGcsShareClicked: {
                    if(USE_VIDEO_CPU || USE_VIDEO_GPU){
                        cameraController.gimbal.setShare(!camState.gcsShare);
                    }
                }

                onWidthChanged: {
                    mainWindow.updatePanelsSize();
                }
                Component.onCompleted: {
                    mainWindow.updatePanelsSize();
                }
            }

            MapPane_UAV
            {
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
                function updateObjectsOnMap(){
                    if(videoPane.player !== undefined){
                        for (var id = 0; id < videoPane.player.listTrackObjectInfos.length; id++){
                            var object = videoPane.player.listTrackObjectInfos[id];
                            var pointMapOnScreen =
                                    mapPane.convertLocationToScreen(object.latitude,object.longitude);
                            objectsOnMap.updateObjectPosition(id,pointMapOnScreen.x,pointMapOnScreen.y);
                        }
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
                    updateObjectsOnMap();
                }

                onHomePositionChanged: {
                    console.log("Home change to "+lat+","+lon);
                    vehicle.setHomeLocation(lat,lon);
                }
                onShowAdvancedConfigChanged: {
                    pageConfig.showAdvancedConfig(true);
                }
                onTotalWPsDistanceChanged: {
                    if(vehicle){
                        vehicle.setTotalWPDistance(val);
                    }
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
                ObjectsOnMap{
                    id: objectsOnMap
                    anchors.fill: parent
                    player: videoPane.player
                    visible: camState.gcsTargetLocalization
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
                anchors {right: rightBar.left; top: parent.top;topMargin: UIConstants.sRect + 8; rightMargin: 8 }
                z: 5
                visible: UIConstants.monitorMode === UIConstants.monitorModeFlight &&
                         (vehicle.vehicleType === 2 || vehicle.vehicleType === 3)
            }
            AhrsHUD{
                id:ahrsHUD
                visible: UIConstants.monitorMode === UIConstants.monitorModeFlight &&
                         (vehicle.vehicleType === 1)
                anchors{
                    bottom: mapPane.bottom
                    bottomMargin: 8
                    left: mapPane.left
                    leftMargin: 8
                }
                z: 5
            }
            StackLayout{
                id: popUpInfo

                clip: true
                visible: UIConstants.monitorMode === UIConstants.monitorModeFlight && UC_API
                width: hud.width
                height: UIConstants.sRect * 10 * 3 / 2
                anchors.top: hud.bottom
                anchors.topMargin: 8
                anchors.right: rightBar.left
                anchors.rightMargin: 8
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
                to: popUpInfo.show ? 8 :-popUpInfo.width
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
                width: UIConstants.sRect
                height: UIConstants.sRect * 3
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
                    width: UIConstants.sRect
                    height: UIConstants.sRect * 3
                    z: 5
                    anchors.verticalCenter: parent.verticalCenter
                    anchors.horizontalCenter: parent.horizontalCenter
                    iconSize: UIConstants.fontSize * 2
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
                anchors.leftMargin: 8
                anchors.topMargin: UIConstants.sRect + 8
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
            FlatButtonText{
                id: chkLog
                visible: mapControl.visible && (USE_VIDEO_CPU || USE_VIDEO_GPU)
                width: UIConstants.sRect * 2.5
                height: UIConstants.sRect + 8
                color: !checked?UIConstants.transparentBlue:UIConstants.transparentGreen
                anchors.left: parent.left
                anchors.leftMargin: 8
                anchors.top: mapControl.bottom
                anchors.topMargin: 8
                border.color: UIConstants.grayColor
                border.width: 1
                radius: UIConstants.rectRadius
                z: 8
                text: "OCR"
                textColor: UIConstants.textColor
                property bool checked: false
                onClicked: {
                    checked = !checked;
                }
            }
            Rectangle{
                id: rectProfilePath
                z: 10
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
                    title: mainWindow.itemListName["DIALOG"]["PROFILE_PATH"]["TITTLE"]
                           [UIConstants.language[UIConstants.languageID]]
                    xName: mainWindow.itemListName["DIALOG"]["PROFILE_PATH"]["DISTANCE"]
                           [UIConstants.language[UIConstants.languageID]]
                    yName: mainWindow.itemListName["DIALOG"]["PROFILE_PATH"]["ALTITUDE"]
                           [UIConstants.language[UIConstants.languageID]]
                    fontSize: UIConstants.fontSize
                    fontFamily: UIConstants.appFont
                    anchors.fill: parent
                    anchors.margins: 4
                    folderPath: computer.homeFolder()+"/ArcGIS/Runtime/Data/elevation/"+mapPane.mapHeightFolder
                }
            }
            UavProfilePath{
                id: uavProfilePath
                z: 11
                width: UIConstants.sRect * 12
                height: mapControl.height + chkLog.height + 8
                mapHeightFolder: mapPane.mapHeightFolder
                anchors.top: parent.top
                anchors.left: mapControl.right
                anchors.leftMargin: 8
                anchors.topMargin: UIConstants.sRect + 8                
                property bool show: true
            }
            PropertyAnimation{
                id: animPopupUAVProfilePath
                target: uavProfilePath
                properties: "anchors.topMargin"
                to: uavProfilePath.show ? UIConstants.sRect + 8:
                                          UIConstants.sRect - uavProfilePath.height


                duration: 800
                easing.type: Easing.InOutBack
                running: false
                onStopped: {
                    if(!uavProfilePath.show)
                        uavProfilePath.visible = uavProfilePath.show;
                }
                onStarted: {
                    if(uavProfilePath.show){
                        uavProfilePath.visible = uavProfilePath.show;
                    }
                }
            }
            PropertyAnimation{
                id: animBtnShowUpUAVProfilePath
                target: btnShowPopupUAVProfilePath
                properties: "iconRotate"
                to: uavProfilePath.show ? 0 : 180
                duration: 800
                easing.type: Easing.InExpo
                running: false
                onStopped: {
                    animPopupUAVProfilePath.start()
                }
            }
            Canvas{
                y: 70
                width: UIConstants.sRect * 3
                height: UIConstants.sRect
                z: 5
//                visible   : UIConstants.monitorMode === UIConstants.monitorModeFlight
                anchors.horizontalCenter: uavProfilePath.horizontalCenter
                anchors.top: uavProfilePath.bottom
                anchors.topMargin: 0
                onPaint: {
                    var ctx = getContext("2d");
                    var drawColor = UIConstants.bgAppColor;
                    ctx.strokeStyle = drawColor;
                    ctx.fillStyle = drawColor;
                    ctx.beginPath();
                    ctx.moveTo(0,0);
                    ctx.lineTo(width*1/6,height);
                    ctx.lineTo(width*5/6,height);
                    ctx.lineTo(width,0);
                    ctx.closePath();
                    ctx.fill()
                }
                FlatButtonIcon{
                    id: btnShowPopupUAVProfilePath
                    width: UIConstants.sRect * 3
                    height: UIConstants.sRect
                    z: 5
                    anchors.verticalCenter: parent.verticalCenter
                    anchors.horizontalCenter: parent.horizontalCenter
                    iconSize: UIConstants.fontSize * 2
                    icon: UIConstants.iChevronDown
                    rotation: 180
                    isAutoReturn: true
                    isShowRect: false
                    isSolid: true

                    onClicked: {
                        uavProfilePath.show = !uavProfilePath.show;
                        animBtnShowUpUAVProfilePath.start();
                    }
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
                onClicked: {
                    if(!footerBar.isShowConfirm){
                        if(func === "QUIT_APP"){
                            var compo = Qt.createComponent("qrc:/CustomViews/Dialogs/ConfirmDialog.qml");
                            var confirmDialogObj = compo.createObject(parent,{
                                              "title":mainWindow.itemListName["DIALOG"]["CONFIRM"]["QUIT_APP"]
                                                        [UIConstants.language[UIConstants.languageID]],
                                              "type": "CONFIRM",
                                              "x":parent.width / 2 - UIConstants.sRect * 13 / 2,
                                              "y":parent.height / 2 - UIConstants.sRect * 6 / 2,
                                              "z":200});
                            confirmDialogObj.clicked.connect(function (type,func){
                                if(func === "DIALOG_OK"){
                                    confirmDialogObj.destroy();
                                    compo.destroy();
                                    footerBar.isShowConfirm = false;
                                    computer.quitApplication();
                                }else if(func === "DIALOG_CANCEL"){
                                    confirmDialogObj.destroy();
                                    compo.destroy();
                                    footerBar.isShowConfirm = false;
                                }
                            });
                        }if(func === "RESTART_APP"){
                            var compo = Qt.createComponent("qrc:/CustomViews/Dialogs/ConfirmDialog.qml");
                            var confirmDialogObj = compo.createObject(parent,{
                                              "title":mainWindow.itemListName["DIALOG"]["CONFIRM"]["RESTART_APP"]
                                                        [UIConstants.language[UIConstants.languageID]],
                                              "type": "CONFIRM",
                                              "x":parent.width / 2 - UIConstants.sRect * 13 / 2,
                                              "y":parent.height / 2 - UIConstants.sRect * 6 / 2,
                                              "z":200});
                            confirmDialogObj.clicked.connect(function (type,func){
                                if(func === "DIALOG_OK"){
                                    confirmDialogObj.destroy();
                                    compo.destroy();
                                    footerBar.isShowConfirm = false;
                                    computer.restartApplication();
                                }else if(func === "DIALOG_CANCEL"){
                                    confirmDialogObj.destroy();
                                    compo.destroy();
                                    footerBar.isShowConfirm = false;
                                }
                            });
                        }else if(func === "QUIT_COM"){
                            var compo = Qt.createComponent("qrc:/CustomViews/Dialogs/ConfirmDialog.qml");
                            var confirmDialogObj = compo.createObject(parent,{
                                              "title":mainWindow.itemListName["DIALOG"]["CONFIRM"]["SHUTDOWN_COM"]
                                                [UIConstants.language[UIConstants.languageID]],
                                              "type": "CONFIRM",
                                              "x":parent.width / 2 - UIConstants.sRect * 13 / 2,
                                              "y":parent.height / 2 - UIConstants.sRect * 6 / 2,
                                              "z":200});
                            confirmDialogObj.clicked.connect(function (type,func){
                                if(func === "DIALOG_OK"){
                                    confirmDialogObj.destroy();
                                    compo.destroy();
                                    footerBar.isShowConfirm = false;
                                    computer.shutdownComputer();
                                }else if(func === "DIALOG_CANCEL"){
                                    confirmDialogObj.destroy();
                                    compo.destroy();
                                    footerBar.isShowConfirm = false;
                                }
                            });
                        }
                    }
                }
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
            border.color: UIConstants.navIconColor //UIConstants.grayColor
            radius: UIConstants.rectRadius
            anchors {
                bottom: parent.top;
                bottomMargin: UIConstants.rectRadius
                horizontalCenter: parent.horizontalCenter;
            }
            Connections{
                target: cameraController.gimbal
                onFunctionHandled:{
                    toastFlightControler.callActionAppearance = false;
                    toastFlightControler.rejectButtonAppearance = false;
                    mapPane.createMarkerSymbol(target,"MARKER_TARGET",
                                               Number(mapPane.markerModel.rowCount() + 1).toString());
                    toastFlightControler.toastContent =
                            itemListName["MESSAGE"][message][UIConstants.language[UIConstants.languageID]] +
                            " : " +
                            Number(distance).toFixed(7) + "m";
                    toastFlightControler.show();
                }
            }
        }
        onDoLoadMap: {
            console.log("mapPane.dataPath = "+mapPane.dataPath);
            if(!isShowConfirm){
                isShowConfirm = true;
                var x = mainWindow.width/2-UIConstants.sRect*10/2;
                var y = mainWindow.height/2-UIConstants.sRect*15/2;
                var compo = Qt.createComponent("qrc:/CustomViews/Dialogs/MapDialog.qml");
                var fileDialogObj = compo.createObject(parent,{"title":mainWindow.itemListName["DIALOG"]["LOAD_MAP"]
                                                           [UIConstants.language[UIConstants.languageID]],
                                                           "folder":mapPane.dataPath+"tpk",
                                                           //                                                            "nameFilters":["*.tpk"],
                                                           "x": x,
                                                           "y": y,
                                                           "z": 200
                                                       });
                fileDialogObj.fileSelected.connect(function (file){
                    mapPane.setMap(file);
                    FCSConfig.changeData("MapDefault",file);
                });

                fileDialogObj.modeSelected.connect(function (mode){
                    if(mode === "ONLINE"){
                        mapPane.setMapOnline();
                        FCSConfig.changeData("MapDefault","");
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
                var fileDialogObj = compo.createObject(parent,{"title":mainWindow.itemListName["DIALOG"]["LOAD_MISSION"]
                                                           [UIConstants.language[UIConstants.languageID]],
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
                var fileDialogObj = compo.createObject(parent,{ "title":mainWindow.itemListName["DIALOG"]["LOAD_MISSION"]
                                                           [UIConstants.language[UIConstants.languageID]],
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
        onPreflightCheckClear:{
            preflightCheck.reload();
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
                                  "title":mainWindow.itemListName["DIALOG"]["CONFIRM"]["FLIGHT_MODE"]
                                            [UIConstants.language[UIConstants.languageID]]+" Auto",
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
                                  "title":mainWindow.itemListName["DIALOG"]["CONFIRM"]["FLIGHT_MODE"]
                                        [UIConstants.language[UIConstants.languageID]]+" Guided",
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
                                  "title":mainWindow.itemListName["DIALOG"]["CONFIRM"]["WANT_TO"]
                                      [UIConstants.language[UIConstants.languageID]]+
                                      "\n"+(!vehicle.armed?
                                                (mainWindow.itemListName["DIALOG"]["CONFIRM"]["ARM"]
                                                 [UIConstants.language[UIConstants.languageID]]+" and "):
                                                "")
                                        +mainWindow.itemListName["DIALOG"]["CONFIRM"]["TAKE_OFF"]
                                        [UIConstants.language[UIConstants.languageID]]+" ?",
                                  "type": "CONFIRM",
                                  "x":parent.width / 2 - UIConstants.sRect * 13 / 2,
                                  "y":parent.height / 2 - UIConstants.sRect * 6 / 2,
                                  "z":200});
                    confirmDialogObj.clicked.connect(function (type,func){
                        if(func === "DIALOG_OK"){
                            vehicle.startMission();
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
                    var maxValue = 500;
                    var currentValue = 0;
                    if(vehicle.vehicleType === 2 || vehicle.vehicleType == 14){
                        minValue = 10;
                        maxValue = 500;
                    }else {
                        minValue = 150;
                        maxValue = 3000;
                    }

                    var compo = Qt.createComponent("qrc:/CustomViews/Dialogs/AltitudeEditor.qml");
                    var confirmDialogObj = compo.createObject(parent,{
                                      "x":parent.width / 2 - UIConstants.sRect * 14 / 2,
                                      "y":parent.height / 2 - UIConstants.sRect * 18 / 4,
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
                        maxValue = 45;
                    }else {
                        minValue = 85;
                        maxValue = 110;
                    }
                    var compo = Qt.createComponent("qrc:/CustomViews/Dialogs/SpeedEditor.qml");
                    var confirmDialogObj = compo.createObject(parent,{
                                                                  "x":parent.width / 2 - UIConstants.sRect * 14 / 2,
                                                                  "y":parent.height / 2 - UIConstants.sRect * 18 / 4,
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
                                                                  "y":parent.height / 2 - UIConstants.sRect * 18 / 4,
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

        onDoStartEngine: {
            if(!isShowConfirm){
                isShowConfirm = true;
                console.log("Do Start Engine");
                var compo = Qt.createComponent("qrc:/CustomViews/Dialogs/ConfirmDialog.qml");
                var confirmDialogObj = compo.createObject(parent,{
                    "title":"Are you sure to want to start Engine\n",
                    "type": "CONFIRM",
                    "x":parent.width / 2 - UIConstants.sRect * 13 / 2,
                    "y":parent.height / 2 - UIConstants.sRect * 6 / 2,
                    "z":200});
                confirmDialogObj.clicked.connect(function (type,func){
                    if(func === "DIALOG_OK"){
                        vehicle.startEngine()
//                            navbar.startFlightTimer();
                    }else if(func === "DIALOG_CANCEL"){

                    }
                    confirmDialogObj.destroy();
                    compo.destroy();
                    footerBar.isShowConfirm = false;
                });
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
            mapPane.selectedIndex ++;
            if(mapPane.selectedIndex > mapPane.lastWPIndex()){
                mapPane.selectedIndex = 0;
            }
            mapPane.focusOnWP(mapPane.selectedIndex);
        }

        onDoDownloadPlan: {
            planController.loadFromVehicle();
        }

        function uploadMission(){
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

        onDoUploadPlan: {
            var totalDistance = mapPane.getTotalDistanceWP()
            if(totalDistance > UIConstants.maxDistance){
                if(!isShowConfirm){
                    isShowConfirm = true;
                    console.log("Check total distance");
                    var compo = Qt.createComponent("qrc:/CustomViews/Dialogs/ConfirmDialog.qml");
                    var confirmDialogObj = compo.createObject(parent,{
                                                                  "title":mainWindow.itemListName["DIALOG"]["CONFIRM"]["TOTAL_DISTANCE"]
                                                                  [UIConstants.language[UIConstants.languageID]],
                                                                  "type": "CONFIRM",
                                                                  "x":parent.width / 2 - UIConstants.sRect * 13 / 2,
                                                                  "y":parent.height / 2 - UIConstants.sRect * 6 / 2,
                                                                  "z":200});
                    confirmDialogObj.clicked.connect(function (type,func){
                        if(func === "DIALOG_OK"){
                            uploadMission()
                        }else if(func === "DIALOG_CANCEL"){

                        }
                        confirmDialogObj.destroy();
                        compo.destroy();
                        footerBar.isShowConfirm = false;
                    });
                }
            }
            else
                uploadMission();
        }

        onDoGoWP: {

            if(mapPane.selectedIndex > 0){
                if(vehicle.flightMode === "Guided"){
                    vehicle.flightMode = "Auto";
                }
                vehicle.setCurrentMissionSequence(mapPane.selectedIndex);
                mapPane.isGotoWP = true;
                missionController.forceCurrentWP = true;//added by nhatdn1
            }else if(mapPane.selectedIndex === 0){
                if(!isShowConfirm){
                    isShowConfirm = true;
                    console.log("Do RTL");
                    var compo = Qt.createComponent("qrc:/CustomViews/Dialogs/ConfirmDialog.qml");
                    var confirmDialogObj = compo.createObject(parent,{
                                  "title":mainWindow.itemListName["DIALOG"]["CONFIRM"]["FLIGHT_MODE"]
                                         [UIConstants.language[UIConstants.languageID]]+" RTL",
                                  "type": "CONFIRM",
                                  "x":parent.width / 2 - UIConstants.sRect * 13 / 2,
                                  "y":parent.height / 2 - UIConstants.sRect * 6 / 2,
                                  "z":200});
                    confirmDialogObj.clicked.connect(function (type,func){
                        if(func === "DIALOG_OK"){
                            vehicle.flightMode = "RTL";
                            mapPane.isGotoWP = true;
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
                                  "title":mainWindow.itemListName["DIALOG"]["CONFIRM"]["WANT_TO"]
                                        [UIConstants.language[UIConstants.languageID]]+" \n"+(!vehicle.armed?
                                          mainWindow.itemListName["DIALOG"]["CONFIRM"]["ARM"]
                                            [UIConstants.language[UIConstants.languageID]]:
                                          mainWindow.itemListName["DIALOG"]["CONFIRM"]["DISARM"]
                                            [UIConstants.language[UIConstants.languageID]]
                                    )+"?",
                                  "type": "CONFIRM",
                                  "x":parent.width / 2 - UIConstants.sRect * 13 / 2,
                                  "y":parent.height / 2 - UIConstants.sRect * 6 / 2,
                                  "z":200});
                confirmDialogObj.clicked.connect(function (type,func){
                    if(func === "DIALOG_OK"){
                        vehicle.setArmed(!vehicle.armed);
                        if(arm){
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
                              "title":mainWindow.itemListName["DIALOG"]["CONFIRM"]["GO_LOCATION_NOT_GUIDED"]
                                [UIConstants.language[UIConstants.languageID]],
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
                              "title":mainWindow.itemListName["DIALOG"]["CONFIRM"]["GO_LOCATION_GUIDED"]
                                [UIConstants.language[UIConstants.languageID]]+"?",
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
    CameraStateManager{
        id: camState
        Timer{
            id: timerLoadVideo
            interval: 100
            repeat: false
            onTriggered: {
                if(camState.gcsExportVideo){
                    var compo = Qt.createComponent("qrc:/CustomViews/Bars/VideoExternal.qml");
                    var confirmDialogObj = compo.createObject(parent,{
                                                                  "x":parent.width / 2 - UIConstants.sRect * 19 / 2,
                                                                  "y":parent.height / 2 - UIConstants.sRect * 13 / 2,
                                                                  "camState": camState,
                                                                  "player": cameraController.videoEngine});
                }
            }
        }

        onGcsExportVideoChanged: {
            console.log("Export video = "+gcsExportVideo);
            timerLoadVideo.start();
        }
    }
    Timer{
        id: timerRequestData
        interval: 100; repeat: true;
        running: false
        property int countGetData: 0
        onTriggered: {
            //            console.log("Get gimbal data");
            var frameID = 0;
            if(USE_VIDEO_CPU || USE_VIDEO_GPU){
                //                frameID = cameraController.videoEngine.frameID;
            }

            var data = cameraController.context.getData(frameID);
            camState.sensorID = data["SENSOR"];
            camState.changeLockMode(data["LOCK_MODE"]);
            camState.record = data["RECORD"];
//            camState.gcsTargetLocalization = data["LOCALIZATION"];
            camState.gcsSearch = data["SEARCH"];
            camState.gcsShare = data["GCS_SHARED"];
            camState.digitalStab = data["STAB_DIGITAL"];
            camState.presetMode = data["PRESET"];
            camState.panPos = data["panPos"];
            camState.tiltPos = data["tiltPos"];
            mapPane.drawTargetLocalization(
                        QtPositioning.coordinate(data["CORNER01"].x,data["CORNER01"].y),
                        QtPositioning.coordinate(data["CORNER02"].x,data["CORNER02"].y),
                        QtPositioning.coordinate(data["CORNER03"].x,data["CORNER03"].y),
                        QtPositioning.coordinate(data["CORNER04"].x,data["CORNER04"].y),
                        QtPositioning.coordinate(data["CENTER"].x,data["CENTER"].y),
                        QtPositioning.coordinate(data["UAV"].x,data["UAV"].y));
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
            }
        }
    }
    Timer{
        repeat: true
        interval: 1000
        running: true
        onTriggered: {
            console.log("=================================================\r\n");
        }
    }

    Timer{
        id: timerStart
        repeat: false
        interval: 2000
        onTriggered: {            
            // --- Map
            // --- Joystick
            joystick.mapFile = "conf/joystick.conf"
            joystick.start();
            // --- Flight
            comVehicle.loadConfig(FCSConfig);
            comVehicle.connectLink();
            vehicle.communication = comVehicle;
            vehicle.planController = planController;
            vehicle.joystick = joystick;
            comTracker.loadConfig(TRKConfig);
            comTracker.connectLink();
            tracker.communication = comTracker;
            tracker.uav = vehicle;

            console.log("CAMERA_CONTROL = "+CAMERA_CONTROL)
            // --- Payload
            if(CAMERA_CONTROL){
                cameraController.loadConfig(PCSConfig);
                cameraController.gimbal.joystick = joystick;
                cameraController.gimbal.setVehicle(vehicle);
                timerRequestData.start();
            }
            if(UC_API)
            {
                timerStartUC.start();
            }
        }
    }
    Timer{
        id: timerStartUC
        repeat: false
        interval: 15000
        running: false
        onTriggered: {
            if(UC_API)
            {
                UcApi.notifyQmlReady();
            }
        }
    }
    Component.onCompleted: {
        UIConstants.changeTheme(UIConstants.themeNormal);
        // --- Map
        if(FCSConfig.value("Settings:MapDefault:Value:data") !== ""){
            mapPane.setMap(FCSConfig.value("Settings:MapDefault:Value:data"));
            mapPane.setMap(FCSConfig.value("Settings:MapDefault:Value:data"));
        }
        if(FCSConfig.value("Settings:VehicleType:Value:data") !== ""){
            mapPane.changeVehicleType(parseInt(FCSConfig.value("Settings:VehicleType:Value:data")));
        }

        if(ApplicationConfig.value("Settings:Language:Value:data") !== "")
            UIConstants.languageID = ApplicationConfig.value("Settings:Language:Value:data");
        timerStart.start();
    }
}
