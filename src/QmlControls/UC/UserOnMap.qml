import QtQuick 2.9
import QtQuick.Window 2.2
import QtQuick.Controls 2.4
import QtQuick.Layouts 1.3
import QtWebEngine 1.7
import QtGraphicalEffects 1.0

import QSyncable 1.0
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0
import io.qdt.dev   1.0

//-------- :User location on map
Item {
    id: rootComponent
    anchors.fill: parent

    //---- Properties
    property alias pcdVideoSharing: pcdVideoView.sharing
    property alias pcdActive: pcdVideoView.pcdActive
    property var pcdVideoView: pcdVideoView
    property var listUsersOnMap: listUsersOnMap
    property int currentPcdId: -1  //--- User sequence in list users model
    property int currentSelectedId: -1

    //---- Signals
    signal invalidOpenPcdVideo(var invalidCase)
    function updateUCClientPosition(id, newScreenX, newScreenY){
        listUsersOnMap.itemAt(id).screenX_  = newScreenX;
        listUsersOnMap.itemAt(id).screenY_  = newScreenY;
    }

    function updatePcdVideo(id){
//        console.log("updatePcdVideo["+id+"]");
        if(id >= 0){
            pcdVideoView.x = listUsersOnMap.itemAt(id).screenX_+80;
            pcdVideoView.y = listUsersOnMap.itemAt(id).screenY_-70;

            if(currentPcdId !== id){
                currentPcdId = id;
            }
        }
    }
    //--- Users on Map
    Repeater {
        id: listUsersOnMap
        model: JsonListModel {
           keyField: "uid"
           source: JSON.parse(JSON.stringify(Object.keys(UCDataModel.listUsers).map(function(key) {return UCDataModel.listUsers[key]})))
           fields: ["ipAddress", "userId", "roomName", "available", "role", "shared", "latitude", "longitude", "uid", "name", "isSelected", "warning", "connectionState"]
        }

        delegate: Rectangle {
            id: userOnMap
            property bool show_popup: (model.isSelected && (role !== UserRoles.FCS)) || (pcdVideoView.visible && pcdVideoView.pcdActive === model.uid) || model.warning
            property string spreadColor: model.warning ? "#fc5c65" : "#1abc9c"
            property string ipAddress_: model.ipAddress
            property string userId_: model.userId
            property string roomName_: model.roomName
            property bool available_: model.available
            property string role_: model.role
            property bool shared_: model.shared
            property double latitude_: model.latitude
            property double longitude_: model.longitude
            property string uid_: model.uid
            property string name_: model.name
            property int screenX_: 0
            property int screenY_: 0
            property int connectionState_: model.connectionState
            property bool isSelected_: model.isSelected
            onLatitude_Changed: {
                mapPane.updateUsersOnMap();
            }
            onLongitude_Changed: {
                mapPane.updateUsersOnMap();
            }

            width: 35
            height: 35
            x: screenX_
            y: screenY_
            color: "#16a085"
            radius: 20
            z: model.isSelected ? 99999 : 99998
            Rectangle {
                id: rect
                anchors.fill: parent
                color: userOnMap.spreadColor
                radius: 20
                opacity: userOnMap.show_popup ? 1 : 0
            }

            Text {
                anchors.centerIn: parent
                text: UIConstants.iPatrolMan
                font{ pixelSize: 16;
                    weight: Font.Bold;
                    //family: ExternalFontLoader.solidFont
                }
                color: "#bdc3c7"
            }

            Text {
                id: flag
                text: UIConstants.iFlag
                font{ pixelSize: 20;
                    weight: Font.Bold;
                    //family: ExternalFontLoader.solidFont
                }
                color: "#fc5c65"
                opacity: 0.9
                y: - 25
                x: 30
                transform: Rotation { angle: 30 }
                visible: userOnMap.show_popup
            }

            Rectangle {
                id: popup
                width: 120
                height: 60
                color: "#dff9fb"
                y: - 70
                anchors.horizontalCenter: parent.horizontalCenter
                radius: 5
                border {width: 1; color: "#a5b1c2"}
                visible: userOnMap.show_popup
                z: 100000
                Column {
                    anchors.fill: parent
                    Rectangle {
                        width: parent.width
                        height: parent.height / 2
                        color: "transparent"
                        Text {
                            text: userOnMap.name_
                            anchors.centerIn: parent
                            font.pixelSize: UIConstants.fontSize
                            color: "#2c3a47"
                        }

                        Canvas {
                            anchors.fill: parent
                            onPaint: {
                                var ctx = getContext("2d");
                                ctx.lineWidth = 3;
                                ctx.strokeStyle = "#d1d8e0";
                                ctx.beginPath();
                                ctx.moveTo(20, height);
                                ctx.lineTo(width - 20, height);
                                ctx.closePath();
                                ctx.stroke();
                            }
                        }
                    }
                    Rectangle {
                        id: rectBtns
                        width:  parent.width
                        height: parent.height / 2
                        color: "transparent"
                        Row {
                            anchors.fill: parent
                            anchors.leftMargin: parent.width *1/9
                            spacing: parent.width / 9
                            Rectangle {
                                id: popupBtn1
                                width: rectBtns.width / 3
                                height: rectBtns.height - 5
                                color: "#2d98da"
                                x: 20
                                radius: 5
                                border {width: 1; color: "#d1d8e0"}
                                enabled: userOnMap.available_ || userOnMap.roomName_
                                opacity: userOnMap.connectionState_ ? 1 : .5
                                Text {
                                    id: popupBtn1Txt
                                    anchors.centerIn: parent
                                    text: userOnMap.available_ ? UIConstants.iAdd :
                                                                 (userOnMap.roomName_ ?
                                                                      UIConstants.iUserMinus : UIConstants.iMinus)
                                    font.bold: true
                                    font.pixelSize: UIConstants.fontSize
                                    color: "#fff"
                                }
                                MouseArea {
                                    anchors.fill: parent
                                    visible: userOnMap.connectionState_ ? true : false
                                    onClicked: {
                                        if( !userOnMap.available_ && userOnMap.roomName_ === UcApi.getRoomName() ) {
                                            UcApi.removePcdFromRoom(userOnMap.uid_);
                                            UCDataModel.updateUser(userOnMap.uid_, 12, false);
                                            pcdVideoView.visible=false;
                                        } else {
                                            UcApi.addPcdToRoom(userOnMap.uid_);
                                        }
                                    }
                                }

                                Canvas {
                                    anchors.fill: parent
                                    visible: !userOnMap.connectionState_
                                    onPaint: {
                                        var ctx = getContext("2d");
                                        ctx.lineWidth = 2;
                                        ctx.strokeStyle = "#dfe4ea";
                                        ctx.beginPath();
                                        ctx.moveTo(0, 0);
                                        ctx.lineTo(width, height);
                                        ctx.closePath();
                                        ctx.stroke();
                                    }
                                }
                            }
                            Rectangle {
                                id: popupBtn2
                                width: rectBtns.width / 3
                                height: rectBtns.height - 5
                                color: "#3867d6"
                                x: parent.width / 9
                                radius: 5
                                border {width: 1; color: "#d1d8e0"}
                                opacity: userOnMap.connectionState_ ? 1 : .5
                                Text {
                                    id: videoIcon
                                    anchors.centerIn: parent
                                    text: "\uf03d"
                                    font{ pixelSize: 12;
                                        weight: Font.Bold;
                                        //family: ExternalFontLoader.solidFont
                                    }
                                    color: "#fff"
                                    font.bold: true
                                }
                                SequentialAnimation {
                                    running: warning
                                    loops: Animation.Infinite
                                    PropertyAnimation {
                                        target: videoIcon
                                        properties: "color"
                                        to: "#fc5c65"
                                        duration: 500
                                        easing.type: Easing.InOutBounce
                                    }
                                    PropertyAnimation {
                                        target: videoIcon
                                        properties: "color"
                                        to: "#fff"
                                        duration: 300
                                        easing.type: Easing.InOutBounce
                                    }
                                    onRunningChanged: {
                                        if (!model.warning) {
                                            videoIcon.color = "#fff";
                                        }
                                    }
                                }

                                MouseArea {
                                    anchors.fill: parent
                                    visible: userOnMap.connectionState_ ? true : false
                                    onClicked: {
                                        if( userOnMap.roomName_ === UcApi.getRoomName() ) {
                                            if( pcdVideoView.visible === true ) {
                                                //toastContent.text = "Chỉ mở được video của 1 PCD tại 1 thời điểm.! \nĐóng PCD video trước khi mở video của PCD khác";
                                                //toast.show();
                                                //invalidOpenPcdVideo("duplicate");
                                                UCEventListener.invalidOpenPcdVideo(UCEventEnums.PCD_VIDEO_DUPLICATE);
                                            } else {
                                                pcdVideoView.pcdActive =  userOnMap.uid_;
                                                pcdVideoView.sharing = userOnMap.shared_;
                                                UcApi.requestOpenParticularPcdVideo(userOnMap.uid_);
                                                pcdVideoView.visible = true;
                                                updatePcdVideo(index);
                                                UCDataModel.updateUser(userOnMap.uid_, 12, false);
                                            }
                                        } else {
                                            //toast.toastContent  = "Ko hợp lệ ! \n Chỉ xem được hình ảnh của PCD ở trong phòng !" ;
                                            //toast.show();
                                            //invalidOpenPcdVideo("not-in-room");
                                            UCEventListener.invalidOpenPcdVideo(UCEventEnums.USER_NOT_IN_ROOM);
                                        }
                                    }
                                }

                                Canvas {
                                    anchors.fill: parent
                                    visible: !userOnMap.connectionState_
                                    onPaint: {
                                        var ctx = getContext("2d");
                                        ctx.lineWidth = 2;
                                        ctx.strokeStyle = "#dfe4ea";
                                        ctx.beginPath();
                                        ctx.moveTo(0, 0);
                                        ctx.lineTo(width, height);
                                        ctx.closePath();
                                        ctx.stroke();
                                    }
                                }
                            }
                        }
                    }
                }

                Canvas {
                    width: parent.width
                    height: parent.height + 10
                    z: 100
                    onPaint: {
                        var ctx = getContext("2d");
                        ctx.lineWidth = 1;
                        ctx.strokeStyle = "#a5b1c2"
                        ctx.fillStyle = "#dff9fb";
                        ctx.beginPath();
                        ctx.moveTo(width / 2 - 7, height - 11);
                        ctx.lineTo(width / 2, height);
                        ctx.lineTo(width / 2 + 7, height - 11);
                        //ctx.lineTo(width / 2 - 7, height - 10);
                        ctx.closePath();
                        ctx.fill();
                        ctx.stroke();
                    }
                }

            }

            MouseArea {
                anchors.fill: parent
                onClicked: {
                    for (var i = 0; i < listUsersOnMap.count; i++) {
                        if (i != index) {
                            listUsersOnMap.itemAt(i).z = 99998;
                        }
                    }
                    if(/*parent.room === UcApi.getRoomName() &&*/ userOnMap.role_ !== UserRoles.FCS) {
                        if (pcdVideoView.visible) {
                            pcdVideoView.visible = false;
                            UcApi.closePcdVideo(pcdVideoView.pcdActive);
                        }
                        UCDataModel.updateUser(userOnMap.uid_, 10, !userOnMap.isSelected_);
                        UCEventListener.pointToPcdFromSidebar(userOnMap.uid_, !userOnMap.isSelected_);
                    }
                }
            }

            SequentialAnimation {
                id: mapActiveAnimation
                running: userOnMap.show_popup
                loops: Animation.Infinite
                ParallelAnimation {
                    ScaleAnimator {
                        target: rect
                        from: 1
                        to: 1.1
                        duration: 400
                    }
                    NumberAnimation {
                        target: rect
                        properties: "opacity"
                        from: 0
                        to: 0.5
                        duration: 400
                    }
                }

                ParallelAnimation {
                    ScaleAnimator {
                        target: rect
                        from: 1.1
                        to: 1.2
                        duration: 500
                    }
                    NumberAnimation {
                        target: rect
                        properties: "opacity"
                        from: 0.5
                        to: 1
                        duration: 500
                    }
                }

                ParallelAnimation {
                    ScaleAnimator {
                        target: rect
                        from: 1.4
                        to: 1.5
                        duration: 500
                    }
                    NumberAnimation {
                        target: rect
                        properties: "opacity"
                        from: 1
                        to: 0
                        duration: 500
                    }
                }
            }
        }
    }


    //--- Pcd Video
    PcdVideo {
        id: pcdVideoView
        visible: false
        width: 300
        height: 300
    }

    function reloadPcdVideo() {
        pcdVideoView.reload();
    }

    Connections {
        target: UCEventListener
        onUserIsPointed: {
            console.log("Pointed to " + pcdUid);
            if (pcdVideoView.visible) {
                console.log("Close pcd video if it already opened!!!!");
                pcdVideoView.visible = false;
                UcApi.closePcdVideo(pcdVideoView.pcdActive);
            }
        }
    }
}
