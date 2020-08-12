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
import CustomViews.SubComponents 1.0
//import QGroundControl 1.0

// ----------------- Include UC
import UC       1.0
//---------------- Component definition ---------------------------------------
Rectangle {
    id: rootItem
    width: UIConstants.sRect * 12
    height: UIConstants.sRect * 10 * 3 /2
    color: "transparent"
    radius: UIConstants.rectRadius
    clip: true
    property bool showInfoDrones: true
    property bool showInfoPMCC: true
    property int fontSize: 15
    signal changeShowState(string type, int id, bool show);
    signal openChatBoxClicked(string ip);
    function showInfo(type,show){
        if(type === "DR"){
            //            lstDroneInfo.height = show? UIConstants.sRect * 8:0
            showInfoDrones = show;
            lstDroneInfo.state = show?"show":"hide"
        }else if(type === "PM_CC"){
            //            lstPMInfo.height = show? UIConstants.sRect * 8:0
            showInfoPMCC = show;
            lstPMInfo.state = show?"show":"hide"
            ccInfo.state = show?"show":"hide"
        }

    }

    Rectangle {
        id: rectHeaderDrones
        height: UIConstants.sRect*3/2
        color: UIConstants.bgAppColor
        anchors.right: parent.right
        anchors.rightMargin: 0
        anchors.left: parent.left
        anchors.leftMargin: 0
        anchors.top: parent.top
        anchors.topMargin: 0
        IconSVG{
            id: iconDroneOnline
            width: height
            size: UIConstants.sRect
            anchors.top: parent.top
            anchors.topMargin: 2
            anchors.left: parent.left
            anchors.leftMargin: 2
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 2
            source: "qrc:/assets/images/icons/Quad.svg"
            color: "green"
        }

        Label {
            id: lblDroneOnline
            text: UC_API?"Drone Online "+"["+Number(UCDataModel.listRooms.length)+"]":""
            color: UIConstants.textColor
            font.family: UIConstants.appFont
            font.pixelSize: UIConstants.fontSize
            anchors.right: parent.right
            anchors.rightMargin: 0
            anchors.left: parent.left
            anchors.leftMargin: UIConstants.sRect * 2
            anchors.top: parent.top
            anchors.topMargin: 0
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 0
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignLeft
        }
    }
    OnlineFCSs{
        id: lstDroneInfo
//        interactive: false
        height: UIConstants.sRect * 3
        anchors.right: parent.right
        anchors.rightMargin: 0
        anchors.left: parent.left
        anchors.leftMargin: 0
        anchors.top: rectHeaderDrones.bottom
        anchors.topMargin: 0
    }
    Rectangle {
        id: rectHeaderPM
        height: UIConstants.sRect*3/2
        color: UIConstants.bgAppColor
        anchors.right: parent.right
        anchors.rightMargin: 0
        anchors.left: parent.left
        anchors.leftMargin: 0
        anchors.top: lstDroneInfo.bottom
        anchors.topMargin: 0
        FlatIcon{
            id: iconPMOnline
            width: height
            anchors.top: parent.top
            anchors.topMargin: 2
            anchors.left: parent.left
            anchors.leftMargin: 2
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 2
            color: "green"
            iconSize: UIConstants.sRect
            icon:  UIConstants.iPatrolMan
        }

        Label {
            id: lblPMOnline
            text: "PM Online "+"["+Number(UCDataModel.listUsers.length).toFixed(0).toString()+"]"
            color: UIConstants.textColor
            font.family: UIConstants.appFont
            font.pixelSize: UIConstants.fontSize
            anchors.right: parent.right
            anchors.rightMargin: 0
            anchors.left: parent.left
            anchors.leftMargin: UIConstants.sRect * 2
            anchors.top: parent.top
            anchors.topMargin: 0
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 0
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignLeft
        }
    }
    OnlineUsers{
        id: lstPMInfo
        anchors.top: rectHeaderPM.bottom
        height: UIConstants.sRect * 6
        //onDoubleClicked: {
        //    openChatBoxClicked(uid);
        //}
    }
    Rectangle {
        id: rectHeaderCC
        height: UIConstants.sRect*3/2
        color: UIConstants.bgAppColor
        anchors.right: parent.right
        anchors.rightMargin: 0
        anchors.left: parent.left
        anchors.leftMargin: 0
        anchors.top: lstPMInfo.bottom
        anchors.topMargin: 0
        FlatIcon{
            id: iconCCOnline
            width: height
            anchors.top: parent.top
            anchors.topMargin: 2
            anchors.left: parent.left
            anchors.leftMargin: 2
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 2
            color: "green"
            iconSize: UIConstants.sRect
            icon:  UIConstants.iCenterCommander
        }

        Label {
            id: lblCCOnline
            text: "C&C Online "+qsTr("[0]")
            color: UIConstants.textColor
            font.family: UIConstants.appFont
            font.pixelSize: UIConstants.fontSize
            anchors.right: parent.right
            anchors.rightMargin: 0
            anchors.left: parent.left
            anchors.leftMargin: UIConstants.sRect * 2
            anchors.top: parent.top
            anchors.topMargin: 0
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 0
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignLeft
        }
    }
    OnlineCCs{
        id: lstCCInfo
        anchors.top: rectHeaderCC.bottom
        height: UIConstants.sRect * 3/2
        anchors.right: parent.right
        anchors.rightMargin: 0
        anchors.left: parent.left
        anchors.leftMargin: 0
        anchors.topMargin: 0
        state: "show"
    }
    FlatButtonIcon {
        id: btnChatIcon
        isShowRect: false
        width: UIConstants.sRect*3/2
        height: UIConstants.sRect*3/2
        iconSize: UIConstants.fontSize * 3 / 2
        icon: UIConstants.iChatIcon
        isSolid: true
        isAutoReturn: true
        iconColor: UIConstants.textFooterColor
        anchors.top: parent.top
        anchors.topMargin: 4
        anchors.right: parent.right
        anchors.rightMargin: 8
        SequentialAnimation{
            id: animNewMessage
            loops: 5
            ColorAnimation{
                target: btnChatIcon
                properties: "iconColor"
                to: "red"
                duration: 100
                easing.type: Easing.Linear
            }
            ColorAnimation{
                target: btnChatIcon
                properties: "iconColor"
                to: UIConstants.textFooterColor
                duration: 100
                easing.type: Easing.Linear
            }
        }
        Connections{
            target: UC_API?UcApi:undefined
            onNewRoomMessage:{
                animNewMessage.start();
            }
        }
        onClicked: {
            rootItem.openChatBoxClicked("");
        }
    }
}
