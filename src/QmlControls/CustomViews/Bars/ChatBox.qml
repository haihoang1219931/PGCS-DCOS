//------------------ Include QT libs ------------------------------------------
import QtQuick.Window 2.2
import QtQuick 2.6
import QtQuick.Controls 2.1
import QtQuick.Layouts 1.3
import QtGraphicalEffects 1.0
import QtQuick 2.0
import QtQuick 2.11
//------------------ Include Custom modules/plugins
import CustomViews.Components 1.0
import CustomViews.SubComponents 1.0
import CustomViews.UIConstants 1.0
import UC           1.0
import io.qdt.dev   1.0
//---------------- Component definition ---------------------------------------
Rectangle {
    id: rootItem
    width: UIConstants.sRect * 12
    height: UIConstants.sRect * 18
    color: UIConstants.transparentBlue
    radius: UIConstants.rectRadius
    clip: true
    property bool isNewMessage: false
    property int receiverID: 0
    property var itemListName:
        UIConstants.itemTextMultilanguages["CHATBOX"]
    signal closeClicked()

    state: "show"
    states: [
        State {
            name: "show"
            PropertyChanges{
                target: rectChat
                visible: true
            }
            PropertyChanges{
                target: rootItem
                width: UIConstants.sRect * 12
                height: UIConstants.sRect * 22
            }
            PropertyChanges{
                target: rectHeader
                width: UIConstants.sRect * 12
                height: UIConstants.sRect * 2
                color: UIConstants.bgAppColor
            }
            PropertyChanges{
                target: btnExit
                icon: UIConstants.iChatClose
            }
            PropertyChanges{
                target: rectChat
                visible: true
            }
            PropertyChanges{
                target: lblChatbox
                visible: true
            }
        },
        State {
            name: "hide"
            PropertyChanges{
                target: rectChat
                visible: true
            }
            PropertyChanges{
                target: rootItem
                height: UIConstants.sRect * 2
                width: UIConstants.sRect * 0
            }
            PropertyChanges{
                target: rectHeader
                height: UIConstants.sRect * 2
                width: UIConstants.sRect * 0
            }
            PropertyChanges{
                target: btnExit
                icon: UIConstants.iChatIcon
            }
            PropertyChanges{
                target: rectChat
                visible: false
            }
            PropertyChanges{
                target: lblChatbox
                visible: false
            }
        }
    ]
    function closeWindows(){
        if(state === "hide"){
            state = "show"
        }else if(state === "show"){
            state = "hide"
        }
    }

    function newMessageWarning(source,msg){
        chatContent.append({"content": source+": "+ msg,"msgPCS" :false})
        chatView.positionViewAtEnd()
        isNewMessage = true;
    }
    Connections{
        target: UC_API?UcApi:undefined
        onUpdateRoom:{
            var dataObject =  JSON.parse(dataObjectStr);
            var regex = /[a-zA-Z0-9\s]+/;
            var re =  dataObject.participant.name.match(regex);
            if (re.length > 0){
                newMessageWarning(re[0], dataObject.action +" room");
            }
        }

        //--- Signal notify that just received new room message
        onNewRoomMessage:{
//            console.log("sourceUserUid: "+sourceUserUid);
//            console.log("sourceUserName: "+sourceUserName);
//            var r = /\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b/;
//            var t = sourceIp.match(r);
            console.log(sourceUserName+":"+msg);
            if(!sourceUserName.includes(UcApi.getStationName())){
                var regex = /[a-zA-Z0-9\s]+/;
                var userName =  sourceUserName.match(regex);
                if (userName.length > 0){
                    newMessageWarning(userName[0], msg);
                }
            }
        }
    }


    Rectangle {
        id: rectHeader
        height: UIConstants.sRect * 3/2
        color: UIConstants.bgAppColor
        anchors.right: parent.right
        anchors.rightMargin: 0
        anchors.left: parent.left
        anchors.leftMargin: 0
        anchors.top: parent.top
        anchors.topMargin: 0
        z:1

        FlatButtonIcon {
            id: btnExit
            x: 262
            iconSize: UIConstants.sRect
            isShowRect: false
            width: UIConstants.sRect * 3/2
            height: UIConstants.sRect * 3/2
            anchors.verticalCenter: parent.verticalCenter
            icon: UIConstants.iChatClose
            isSolid: true
            isAutoReturn: true
            iconColor: UIConstants.textFooterColor
            anchors.right: parent.right
            anchors.rightMargin: 8
            onClicked: {
                rootItem.closeClicked();
            }
        }
        state: rootItem.isNewMessage ? "message"+"-"+rootItem.state:"normal"+"-"+rootItem.state
        states: [
            State{
                name: "normal-hide"
                PropertyChanges{
                    target: btnExit
                    iconColor: UIConstants.textFooterColor
                }
            },
            State{
                name: "message-hide"
            },
            State{
                name: "normal-show"
                PropertyChanges{
                    target: rectHeader
                    color: UIConstants.bgAppColor
                }
            },
            State{
                name: "message-show"
            }
        ]
        transitions: [
            Transition {
                from: "normal-hide"
                to: "message-hide"
                SequentialAnimation{
                    loops: 5
                    ColorAnimation{
                        target: btnExit
                        properties: "iconColor"
                        to: UIConstants.textFooterColor
                        duration: 100
                        easing.type: Easing.Linear
                    }
                    ColorAnimation{
                        target: btnExit
                        properties: "iconColor"
                        to: "red"
                        duration: 100
                        easing.type: Easing.Linear
                    }
                }
            },
            Transition {
                from: "normal-show"
                to: "message-show"
                SequentialAnimation{
                    loops: 5
                    ColorAnimation{
                        target: rectHeader
                        properties: "color"
                        to: "red"
                        duration: 100
                        easing.type: Easing.Linear
                    }
                    ColorAnimation{
                        target: rectHeader
                        properties: "color"
                        to: UIConstants.bgAppColor
                        duration: 100
                        easing.type: Easing.Linear
                    }
                }
            }
        ]
        Label {
            id: lblChatbox
            y: 12
            height: 30
            text: itemListName["TITTLE"]
                  [UIConstants.language[UIConstants.languageID]]
            font.family: UIConstants.appFont
            font.pixelSize: UIConstants.fontSize
            color: UIConstants.textColor
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignLeft
            anchors.right: parent.right
            anchors.rightMargin: 86
            anchors.left: parent.left
            anchors.leftMargin: 8
            anchors.verticalCenter: parent.verticalCenter
        }
    }

    Rectangle {
        id: rectChat
        color: "transparent"
        anchors.top: rectHeader.bottom
        anchors.topMargin: 0
        anchors.right: parent.right
        anchors.rightMargin: 1
        anchors.left: parent.left
        anchors.leftMargin: 1
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 1
        z:0
        function sendMessage(receiver)
        {
            // toogle focus to force end of input method composer
            var hasFocus = input.focus;
            input.focus = false;
            var data = input.text
            chatContent.append({content: receiver+": "+ data, msgPCS: true})
            chatView.positionViewAtEnd()
            input.focus = hasFocus;
            input.clear()
        }
        TextArea{
            id: input
            anchors.right: sendButton.left
            anchors.rightMargin: 5
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 5
            anchors.leftMargin: 5
            anchors.left: parent.left
            height: UIConstants.sRect
            color: UIConstants.textColor
            wrapMode: TextArea.Wrap
            Layout.fillHeight: true
            font.family: UIConstants.appFont
            font.pixelSize: UIConstants.fontSize
            placeholderText: itemListName["MESSAGE"]
                             [UIConstants.language[UIConstants.languageID]]
            Keys.onReturnPressed: {
                UcApi.sendMsgToRoom(input.text)
                rectChat.sendMessage(UcApi.getStationName());
            }
            background: Rectangle{
                anchors.fill: parent
                color: UIConstants.transparentColor
                border.width: 1
                border.color: UIConstants.grayLighterColor
                radius: UIConstants.rectRadius
            }
        }

        FlatButtonIcon {
            id: sendButton
            width: UIConstants.sRect
            height: UIConstants.sRect
            anchors.right: parent.right
            anchors.rightMargin: 8
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 5
            icon: UIConstants.iSendPaperPlane
            isSolid: true
            isAutoReturn: true
            isShowRect: false
            border.width: 1
            border.color: UIConstants.grayLighterColor
            iconSize: UIConstants.fontSize
            onClicked: {
                UcApi.sendMsgToRoom(input.text)
                rectChat.sendMessage(UcApi.getStationName());
            }
        }


        ListView {
            id: chatView
            anchors.top: parent.top
            anchors.topMargin: 0
            anchors.bottom: input.top
            anchors.bottomMargin: 5
            anchors.right: parent.right
            anchors.rightMargin: 5
            anchors.left: parent.left
            anchors.leftMargin: 5
            model: ListModel {
                id: chatContent
            }
            clip: true
            spacing: 5
            delegate: Component {

                Label {
                    width: parent.width
                    text: content
                    color: UIConstants.textColor
                    font.family: UIConstants.appFont
                    font.pixelSize: UIConstants.fontSize
                    wrapMode: TextArea.Wrap
                    horizontalAlignment: !msgPCS?Text.AlignRight:Text.AlignLeft
                    Layout.fillHeight: true
                }
            }
        }
    }
}
