/**
 * ==============================================================================
 * @Project: VCM01TargetViewer
 * @Component: Flat Button
 * @Breif:
 * @Author: Trung Nguyen
 * @Date: 18/02/2019
 * @Language: QML
 * @License: (c) Viettel Aerospace Institude - Viettel Group
 * ============================================================================
 */

//----------------------- Include QT libs -------------------------------------
import QtQuick 2.6
import QtQuick.Controls 2.1

//---------------- Include custom libs ----------------------------------------
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0

//----------------------- Component definition- ------------------------------
Rectangle {
    id: root
    //---------- properties
    color: UIConstants.transparentBlue
    radius: UIConstants.rectRadius
    width: UIConstants.sRect * 24
    height: UIConstants.sRect * 16.5
    border.color: "gray"
    border.width: 1
    property alias waypointModeEnabled: rectWaypointMode.enabled
    property string degreeSymbol : "\u00B0"
    property var lstTxt: [
        txtLat,txtLatD1,txtLatM1,txtLatS1,txtLatD2,txtLatM2,
        txtLon,txtLonD1,txtLonM1,txtLonS1,txtLonD2,txtLonM2,
        txtAGL,txtAMSL,
        txtParam1,txtParam2,txtParam3,txtParam4]

    signal waypointModeChanged(string waypointMode,
                               int param1,int param2,int param3,int param4)
    signal confirmClicked()
    signal cancelClicked()
    signal textChanged(string newText)

    property bool editting: false
    property real latitude: 0
    property real longitude: 0
    property real agl: 0
    property real asl: 0
    property real amsl: 0
    property int param1: 0
    property int param2: 0
    property int param3: 0
    property int param4: 0

    property var validatorAltitude: /^([0-9]|[1-9][0-9]|[1-9][0-9][0-9]|[1-9][0-9][0-9][0-9])/
    property var validatorParam: /^([-]|)([0-9]|[1-9][0-9]|[1-9][0-9][0-9]|[1-9][0-9][0-9][0-9])(\.)([0-9][0-9])/
    property var validatorLat: /^([-]|)([0-9]|[1-8][0-9])(\.)([0-9][0-9][0-9][0-9][0-9][0-9][0-9])/
    property var validatorLon: /^([-]|)([0-9]|[1-9][0-9]|[1][0-7][0-9])(\.)([0-9][0-9][0-9][0-9][0-9][0-9][0-9])/
    property var validatorLatDecimal: /^([-]|)([0-9]|[1-8][0-9])/
    property var validatorLonDecimal: /^([-]|)([0-9]|[1-9][0-9]|[1][0-7][0-9])/
    property var validatorMinute: /^([0-9]|[1-5][0-9])/
    property var validatorSecond: /^([0-9]|[1-5][0-9])/
    property var validatorMinuteFloat: /^([0-9]|[1-5][0-9])(\.)([0-9][0-9][0-9])/
    property var waypointModes: Object.keys(lstWaypointCommand[vehicleType]).reverse()
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

    function changeAllFocus(enable){
        for(var i =0; i < lstTxt.length; i++){
            if(lstTxt[i].focus !== enable)
                lstTxt[i].focus = enable;
        }
    }

    function changeASL(value){
        root.asl = Number(value).toFixed(2);
    }

    function changeCoordinate(_coordinate){
        changeAllFocus(false);
        root.latitude = Number(_coordinate.latitude).toFixed(7);
        root.longitude = Number(_coordinate.longitude).toFixed(7);
        root.agl = Number(_coordinate.altitude).toFixed(2);
    }
    function loadInfo(_coordinate,command,param1,param2,param3,param4){
        changeCoordinate(_coordinate);
        var listCommand = Object.keys(lstWaypointCommand[vehicleType]).reverse();
        for(var index = 0; index < listCommand.length; index++){
            if(lstWaypointCommand[vehicleType][listCommand[index]]["COMMAND"] ===
                    command){
                lstWaypointMode.setCurrentText(listCommand[index]);
                break;
            }
        }
        root.param1 = param1;
        root.param2 = param2;
        root.param3 = param3;
        root.param4 = param4;
    }
    onVisibleChanged: {
        if(!visible){
            root.changeAllFocus(false);
        }
    }
    MouseArea{
        anchors.fill: parent
        hoverEnabled: true
    }

    FlatButtonIcon{
        id: btnConfirm
        y: 192
        height: 30
        width: 60
        icon: UIConstants.iChecked
        isSolid: true
        color: (root.latitude !== 0 && root.longitude !== 0)?"green":"gray"
        anchors.left: parent.left
        anchors.leftMargin: 8
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 8
        isAutoReturn: true
        radius: root.radius
        enabled: (root.latitude !== 0 && root.longitude !== 0)

        onClicked: {
            root.confirmClicked();
            root.changeAllFocus(false);
        }
    }
    FlatButtonIcon{
        id: btnCancel
        x: 102
        y: 192
        height: 30
        width: 60
        icon: UIConstants.iMouse
        isSolid: true
        color: "red"
        anchors.right: parent.right
        anchors.rightMargin: 8
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 8
        isAutoReturn: true
        radius: root.radius
        onClicked: {
            root.cancelClicked();
            root.changeAllFocus(false);
        }
    }

    Rectangle {
        id: rectParam
        y: 26
        width: 348
        height: 293
        color: UIConstants.transparentColor
        border.color: UIConstants.grayColor
        radius: UIConstants.rectRadius
        anchors.left: parent.left
        anchors.leftMargin: 8

        Label {
            id: lblParam1
            x: 47
            y: 172
            width: 72
            height: 25
            color: UIConstants.textColor
            text: lstWaypointCommand[vehicleType][waypointModes[lstWaypointMode.currentIndex]]["PARAM1"]["LABEL"]
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignHCenter
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
        }

        Rectangle {
            x: -5
            y: 203
            width: 158
            height: 25
            color: UIConstants.transparentColor
            radius: 1
            anchors.leftMargin: 4
            border.width: 1
            TextInput {
                id: txtParam1
                color: UIConstants.textColor
                text: focus?text:Number(root.param1).toString()
                anchors.margins: UIConstants.rectRadius/2
                horizontalAlignment: Text.AlignLeft
                anchors.fill: parent
                clip: true
                font.pixelSize: UIConstants.fontSize
                font.family: UIConstants.appFont
                validator: RegExpValidator { regExp: validatorParam }
                enabled: lstWaypointCommand[vehicleType][waypointModes[lstWaypointMode.currentIndex]]["PARAM1"]["EDITABLE"]
                onTextChanged: {
                    if(focus){
                        if(text === "" || isNaN(text)){
                            root.param1 = 0;
                        }else{
                            root.param1 = parseInt(text);
                        }
                    }
                }
            }
            anchors.left: parent.left
            border.color: UIConstants.grayColor
        }

        Label {
            id: lblParam2
            x: 225
            y: 172
            width: 72
            height: 25
            color: UIConstants.textColor
            text: lstWaypointCommand[vehicleType][waypointModes[lstWaypointMode.currentIndex]]["PARAM2"]["LABEL"]
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignHCenter
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
        }

        Rectangle {
            x: -5
            y: 203
            width: 158
            height: 25
            color: UIConstants.transparentColor
            radius: 1
            anchors.leftMargin: 182
            border.width: 1
            TextInput {
                id: txtParam2
                color: UIConstants.textColor
                text: focus?text:Number(root.param2).toString()
                anchors.margins: UIConstants.rectRadius/2
                horizontalAlignment: Text.AlignLeft
                anchors.fill: parent
                clip: true
                font.pixelSize: UIConstants.fontSize
                font.family: UIConstants.appFont
                validator: RegExpValidator { regExp: validatorParam }
                enabled: lstWaypointCommand[vehicleType][waypointModes[lstWaypointMode.currentIndex]]["PARAM2"]["EDITABLE"]
                onTextChanged: {
                    if(focus){
                        if(text === "" || isNaN(text)){
                            root.param2 = 0;
                        }else{
                            root.param2 = parseInt(text);
                        }
                    }
                }
            }
            anchors.left: parent.left
            border.color: UIConstants.grayColor
        }

        Label {
            id: lblParam3
            x: 47
            y: 229
            width: 72
            height: 25
            color: UIConstants.textColor
            text: lstWaypointCommand[vehicleType][waypointModes[lstWaypointMode.currentIndex]]["PARAM3"]["LABEL"]
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignHCenter
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
        }

        Rectangle {
            x: -5
            y: 260
            width: 158
            height: 25
            color: UIConstants.transparentColor
            radius: 1
            anchors.leftMargin: 4
            border.width: 1
            TextInput {
                id: txtParam3
                color: UIConstants.textColor
                text: focus?text:Number(root.param3).toString()
                anchors.margins: UIConstants.rectRadius/2
                horizontalAlignment: Text.AlignLeft
                anchors.fill: parent
                clip: true
                font.pixelSize: UIConstants.fontSize
                font.family: UIConstants.appFont
                validator: RegExpValidator { regExp: validatorParam }
                enabled: lstWaypointCommand[vehicleType][waypointModes[lstWaypointMode.currentIndex]]["PARAM3"]["EDITABLE"]
                onTextChanged: {
                    if(focus){
                        if(text === "" || isNaN(text)){
                            root.param3 = 0;
                        }else{
                            root.param3 = parseInt(text);
                        }
                    }
                }
            }
            anchors.left: parent.left
            border.color: UIConstants.grayColor
        }

        Label {
            id: lblParam4
            x: 225
            y: 229
            width: 72
            height: 25
            color: UIConstants.textColor
            text: lstWaypointCommand[vehicleType][waypointModes[lstWaypointMode.currentIndex]]["PARAM4"]["LABEL"]
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignHCenter
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
        }

        Rectangle {
            x: -5
            y: 260
            width: 158
            height: 25
            color: UIConstants.transparentColor
            radius: 1
            anchors.leftMargin: 182
            border.width: 1
            TextInput {
                id: txtParam4
                color: UIConstants.textColor
                text: focus?text:Number(root.param4).toString()
                anchors.margins: UIConstants.rectRadius/2
                horizontalAlignment: Text.AlignLeft
                anchors.fill: parent
                clip: true
                font.pixelSize: UIConstants.fontSize
                font.family: UIConstants.appFont
                validator: RegExpValidator { regExp: validatorParam }
                enabled: lstWaypointCommand[vehicleType][waypointModes[lstWaypointMode.currentIndex]]["PARAM4"]["EDITABLE"]
                onTextChanged: {
                    if(focus){
                        if(text === "" || isNaN(text)){
                            root.param4 = 0;
                        }else{
                            root.param4 = parseInt(text);
                        }
                    }
                }
            }
            anchors.left: parent.left
            border.color: UIConstants.grayColor
        }
    }

    Rectangle {
        id: rectWaypointMode
        y: 26
        height: 293
        color: UIConstants.transparentColor
        border.color: UIConstants.grayColor
        radius: UIConstants.rectRadius
        anchors.left: rectParam.right
        anchors.leftMargin: 8
        anchors.right: parent.right
        anchors.rightMargin: 8

        SubNav {
            id: lstWaypointMode
            anchors.rightMargin: 2
            anchors.leftMargin: 2
            anchors.bottomMargin: 2
            anchors.topMargin: 2
            anchors.fill: parent
            model: root.waypointModes
            onListViewClicked: {
                waypointModeChanged(choosedItem,
                    param1,param2,param3,param4)
            }
        }
    }

    Rectangle {
        x: -9
        y: 57
        width: 26
        height: 25
        color: UIConstants.transparentColor
        radius: 1
        anchors.leftMargin: 144
        border.color: UIConstants.grayColor
        Label {
            color: UIConstants.textColor
            text: "E"
            clip: true
            horizontalAlignment: Text.AlignHCenter
            anchors.fill: parent
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
            anchors.margins: UIConstants.rectRadius/2
        }
        anchors.left: parent.left
        border.width: 1
    }

    Rectangle {
        x: 0
        y: 57
        width: 26
        height: 25
        color: UIConstants.transparentColor
        radius: 1
        anchors.leftMargin: 324
        border.color: UIConstants.grayColor
        Label {
            color: UIConstants.textColor
            text: "N"
            clip: true
            horizontalAlignment: Text.AlignHCenter
            anchors.fill: parent
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
            anchors.margins: UIConstants.rectRadius/2
        }
        anchors.left: parent.left
        border.width: 1
    }

    Rectangle {
        x: -8
        y: 88
        width: 26
        height: 25
        color: UIConstants.transparentColor
        radius: 1
        anchors.leftMargin: 144
        border.color: UIConstants.grayColor
        Label {
            color: UIConstants.textColor
            text: "E"
            clip: true
            horizontalAlignment: Text.AlignHCenter
            anchors.fill: parent
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
            anchors.margins: UIConstants.rectRadius/2
        }
        anchors.left: parent.left
        border.width: 1
    }

    Rectangle {
        x: -5
        y: 119
        width: 26
        height: 25
        color: UIConstants.transparentColor
        radius: 1
        anchors.leftMargin: 144
        border.color: UIConstants.grayColor
        Label {
            color: UIConstants.textColor
            text: "E"
            clip: true
            horizontalAlignment: Text.AlignHCenter
            anchors.fill: parent
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
            anchors.margins: UIConstants.rectRadius/2
        }
        anchors.left: parent.left
        border.width: 1
    }

    Rectangle {
        x: -9
        y: 88
        width: 26
        height: 25
        color: UIConstants.transparentColor
        radius: 1
        anchors.leftMargin: 324
        border.color: UIConstants.grayColor
        Label {
            color: UIConstants.textColor
            text: "N"
            clip: true
            horizontalAlignment: Text.AlignHCenter
            anchors.fill: parent
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
            anchors.margins: UIConstants.rectRadius/2
        }
        anchors.left: parent.left
        border.width: 1
    }

    Rectangle {
        x: -1
        y: 119
        width: 26
        height: 25
        color: UIConstants.transparentColor
        radius: 1
        anchors.leftMargin: 324
        border.color: UIConstants.grayColor
        Label {
            color: UIConstants.textColor
            text: "N"
            clip: true
            horizontalAlignment: Text.AlignHCenter
            anchors.fill: parent
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
            anchors.margins: UIConstants.rectRadius/2
        }
        anchors.left: parent.left
        border.width: 1
    }

    Rectangle {
        x: -7
        y: 88
        width: 31
        height: 25
        color: UIConstants.transparentColor
        radius: 1
        anchors.leftMargin: 12
        border.color: UIConstants.grayColor
        TextInput {
            id: txtLatD1
            color: UIConstants.textColor
            clip: true
            text: focus?text:parseInt(root.latitude)
            horizontalAlignment: Text.AlignLeft
            anchors.fill: parent
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
            anchors.margins: UIConstants.rectRadius/2
            validator: RegExpValidator { regExp: validatorLatDecimal }
            onTextChanged: {
                if(focus){
                    if(text === "" || isNaN(text)){
                        var newValue = Number(
                        parseFloat(0) +
                        parseFloat(txtLatM1.text)/60 +
                        parseFloat(txtLatS1.text)/3600).toFixed(7);
                        root.latitude = newValue;
                    }else{
                        var newValue = Number(
                        parseFloat(txtLatD1.text) +
                        parseFloat(txtLatM1.text)/60 +
                        parseFloat(txtLatS1.text)/3600).toFixed(7);
                        root.latitude = newValue;
                    }
                }
            }
        }
        anchors.left: parent.left
        border.width: 1
    }

    Rectangle {
        x: 2
        y: 88
        width: 31
        height: 25
        color: UIConstants.transparentColor
        radius: 1
        anchors.leftMargin: 57
        border.color: UIConstants.grayColor
        TextInput {
            id: txtLatM1
            color: UIConstants.textColor
            clip: true
            text: focus?text:Math.round(Math.floor(
                             Math.abs(
                                 (root.latitude-parseInt(root.latitude))*60)
                             )
                         )
            horizontalAlignment: Text.AlignLeft
            anchors.fill: parent
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
            anchors.margins: UIConstants.rectRadius/2
            validator: RegExpValidator { regExp: validatorMinute }
            onTextChanged: {
                if(focus){
                    if(text === "" || isNaN(text)){
                        var newValue = Number(
                        parseFloat(txtLatD1.text) +
                        parseFloat(0)/60 +
                        parseFloat(txtLatS1.text)/3600).toFixed(7);
                        root.latitude = newValue;
                    }else{
                        var newValue = Number(
                        parseFloat(txtLatD1.text) +
                        parseFloat(txtLatM1.text)/60 +
                        parseFloat(txtLatS1.text)/3600).toFixed(7);
                        root.latitude = newValue;
                    }
                }
            }
        }
        anchors.left: parent.left
        border.width: 1
    }

    Rectangle {
        x: -11
        y: 88
        width: 31
        height: 25
        color: UIConstants.transparentColor
        radius: 1
        anchors.leftMargin: 101
        border.color: UIConstants.grayColor
        TextInput {
            id: txtLatS1
            color: UIConstants.textColor
            clip: true
            text: focus?text:parseInt(
                      (Math.abs(
                           (root.latitude-parseInt(root.latitude))*60)-Math.floor(
                                  Math.abs(
                                      (root.latitude-parseInt(root.latitude))*60)
                                  )
                              )*60
                        )
            horizontalAlignment: Text.AlignLeft
            anchors.fill: parent
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
            anchors.margins: UIConstants.rectRadius/2
            validator: RegExpValidator { regExp: validatorSecond }
            onTextChanged: {
                if(focus){
                    if(text === "" || isNaN(text)){
                        var newValue = Number(
                        parseFloat(txtLatD1.text) +
                        parseFloat(txtLatM1.text)/60 +
                        parseFloat(0)/3600).toFixed(7);
                        root.latitude = newValue;
                    }else{
                        var newValue = Number(
                        parseFloat(txtLatD1.text) +
                        parseFloat(txtLatM1.text)/60 +
                        parseFloat(txtLatS1.text)/3600).toFixed(7);
                        root.latitude = newValue;
                    }
                }
            }
        }
        anchors.left: parent.left
        border.width: 1
    }

    Rectangle {
        x: -8
        y: 88
        width: 31
        height: 25
        color: UIConstants.transparentColor
        radius: 1
        anchors.leftMargin: 190
        border.color: UIConstants.grayColor
        TextInput {
            id: txtLonD1
            color: UIConstants.textColor
            clip: true
            text: focus?text:parseInt(root.longitude)
            horizontalAlignment: Text.AlignLeft
            anchors.fill: parent
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
            anchors.margins: UIConstants.rectRadius/2
            validator: RegExpValidator { regExp: validatorLonDecimal }
            onTextChanged: {
                if(focus){
                    if(text === "" || isNaN(text)){
                        var newValue = Number(
                        parseFloat(0) +
                        parseFloat(txtLonM1.text)/60 +
                        parseFloat(txtLonS1.text)/3600).toFixed(7);
                        root.longitude = newValue;
                    }else{
                        var newValue = Number(
                        parseFloat(txtLonD1.text) +
                        parseFloat(txtLonM1.text)/60 +
                        parseFloat(txtLonS1.text)/3600).toFixed(7);
                        root.longitude = newValue;
                    }
                }
            }
        }
        anchors.left: parent.left
        border.width: 1
    }

    Rectangle {
        x: 1
        y: 88
        width: 31
        height: 25
        color: UIConstants.transparentColor
        radius: 1
        anchors.leftMargin: 235
        border.color: UIConstants.grayColor
        TextInput {
            id: txtLonM1
            color: UIConstants.textColor
            clip: true
            text: focus?text:Math.round(Math.floor(
                             Math.abs(
                                 (root.longitude-parseInt(root.longitude))*60)
                             )
                         )
            horizontalAlignment: Text.AlignLeft
            anchors.fill: parent
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
            anchors.margins: UIConstants.rectRadius/2
            validator: RegExpValidator { regExp: validatorMinute }
            onTextChanged: {
                if(focus){
                    if(text === "" || isNaN(text)){
                        var newValue = Number(
                        parseFloat(txtLonD1.text) +
                        parseFloat(0)/60 +
                        parseFloat(txtLonS1.text)/3600).toFixed(7);
                        root.longitude = newValue;
                    }else{
                        var newValue = Number(
                        parseFloat(txtLonD1.text) +
                        parseFloat(txtLonM1.text)/60 +
                        parseFloat(txtLonS1.text)/3600).toFixed(7);
                        root.longitude = newValue;
                    }
                }
            }
        }
        anchors.left: parent.left
        border.width: 1
    }

    Rectangle {
        x: -12
        y: 88
        width: 31
        height: 25
        color: UIConstants.transparentColor
        radius: 1
        anchors.leftMargin: 279
        border.color: UIConstants.grayColor
        TextInput {
            id: txtLonS1
            color: UIConstants.textColor
            clip: true
            text: focus?text:parseInt(
                      (Math.abs(
                           (root.longitude-parseInt(root.longitude))*60)-Math.floor(
                                  Math.abs(
                                      (root.longitude-parseInt(root.longitude))*60)
                                  )
                              )*60
                        )
            horizontalAlignment: Text.AlignLeft
            anchors.fill: parent
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
            anchors.margins: UIConstants.rectRadius/2
            validator: RegExpValidator { regExp: validatorSecond }
            onTextChanged: {
                if(focus){
                    if(text === "" || isNaN(text)){
                        var newValue = Number(
                        parseFloat(txtLonD1.text) +
                        parseFloat(txtLonM1.text)/60 +
                        parseFloat(0)/3600).toFixed(7);
                        root.longitude = newValue;
                    }else{
                        var newValue = Number(
                        parseFloat(txtLonD1.text) +
                        parseFloat(txtLonM1.text)/60 +
                        parseFloat(txtLonS1.text)/3600).toFixed(7);
                        root.longitude = newValue;
                    }
                }
            }
        }
        anchors.left: parent.left
        border.width: 1
    }

    Rectangle {
        x: -13
        y: 119
        width: 31
        height: 25
        color: UIConstants.transparentColor
        radius: 1
        anchors.leftMargin: 12
        border.color: UIConstants.grayColor
        TextInput {
            id: txtLatD2
            color: UIConstants.textColor
            clip: true
            text: focus?text:parseInt(root.latitude)
            horizontalAlignment: Text.AlignLeft
            anchors.fill: parent
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
            anchors.margins: UIConstants.rectRadius/2
            validator: RegExpValidator { regExp: validatorLatDecimal }
            onTextChanged: {
                if(focus){
                    if(text === "" || isNaN(text)){
                        var newValue = Number(parseFloat(0) +
                                               parseFloat(txtLatM2.text)/60).toFixed(7);
                        root.latitude = newValue;
                    }else{
                        var newValue = Number(parseFloat(txtLatD2.text) +
                                               parseFloat(txtLatM2.text)/60).toFixed(7);
                        root.latitude = newValue;
                    }
                }
            }
        }
        anchors.left: parent.left
        border.width: 1
    }

    Rectangle {
        x: 2
        y: 119
        width: 75
        height: 25
        color: UIConstants.transparentColor
        radius: 1
        anchors.leftMargin: 57
        border.color: UIConstants.grayColor
        TextInput {
            id: txtLatM2
            color: UIConstants.textColor
            clip: true
            text: focus?text:Number((root.latitude-parseInt(root.latitude))*60).toFixed(3);
            horizontalAlignment: Text.AlignLeft
            anchors.fill: parent
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
            anchors.margins: UIConstants.rectRadius/2
            validator: RegExpValidator { regExp: validatorMinuteFloat }
            onTextChanged: {
                if(focus){
                    if(text === "" || isNaN(text)){
                        var newValue = Number(parseFloat(txtLatD2.text) +
                                               parseFloat(0)/60).toFixed(7);
                        root.latitude = newValue;
                    }else{
                        var newValue = Number(parseFloat(txtLatD2.text) +
                                               parseFloat(txtLatM2.text)/60).toFixed(7);
                        root.latitude = newValue;
                    }
                }
            }
        }
        anchors.left: parent.left
        border.width: 1
    }

    Rectangle {
        x: -10
        y: 119
        width: 31
        height: 25
        color: UIConstants.transparentColor
        radius: 1
        anchors.leftMargin: 190
        border.color: UIConstants.grayColor
        TextInput {
            id: txtLonD2
            color: UIConstants.textColor
            clip: true
            text: focus?text:parseInt(root.longitude)
            horizontalAlignment: Text.AlignLeft
            anchors.fill: parent
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
            anchors.margins: UIConstants.rectRadius/2
            validator: RegExpValidator { regExp: validatorLonDecimal }
            onTextChanged: {
                if(focus){
                    if(text === "" || isNaN(text)){
                        var newValue = Number(
                                    parseFloat(0) +
                                    parseFloat(txtLonM2.text)/60).toFixed(7);
                        root.longitude = newValue;
                    }else{
                        var newValue = Number(
                                    parseFloat(txtLonD2.text) +
                                    parseFloat(txtLonM2.text)/60).toFixed(7);
                        root.longitude = newValue;
                    }
                }
            }
        }
        anchors.left: parent.left
        border.width: 1
    }

    Rectangle {
        x: -1
        y: 119
        width: 75
        height: 25
        color: UIConstants.transparentColor
        radius: 1
        anchors.leftMargin: 235
        border.color: UIConstants.grayColor
        TextInput {
            id: txtLonM2
            color: UIConstants.textColor
            clip: true
            text: focus?text:Number((root.longitude-parseInt(root.longitude))*60).toFixed(3);
            horizontalAlignment: Text.AlignLeft
            anchors.fill: parent
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
            anchors.margins: UIConstants.rectRadius/2
            validator: RegExpValidator { regExp: validatorMinuteFloat }
            onTextChanged: {
                if(focus){
                    if(text === "" || isNaN(text)){
                        var newValue = Number(
                                    parseFloat(txtLonD2.text) +
                                    parseFloat(0)/60).toFixed(7);
                        root.longitude = newValue;
                    }else{
                        var newValue = Number(
                                    parseFloat(txtLonD2.text) +
                                    parseFloat(txtLonM2.text)/60).toFixed(7);
                        root.longitude = newValue;
                    }
                }
            }
        }
        anchors.left: parent.left
        border.width: 1
    }

    Label {
        id: label3
        x: 232
        y: 144
        width: 72
        height: 25
        color: UIConstants.textColor
        text: qsTr("AGL")
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        font.pixelSize: UIConstants.fontSize
        font.family: UIConstants.appFont
    }

    Rectangle {
        x: 1
        y: 173
        width: 158
        height: 25
        color: UIConstants.transparentColor
        radius: 1
        anchors.leftMargin: 189
        border.color: UIConstants.grayColor
        TextInput {
            id: txtAGL
            color: UIConstants.textColor
            clip: true
            text: focus?text:root.agl
            horizontalAlignment: Text.AlignLeft
            anchors.fill: parent
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
            anchors.margins: UIConstants.rectRadius/2
            validator: RegExpValidator { regExp: validatorAltitude }
            onTextChanged: {
                if(focus){
                    if(text === "" || isNaN(text)){
                        root.agl = 0;
                    }else{
                        var newValue = parseFloat(text);
                        root.agl = newValue;
                    }
                }
            }
        }
        anchors.left: parent.left
        border.width: 1
    }
    Label {
        id: label2
        x: 55
        y: 144
        width: 72
        height: 25
        color: UIConstants.textColor
        text: qsTr("AMSL")
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        font.pixelSize: UIConstants.fontSize
        font.family: UIConstants.appFont
    }
    Rectangle {
        x: -5
        y: 173
        width: 158
        height: 25
        color: UIConstants.transparentColor
        radius: 1
        anchors.leftMargin: 12
        border.color: UIConstants.grayColor
        TextInput {
            id: txtAMSL
            color: UIConstants.textColor
            clip: true
            text: focus?text:Number(root.agl+root.asl).toFixed(2)
            horizontalAlignment: Text.AlignLeft
            anchors.fill: parent
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
            anchors.margins: UIConstants.rectRadius/2
            enabled: false
        }
        anchors.left: parent.left
        border.width: 1
    }



    Label {
        x: 43
        y: 88
        width: 14
        height: 25
        color: UIConstants.textColor
        text: degreeSymbol
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignTop
        font.pixelSize: UIConstants.fontSize
        font.family: UIConstants.appFont
    }

    Label {
        x: 88
        y: 88
        width: 14
        height: 25
        color: UIConstants.textColor
        text: qsTr("'")
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignTop
        font.pixelSize: UIConstants.fontSize
        font.family: UIConstants.appFont
    }

    Label {
        x: 132
        y: 88
        width: 14
        height: 25
        color: UIConstants.textColor
        text: qsTr("\"")
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignTop
        font.pixelSize: UIConstants.fontSize
        font.family: UIConstants.appFont
    }

    Label {
        x: 43
        y: 119
        width: 14
        height: 25
        color: UIConstants.textColor
        text: degreeSymbol
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignTop
        font.pixelSize: UIConstants.fontSize
        font.family: UIConstants.appFont
    }

    Label {
        x: 132
        y: 119
        width: 14
        height: 25
        color: UIConstants.textColor
        text: qsTr("'")
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignTop
        font.pixelSize: UIConstants.fontSize
        font.family: UIConstants.appFont
    }

    Label {
        x: 221
        y: 88
        width: 14
        height: 25
        color: UIConstants.textColor
        text: degreeSymbol
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignTop
        font.pixelSize: UIConstants.fontSize
        font.family: UIConstants.appFont
    }

    Label {
        x: 266
        y: 88
        width: 14
        height: 25
        color: UIConstants.textColor
        text: qsTr("'")
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignTop
        font.pixelSize: UIConstants.fontSize
        font.family: UIConstants.appFont
    }

    Label {
        x: 307
        y: 88
        width: 14
        height: 25
        color: UIConstants.textColor
        text: qsTr("\"")
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignTop
        font.pixelSize: UIConstants.fontSize
        font.family: UIConstants.appFont
    }

    Label {
        x: 221
        y: 119
        width: 14
        height: 25
        color: UIConstants.textColor
        text: degreeSymbol
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignTop
        font.pixelSize: UIConstants.fontSize
        font.family: UIConstants.appFont
    }

    Label {
        x: 307
        y: 119
        width: 14
        height: 25
        color: UIConstants.textColor
        text: qsTr("'")
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignTop
        font.pixelSize: UIConstants.fontSize
        font.family: UIConstants.appFont
    }

    Label {
        x: 362
        y: 0
        width: 105
        height: 25
        color: UIConstants.textColor
        text: qsTr("Waypoint Mode")
        horizontalAlignment: Text.AlignLeft
        verticalAlignment: Text.AlignVCenter
        font.pixelSize: UIConstants.fontSize
        font.family: UIConstants.appFont
    }

    Label {
        x: 8
        y: 0
        width: 80
        height: 25
        color: UIConstants.textColor
        text: qsTr("Coordinate")
        horizontalAlignment: Text.AlignLeft
        verticalAlignment: Text.AlignVCenter
        font.pixelSize: UIConstants.fontSize
        font.family: UIConstants.appFont
    }
    Label {
        x: 55
        y: 31
        width: 72
        height: 25
        text: qsTr("Latitude")
        verticalAlignment: Text.AlignVCenter
        horizontalAlignment: Text.AlignHCenter
        color: UIConstants.textColor
        font.pixelSize: UIConstants.fontSize
        font.family: UIConstants.appFont
    }

    Rectangle {
        y: 57
        width: 120
        height: 25
        anchors.left: parent.left
        anchors.leftMargin: 13
        color: UIConstants.transparentColor
        radius: 1
        border.color: UIConstants.grayColor
        border.width: 1
        TextInput {
            id: txtLat
            anchors.fill: parent
            anchors.margins: UIConstants.rectRadius/2
            text: focus?text:Number(root.latitude).toFixed(7)
            clip: true
            horizontalAlignment: Text.AlignLeft
            color: UIConstants.textColor
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
            validator: RegExpValidator { regExp: validatorLat }
            onTextChanged: {
                if(focus){
                    console.log("isNaN("+text+")"+isNaN(text));
                    if(text === "" || isNaN(text)){
                        root.latitude = 0;
                    }else{
                        var newValue = parseFloat(text);
                        root.latitude = newValue;
                    }
                }
            }
        }
    }

    Label {
        x: 232
        y: 31
        width: 72
        height: 25
        text: qsTr("Longitude")
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        color: UIConstants.textColor
        font.pixelSize: UIConstants.fontSize
        font.family: UIConstants.appFont
    }

    Rectangle {
        y: 57
        width: 120
        height: 25
        anchors.left: parent.left
        anchors.leftMargin: 191
        color: UIConstants.transparentColor
        border.color: UIConstants.grayColor
        border.width: 1
        radius: 1
        TextInput {
            id: txtLon
            anchors.fill: parent
            anchors.margins: UIConstants.rectRadius/2
            text: focus?text:Number(root.longitude).toFixed(7)
            clip: true
            horizontalAlignment: Text.AlignLeft
            color: UIConstants.textColor
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
            validator: RegExpValidator { regExp: validatorLon }
            onTextChanged: {
                if(focus){
                    if(text === "" || isNaN(text)){
                        root.longitude = 0;
                    }else{
                        var newValue = parseFloat(text);
                        root.longitude = newValue;
                    }
                }
            }
        }
    }
}

/*##^## Designer {
    D{i:4;anchors_x:8}D{i:21;anchors_width:130;anchors_x:362}
}
 ##^##*/
