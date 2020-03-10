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
import QtQuick.Layouts 1.3
//---------------- Include custom libs ----------------------------------------
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0

//----------------------- Component definition- ------------------------------
Rectangle {
    id: root
    //---------- properties
    color: UIConstants.transparentBlue
    radius: UIConstants.rectRadius
    width: UIConstants.sRect * 19
    height: UIConstants.sRect * 14.5
    border.color: "gray"
    border.width: 1
    property alias waypointModeEnabled: rectWaypointMode.enabled
    property string degreeSymbol : "\u00B0"
    property var lstTxt: [
        cdeLat,cdeLon,txtAGL,
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
    Label{
        id: lblTitle
        verticalAlignment: Text.AlignVCenter
        horizontalAlignment: Text.AlignLeft
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.topMargin: 8
        anchors.leftMargin: 8
        color: UIConstants.textColor
        text: "Waypoint Editor"
        font.family: UIConstants.appFont
        font.pixelSize: UIConstants.fontSize
    }

    ColumnLayout {
        id: rectParam
        width: UIConstants.sRect * 12
        height: UIConstants.sRect * 11
        anchors.left: parent.left
        anchors.leftMargin: 8
        anchors.top: lblTitle.bottom
        anchors.topMargin: 8
        spacing: 8
        RowLayout{
            id: rlCoordinate
            Layout.preferredWidth: parent.width
            Layout.preferredHeight: UIConstants.sRect * 4
            spacing: 16
            CoordinateEditor{
                id: cdeLat
                Layout.fillHeight: true
                width: UIConstants.sRect * 6
                title: "Latitude"
                directionLabel: "E"
                value: root.latitude
                validatorValue: root.validatorLat
                validatorValueDecimal: root.validatorLatDecimal
            }
            CoordinateEditor{
                id: cdeLon
                Layout.fillHeight: true
                width: UIConstants.sRect * 6
                title: "Longitude"
                directionLabel: "N"
                value: root.longitude
                validatorValue: root.validatorLon
                validatorValueDecimal: root.validatorLonDecimal
            }
        }
        RowLayout {
            id: rlAltitude
            Layout.preferredWidth: parent.width
            Layout.preferredHeight: UIConstants.sRect * 2
            spacing: 16
            QLabeledTextInput {
                id: txtAMSL
                width: UIConstants.sRect * 6
                Layout.fillHeight: true
                enabled: false
                text: focus?text:Number(root.agl+root.asl).toFixed(2)
                title: "AMSL"
            }
            QLabeledTextInput {
                id: txtAGL
                width: UIConstants.sRect * 6
                Layout.fillHeight: true
                text: focus?text:root.agl
                title: "AGL"
                validator: RegExpValidator { regExp: validatorAltitude }
                onTextChanged: {
                    if(focus){
                        if(text != "" && isNaN(text)){
                            var newValue = parseFloat(text);
                            root.agl = newValue;
                        }
                    }
                }
            }

        }

        RowLayout {
            id: rlParams
            Layout.preferredWidth: parent.width
            Layout.preferredHeight: UIConstants.sRect * 4
            spacing: 16
            ColumnLayout{
                spacing: 0
                width: UIConstants.sRect * 6
                Layout.fillHeight: true
                QLabeledTextInput {
                    id: txtParam1
                    width: parent.width
                    Layout.preferredWidth: parent.width
                    Layout.preferredHeight: UIConstants.sRect * 2
                    validator: RegExpValidator { regExp: root.validatorParam }
                    text: focus?text:parseInt(root.param1)
                    title: lstWaypointCommand[vehicleType][waypointModes[lstWaypointMode.currentIndex]]["PARAM1"]["LABEL"]
                    enabled: lstWaypointCommand[vehicleType][waypointModes[lstWaypointMode.currentIndex]]["PARAM1"]["EDITABLE"]
                    onTextChanged: {
                        if(focus){
                            if(text != "" && isNaN(text)){
                                root.param1 = parseInt(text);
                            }
                        }
                    }
                }
                QLabeledTextInput {
                    id: txtParam3
                    width: parent.width
                    Layout.preferredWidth: parent.width
                    Layout.preferredHeight: UIConstants.sRect * 2
                    validator: RegExpValidator { regExp: root.validatorParam }
                    text: focus?text:parseInt(root.param3)
                    title: lstWaypointCommand[vehicleType][waypointModes[lstWaypointMode.currentIndex]]["PARAM3"]["LABEL"]
                    enabled: lstWaypointCommand[vehicleType][waypointModes[lstWaypointMode.currentIndex]]["PARAM3"]["EDITABLE"]
                    onTextChanged: {
                        if(focus){
                            if(text != "" && isNaN(text)){
                                root.param3 = parseInt(text);
                            }
                        }
                    }
                }
            }
            ColumnLayout{
                spacing: 0
                width: UIConstants.sRect * 6
                Layout.fillHeight: true
                QLabeledTextInput {
                    id: txtParam2
                    width: parent.width
                    Layout.preferredWidth: parent.width
                    Layout.preferredHeight: UIConstants.sRect * 2
                    validator: RegExpValidator { regExp: root.validatorParam }
                    text: focus?text:parseInt(root.param2)
                    title: lstWaypointCommand[vehicleType][waypointModes[lstWaypointMode.currentIndex]]["PARAM2"]["LABEL"]
                    enabled: lstWaypointCommand[vehicleType][waypointModes[lstWaypointMode.currentIndex]]["PARAM2"]["EDITABLE"]
                    onTextChanged: {
                        if(focus){
                            if(text != "" && isNaN(text)){
                                root.param2 = parseInt(text);
                            }
                        }
                    }
                }
                QLabeledTextInput {
                    id: txtParam4
                    width: parent.width
                    Layout.preferredWidth: parent.width
                    Layout.preferredHeight: UIConstants.sRect * 2
                    validator: RegExpValidator { regExp: root.validatorParam }
                    text: focus?text:parseInt(root.param4)
                    title: lstWaypointCommand[vehicleType][waypointModes[lstWaypointMode.currentIndex]]["PARAM4"]["LABEL"]
                    enabled: lstWaypointCommand[vehicleType][waypointModes[lstWaypointMode.currentIndex]]["PARAM4"]["EDITABLE"]
                    onTextChanged: {
                        if(focus){
                            if(text != "" && isNaN(text)){
                                root.param4 = parseInt(text);
                            }
                        }
                    }
                }
            }


        }
    }

    Rectangle {
        id: rectWaypointMode
        height: rectParam.height
        color: UIConstants.transparentColor
        border.color: UIConstants.grayColor
        radius: UIConstants.rectRadius
        anchors.left: rectParam.right
        anchors.leftMargin: UIConstants.sRect
        anchors.right: parent.right
        anchors.rightMargin: 8
        anchors.top: lblTitle.bottom
        anchors.topMargin: 8
        SubNav {
            id: lstWaypointMode
            anchors.rightMargin: 2
            anchors.leftMargin: 2
            anchors.bottomMargin: 2
            anchors.topMargin: 2
            anchors.fill: parent
            model: root.waypointModes
            onListViewClicked: {
                if(choosedItem === "LOITER"){
                    param1 = 3600;
                    param3 = 1;
                }else {
                    param1 = 0;
                    param3 = 0;
                }

                waypointModeChanged(choosedItem,
                    param1,param2,param3,param4)

            }
        }
    }
    FlatButtonIcon{
        id: btnConfirm
        height: UIConstants.sRect * 2
        width: UIConstants.sRect * 4
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
        height: UIConstants.sRect * 2
        width: UIConstants.sRect * 4
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
}
