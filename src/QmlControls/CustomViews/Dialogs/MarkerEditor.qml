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
import QtPositioning 5.8
import QtQuick.Layouts 1.3
//---------------- Include custom libs ----------------------------------------
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0

//----------------------- Component definition- ------------------------------
Rectangle {
    id: root
    //---------- properties
    color: UIConstants.transparentBlue
    height: UIConstants.sRect * 11
    width: UIConstants.sRect * 10
    radius: UIConstants.rectRadius
    border.color: "gray"
    border.width: 1
    property real latitude: 0
    property real longitude: 0
    property real agl: 0
    property var validatorAltitude: /^([0-9]|[1-9][0-9]|[1-9][0-9][0-9]|[1-9][0-9][0-9][0-9])/
    property var validatorLat: /^([-]|)([0-9]|[1-8][0-9])(\.)([0-9][0-9][0-9][0-9][0-9][0-9][0-9])/
    property var validatorLon: /^([-]|)([0-9]|[1-9][0-9]|[1][0-7][0-9])(\.)([0-9][0-9][0-9][0-9][0-9][0-9][0-9])/
    property var markerSymbolLink: {
        "MARKER_DEFAULT":"qrc:/qmlimages/markers/FlagIcon.png",
        "MARKER_TANK":"qrc:/qmlimages/markers/TankIcon.png",
        "MARKER_PLANE":"qrc:/qmlimages/markers/PlaneIcon.png",
        "MARKER_SHIP":"qrc:/qmlimages/markers/BattleShip.png",
        "MARKER_TARGET":"qrc:/qmlimages/markers/TargetIcon.png"
    }
    property var lstTxt: [
        txtLat,txtLon,txtAlt]
    property int currentIndex: markerSelector.currentIndex
    signal markerIDChanged(var markerType,var iconSource)
    signal confirmClicked(var currentIndex, var coordinate)
    signal cancelClicked()
    signal textChanged(string newText)
    function changeAllFocus(enable){
        for(var i =0; i < lstTxt.length; i++){
            if(lstTxt[i].focus !== enable)
                lstTxt[i].focus = enable;
        }
    }
    function changeCoordinate(_coordinate){
        changeAllFocus(false);
        txtLat.text = Number(_coordinate.latitude).toFixed(7).toString();
        txtLon.text = Number(_coordinate.longitude).toFixed(7).toString();
        txtAlt.text = Number(_coordinate.altitude).toFixed(2).toString();
    }

    function loadInfo(_coordinate,_type,text){
        changeCoordinate(_coordinate);
        if(_type === "MARKER_DEFAULT"){
            markerSelector.currentIndex = 0;
        }else if(_type === "MARKER_TANK"){
            markerSelector.currentIndex = 1;
        }else if(_type === "MARKER_PLANE"){
            markerSelector.currentIndex = 2;
        }else if(_type === "MARKER_TARGET"){
            markerSelector.currentIndex = 3;
        }else if(_type === "MARKER_SHIP"){
            markerSelector.currentIndex = 4;
        }
        txtMarkerText.text = text;
    }
    onVisibleChanged: {
        if(!visible){
            root.changeAllFocus(false);
        }
    }

    Label {
        id: lblTitle
        text: qsTr("Marker Editor")
        anchors.top: parent.top
        anchors.topMargin: 8
        anchors.horizontalCenter: parent.horizontalCenter
        verticalAlignment: Text.AlignVCenter
        horizontalAlignment: Text.AlignHCenter
        color: UIConstants.textColor
        font.pixelSize: UIConstants.fontSize
        font.family: UIConstants.appFont
    }

    GridView{
        id: markerSelector
        height: UIConstants.sRect * 2 + 4
        anchors.topMargin: 8
        clip: true
        anchors.rightMargin: 8
        anchors.leftMargin: 8
        anchors.right: parent.right
        anchors.left: parent.left
        anchors.top: lblTitle.bottom
        cellWidth: UIConstants.sRect*2
        cellHeight: UIConstants.sRect*2
        highlight: Rectangle {
            width: markerSelector.cellWidth - 2
            height: markerSelector.cellHeight - 2
            color: "lightsteelblue";
            radius: 5
        }
        Component {
            id: contactsDelegate
            Item {
                id: wrapper
                width: markerSelector.cellWidth-3
                height: markerSelector.cellHeight-3
                property int markerSource: markerID
                property string type: markerType
                Image{
                    anchors.horizontalCenter: parent.horizontalCenter
                    anchors.verticalCenter: parent.verticalCenter
                    source: iconSource + ".png"
                }
                MouseArea{
                    anchors.fill: parent
                    onClicked: {
                        markerSelector.currentIndex = wrapper.markerSource;
                        markerIDChanged(markerType,iconSource);
                    }
                }
            }
        }
        model: ListModel {
            ListElement { markerType:"MARKER_DEFAULT";iconSource: "qrc:/qmlimages/markers/FlagIcon";markerID:0}
            ListElement { markerType:"MARKER_TANK";iconSource: "qrc:/qmlimages/markers/TankIcon";markerID:1}
            ListElement { markerType:"MARKER_PLANE";iconSource: "qrc:/qmlimages/markers/PlaneIcon";markerID:2}
            ListElement { markerType:"MARKER_TARGET";iconSource: "qrc:/qmlimages/markers/TargetIcon";markerID:3}
            ListElement { markerType:"MARKER_SHIP";iconSource: "qrc:/qmlimages/markers/BattleShip";markerID:4}
        }
        delegate: contactsDelegate
    }

    ColumnLayout{
        id: layoutGPS
        height: UIConstants.sRect * 3
        anchors.left: parent.left
        anchors.leftMargin: 8
        anchors.right: parent.right
        anchors.rightMargin: 8
        anchors.top: rectMarkerText.bottom
        anchors.topMargin: 8
        spacing: 8
        RowLayout{
            spacing: 0
            Layout.preferredHeight: UIConstants.sRect
            Layout.preferredWidth: parent.width
            Label {
                id: lblLat
                Layout.preferredWidth: UIConstants.sRect * 4
                Layout.preferredHeight: UIConstants.sRect
                text: qsTr("Latitude:")
                verticalAlignment: Text.AlignVCenter
                horizontalAlignment: Text.AlignLeft
                color: UIConstants.textColor
                font.pixelSize: UIConstants.fontSize
                font.family: UIConstants.appFont
            }

            Rectangle {
                Layout.preferredHeight: UIConstants.sRect
                Layout.preferredWidth: parent.width - UIConstants.sRect * 4
                color: UIConstants.transparentColor
                border.color: UIConstants.grayColor
                border.width: 1
                radius: UIConstants.rectRadius
                TextInput {
                    id: txtLat
                    anchors.margins: UIConstants.rectRadius/2
                    text: focus?text:Number(root.latitude).toFixed(7)
                    anchors.fill: parent
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
        }
        RowLayout{
            Layout.preferredHeight: UIConstants.sRect
            Layout.preferredWidth: parent.width
            spacing: 0
            Label {
                id: lblLon
                Layout.preferredWidth: UIConstants.sRect * 4
                Layout.preferredHeight: UIConstants.sRect
                text: qsTr("Longitude:")
                verticalAlignment: Text.AlignVCenter
                color: UIConstants.textColor
                font.pixelSize: UIConstants.fontSize
                font.family: UIConstants.appFont
            }

            Rectangle {
                Layout.preferredHeight: UIConstants.sRect
                Layout.preferredWidth: parent.width - UIConstants.sRect * 4
                color: UIConstants.transparentColor
                border.color: UIConstants.grayColor
                border.width: 1
                radius: UIConstants.rectRadius
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
        RowLayout{
            Layout.preferredHeight: UIConstants.sRect
            Layout.preferredWidth: parent.width
            spacing: 0
            Label {
                id: lblAlt
                Layout.preferredWidth: UIConstants.sRect * 4
                Layout.preferredHeight: UIConstants.sRect
                color: UIConstants.textColor
                text: qsTr("Altitude:")
                verticalAlignment: Text.AlignVCenter
                font.pixelSize: UIConstants.fontSize
                font.family: UIConstants.appFont
            }

            Rectangle {
                Layout.preferredHeight: UIConstants.sRect
                Layout.preferredWidth: parent.width - UIConstants.sRect * 4
                color: UIConstants.transparentColor
                border.color: UIConstants.grayColor
                border.width: 1
                radius: UIConstants.rectRadius
                TextInput {
                    id: txtAlt
                    anchors.fill: parent
                    anchors.margins: UIConstants.rectRadius/2
                    text: focus?text:root.agl
                    clip: true
                    horizontalAlignment: Text.AlignLeft
                    color: UIConstants.textColor
                    font.pixelSize: UIConstants.fontSize
                    font.family: UIConstants.appFont
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
            }
        }
    }

    FlatButtonIcon{
        id: btnConfirm
        height: UIConstants.sRect * 2
        width: UIConstants.sRect * 4
        icon: UIConstants.iChecked
        isSolid: true
        color: "green"
        anchors.left: parent.left
        anchors.leftMargin: 8
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 8
        isAutoReturn: true
        radius: root.radius
        onClicked: {
            var coordinate = QtPositioning.coordinate(
                        Number(txtLat.text).toFixed(7),
                        Number(txtLon.text).toFixed(7),
                        Number(txtAlt.text).toFixed(2)
                        )
            root.confirmClicked(markerSelector.currentIndex,coordinate);
            root.changeAllFocus(false);
        }
    }
    FlatButtonIcon{
        id: btnCancel
        x: 102
        y: 192
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
    Rectangle {
        id: rectMarkerText
        height: UIConstants.sRect
        color: UIConstants.transparentColor
        border.color: UIConstants.grayColor
        border.width: 1
        radius: UIConstants.rectRadius
        anchors.top: markerSelector.bottom
        anchors.topMargin: 8
        anchors.right: parent.right
        anchors.rightMargin: 8
        anchors.left: parent.left
        anchors.leftMargin: 8
        TextEdit {
            id: txtMarkerText
            anchors.fill: parent
            anchors.margins: UIConstants.rectRadius/2
            color: UIConstants.textColor
            horizontalAlignment: Text.AlignRight
            clip: true
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
            onTextChanged: {
                root.textChanged(text)
            }
        }
    }
}
