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
    height: UIConstants.sRect * 14
    width: UIConstants.sRect * 13
    radius: UIConstants.rectRadius
    border.color: "gray"
    border.width: 1
    property real latitude: 0
    property real longitude: 0
    property real asl: 0
    property real agl: 0
    property var validatorAltitude: /^([0-9]|[1-9][0-9]|[1-9][0-9][0-9]|[1-9][0-9][0-9][0-9])/
    property var validatorLat: /^([-]|)([0-9]|[1-8][0-9])(\.)([0-9][0-9][0-9][0-9][0-9][0-9][0-9])/
    property var validatorLon: /^([-]|)([0-9]|[1-9][0-9]|[1][0-7][0-9])(\.)([0-9][0-9][0-9][0-9][0-9][0-9][0-9])/
    property var validatorLatDecimal: /^([-]|)([0-9]|[1-8][0-9])/
    property var validatorLonDecimal: /^([-]|)([0-9]|[1-9][0-9]|[1][0-7][0-9])/
    property var markerSymbolLink: {
        "MARKER_DEFAULT":"qrc:/qmlimages/markers/FlagIcon.png",
        "MARKER_TANK":"qrc:/qmlimages/markers/TankIcon.png",
        "MARKER_PLANE":"qrc:/qmlimages/markers/PlaneIcon.png",
        "MARKER_SHIP":"qrc:/qmlimages/markers/BattleShip.png",
        "MARKER_TARGET":"qrc:/qmlimages/markers/TargetIcon.png"
    }
    property var lstTxt: [
        txtAGL,txtAMSL]
    property var cdeTxt: [
        cdeLat,cdeLon]
    property int currentIndex: markerSelector.currentIndex
    signal markerIDChanged(var markerType,var iconSource)
    signal confirmClicked(var currentIndex, var coordinate)
    signal cancelClicked()
    signal textChanged(string newText)
    function changeAllFocus(enable){
        for(var i =0; i < cdeTxt.length; i++){
            if(cdeTxt[i].focus !== enable)
                cdeTxt[i].focus = enable;
        }
        for(var i =0; i < lstTxt.length; i++){
            if(lstTxt[i].editting !== enable)
                lstTxt[i].editting = enable;
        }
    }
    function changeCoordinate(_coordinate){
        changeAllFocus(false);
        root.latitude = _coordinate.latitude;
        root.longitude = _coordinate.longitude;
        root.agl = _coordinate.altitude;
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
    MouseArea{
        anchors.fill: parent
        hoverEnabled: true
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
        height: UIConstants.sRect * 2
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
            width: markerSelector.cellWidth
            height: markerSelector.cellHeight
            color: "lightsteelblue";
            border.color: UIConstants.transparentColor
            border.width: 4
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

    QTextInput {
        id: txtMarkerText
        height: UIConstants.sRect
        horizontalAlignment: Text.AlignRight
        anchors.top: markerSelector.bottom
        anchors.topMargin: 8
        anchors.right: parent.right
        anchors.rightMargin: 8
        anchors.left: parent.left
        anchors.leftMargin: 8
        onTextChanged: {
            root.textChanged(text)
        }
    }

    ColumnLayout{
        id: layoutGPS
        width: UIConstants.sRect * 12
        height: UIConstants.sRect * 7
        anchors.left: parent.left
        anchors.leftMargin: 8
        anchors.right: parent.right
        anchors.rightMargin: 8
        anchors.top: txtMarkerText.bottom
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
                onValueChanged: {
                    root.latitude = value;
                }
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
                onValueChanged: {
                    root.longitude = value;
                }
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
                text: Number(root.agl+root.asl).toFixed(2)
                title: "AMSL"
            }
            QLabeledTextInput {
                id: txtAGL
                width: UIConstants.sRect * 6
                Layout.fillHeight: true
                text: editting?text:root.agl
                title: "AGL"
                validator: RegExpValidator { regExp: validatorAltitude }
                onTextChanged: {
                    if(editting){
                        if(text != "" && !isNaN(text)){
                            var newValue = parseFloat(text);
                            root.agl = newValue;
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
                        Number(cdeLat.value).toFixed(7),
                        Number(cdeLon.value).toFixed(7),
                        Number(txtAGL.text).toFixed(2)
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
}
