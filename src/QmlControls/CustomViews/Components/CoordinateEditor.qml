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
import QtQuick.Layouts 1.3
import QtQuick.Controls 2.1

//---------------- Include custom libs ----------------------------------------
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0

Item {
    id: root
    width: UIConstants.sRect * 6
    height: UIConstants.sRect * 4.5
    property var arrayDirLabel: ["E","W"]
    property var directionLabel: value > 0?arrayDirLabel[0]:arrayDirLabel[1]
    property string title: "Coordinate"
    property var validatorValue: /^([-]|)([0-9]|[1-8][0-9])(\.)([0-9][0-9][0-9][0-9][0-9][0-9][0-9])/
    property var validatorValueDecimal: /^([-]|)([0-9]|[1-9][0-9]|[1][0-7][0-9])/
    property var validatorValueMinute: /^([0-9]|[1-5][0-9])/
    property var validatorValueSecond: /^([0-9]|[1-5][0-9])/
    property var validatorValueMinuteFloat: /^([0-9]|[1-5][0-9])(\.)([0-9][0-9][0-9])/
    property real value: 0
    property bool isEdittingMDS: false
    property var lstTxt: [
        txtValue,txtValueD1,txtValueM1,txtValueS1,txtValueD2,txtValueM2]
    function changeAllFocus(enable){
//        console.log("Change all focus to "+enable +" "+value+" on "+directionLabel);
        isEdittingMDS = enable;
        for(var i =0; i < lstTxt.length; i++){
            if(lstTxt[i].focus !== enable)
                lstTxt[i].focus = enable;
        }

    }
    function updateValueDMS(type){
        var lastValue = value;
        if(type === "D" &&
                (txtValueM1.text === "" || txtValueS1.text === "")){
            isEdittingMDS = false;
            value = lastValue;
            isEdittingMDS = true;
        }
        if(type === "D" &&
                (txtValueD1.text === "" || txtValueS1.text === "")){
            isEdittingMDS = false;
            value = lastValue;
            isEdittingMDS = true;
        }
        if(type === "D" &&
                (txtValueM1.text === "" || txtValueD1.text === "")){
            isEdittingMDS = false;
            value = lastValue;
            isEdittingMDS = true;
        }

//        if(txtValueD1.text === ""){
//            value = lastValue;
//        }
//        if(txtValueM1.text === ""){
//            value = lastValue;
//        }
//        if(txtValueS1.text === ""){
//            value = lastValue;
//        }
        if(txtValueD1.text !== "" &&
           txtValueM1.text !== "" &&
           txtValueS1.text !== ""){
            value = (directionLabel === arrayDirLabel[0]?1:-1 )* Number(
                parseFloat(txtValueD1.text) +
                parseFloat(txtValueM1.text)/60 +
                parseFloat(txtValueS1.text)/3600).toFixed(7);
        }
    }
    function updateValueDM(type){
        var lastValue = value;
        if(txtValueD2.text === ""){
            value = lastValue;
        }
        if(txtValueM2.text === ""){
            value = lastValue;
        }
        if(txtValueD2.text !== "" &&
           txtValueM2.text !== ""){
            value = (directionLabel === arrayDirLabel[0]?1:-1 )* Number(
                parseFloat(txtValueD2.text) +
                parseFloat(txtValueM2.text)/60).toFixed(7);
        }
    }
    function updateValue(){
        value = (directionLabel === arrayDirLabel[0]?1:-1 )* Number(txtValue.text).toFixed(7);
    }

    function ddToDMS(deg){
        var convertDeg = Math.abs(deg)
        var d = Math.floor(convertDeg);
        var minfloat = (convertDeg-d)*60;
        var m = Math.floor(minfloat);
        var secfloat = (minfloat-m)*60;
        var s = Math.round(secfloat);
        // After rounding, the seconds might become 60. These two
        // if-tests are not necessary if no rounding is done.
        if (s==60) {
         m++;
         s=0;
        }
        if (m==60) {
         d++;
         m=0;
        }
        return {"D":Math.abs(d),"M":m,"S":s,"MS":Number(m+s/60).toFixed(3),"DMS":Number(convertDeg).toFixed(7)}
    }

    Label{
        id: lblTitle
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        anchors.top: parent.top
        anchors.topMargin: 0
        anchors.horizontalCenter: parent.horizontalCenter
        height: UIConstants.sRect
        text: root.title
        color: UIConstants.textColor
        font.family: UIConstants.appFont
        font.pixelSize: UIConstants.fontSize
    }

    ColumnLayout{
        anchors.top: lblTitle.bottom
        anchors.right: parent.right
        anchors.left: parent.left
        anchors.bottom: parent.bottom
        RowLayout{
            Layout.alignment: Qt.AlignRight | Qt.AlignVCenter
            Layout.preferredHeight: UIConstants.sRect
            layoutDirection: Qt.RightToLeft
            QLabel{
                Layout.preferredWidth: UIConstants.sRect
                Layout.preferredHeight: UIConstants.sRect
                text: directionLabel
                MouseArea{
                    anchors.fill: parent
                    hoverEnabled: true
                    onClicked: {
                        root.value = -root.value;
                    }
                }
            }
            Label{
                Layout.preferredWidth: UIConstants.sRect/4
                Layout.preferredHeight: UIConstants.sRect/4
                color: UIConstants.textColor
                text: UIConstants.degreeSymbol
                verticalAlignment: Text.AlignTop
                horizontalAlignment: Text.AlignHCenter
                Layout.alignment: Qt.AlignLeft | Qt.AlignTop
                font.family: UIConstants.appFont
                font.pixelSize: UIConstants.fontSize
            }
            QTextInput{
                id: txtValue
                Layout.fillWidth: true
                Layout.preferredHeight: UIConstants.sRect
                validator: RegExpValidator { regExp: root.validatorValue }
                text: focus?text:ddToDMS(root.value)["DMS"]
                onTextChanged: {
                    if(focus){
                        isEdittingMDS = false;
                        if(text != "" && !isNaN(text)){
                            updateValue()
                        }
                    }
                }
            }
        }
        RowLayout{
            Layout.alignment: Qt.AlignRight | Qt.AlignVCenter
            Layout.preferredHeight: UIConstants.sRect
            layoutDirection: Qt.RightToLeft
            QLabel{
                Layout.preferredWidth: UIConstants.sRect
                Layout.preferredHeight: UIConstants.sRect
                text: directionLabel
                MouseArea{
                    anchors.fill: parent
                    hoverEnabled: true
                    onClicked: {
                        root.value = -root.value;
                    }
                }
            }
            Label{
                Layout.preferredWidth: UIConstants.sRect/4
                Layout.preferredHeight: UIConstants.sRect/4
                color: UIConstants.textColor
                text: "\""
                Layout.alignment: Qt.AlignLeft | Qt.AlignTop
                verticalAlignment: Text.AlignTop
                horizontalAlignment: Text.AlignHCenter
                font.family: UIConstants.appFont
                font.pixelSize: UIConstants.fontSize
            }
            QTextInput{
                id: txtValueS1
                Layout.preferredWidth:  UIConstants.sRect
                Layout.preferredHeight: UIConstants.sRect
                validator: RegExpValidator { regExp: root.validatorValueSecond }
                text: (focus || isEdittingMDS) ?text:ddToDMS(root.value)["S"]
                onTextChanged: {
                    if(focus){
                        isEdittingMDS = true;
//                        console.log("txtValueS1 isEdittingMDS = "+isEdittingMDS);
                        if(text != "" && !isNaN(text)){
                            updateValueDMS("S")
                        }
                    }
                }
            }
            Label{
                Layout.preferredWidth: UIConstants.sRect/4
                Layout.preferredHeight: UIConstants.sRect/4
                color: UIConstants.textColor
                text: "'"
                verticalAlignment: Text.AlignTop
                horizontalAlignment: Text.AlignHCenter
                Layout.alignment: Qt.AlignLeft | Qt.AlignTop
                font.family: UIConstants.appFont
                font.pixelSize: UIConstants.fontSize
            }
            QTextInput{
                id: txtValueM1
                Layout.preferredWidth:  UIConstants.sRect
                Layout.preferredHeight: UIConstants.sRect
                validator: RegExpValidator { regExp: root.validatorValueMinute }
                text: (focus || isEdittingMDS) ?text:ddToDMS(root.value)["M"]
                onTextChanged: {
                    if(focus){
                        isEdittingMDS = true;
//                        console.log("txtValueM1 isEdittingMDS = "+isEdittingMDS);
                        if(text != "" && !isNaN(text)){                            
                            updateValueDMS("M")
                        }
                    }
                }
            }
            Label{
                Layout.preferredWidth: UIConstants.sRect/4
                Layout.preferredHeight: UIConstants.sRect/4
                color: UIConstants.textColor
                text: UIConstants.degreeSymbol
                Layout.alignment: Qt.AlignLeft | Qt.AlignTop
                horizontalAlignment: Text.AlignHCenter
                font.family: UIConstants.appFont
                font.pixelSize: UIConstants.fontSize
            }
            QTextInput{
                id: txtValueD1
                Layout.fillWidth: true
                Layout.preferredHeight: UIConstants.sRect
                validator: RegExpValidator { regExp: root.validatorValueDecimal }
                text: focus?text:ddToDMS(root.value)["D"]
                onTextChanged: {
                    if(focus){
                        if(text != "" && !isNaN(text)){
                            updateValueDMS("D")
                        }
                    }
                }
            }
        }
        RowLayout{
            Layout.preferredWidth: parent.width
            Layout.preferredHeight: UIConstants.sRect
            layoutDirection: Qt.RightToLeft
            QLabel{
                Layout.preferredWidth: UIConstants.sRect
                Layout.preferredHeight: UIConstants.sRect
                text: directionLabel
                MouseArea{
                    anchors.fill: parent
                    hoverEnabled: true
                    onClicked: {
                        root.value = -root.value;
                    }
                }
            }
            Label{
                Layout.preferredWidth: UIConstants.sRect/4
                Layout.preferredHeight: UIConstants.sRect/4
                color: UIConstants.textColor
                text: "'"
                verticalAlignment: Text.AlignTop
                horizontalAlignment: Text.AlignHCenter
                Layout.alignment: Qt.AlignLeft | Qt.AlignTop
                font.family: UIConstants.appFont
                font.pixelSize: UIConstants.fontSize
            }
            QTextInput{
                id: txtValueM2
                Layout.preferredWidth:  UIConstants.sRect*2 + UIConstants.sRect/4 + parent.spacing *2
                Layout.preferredHeight: UIConstants.sRect
                validator: RegExpValidator { regExp: root.validatorValueMinuteFloat }
                text: focus?text:ddToDMS(root.value)["MS"];
                onTextChanged: {
                    if(focus){
                        isEdittingMDS = false;
                        if(text != "" && !isNaN(text)){
                            updateValueDM("MS")
                        }
                    }
                }
            }
            Label{
                Layout.preferredWidth: UIConstants.sRect/4
                Layout.preferredHeight: UIConstants.sRect/4
                color: UIConstants.textColor
                text: UIConstants.degreeSymbol
                Layout.alignment: Qt.AlignLeft | Qt.AlignTop
                horizontalAlignment: Text.AlignHCenter
                font.family: UIConstants.appFont
                font.pixelSize: UIConstants.fontSize
            }
            QTextInput{
                id: txtValueD2
                Layout.fillWidth: true
                Layout.preferredHeight: UIConstants.sRect
                validator: RegExpValidator { regExp: root.validatorValueDecimal }
                text: focus?text:ddToDMS(root.value)["D"]
                onTextChanged: {
                    if(focus){
                        if(text != "" && !isNaN(text)){
                            isEdittingMDS = false;
                            updateValueDM("D")
                        }
                    }
                }
            }
        }
    }
}
