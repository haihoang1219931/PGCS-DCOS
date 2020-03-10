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
    property string directionLabel: "E"
    property string title: "Coordinate"
    property var validatorValue: /^([-]|)([0-9]|[1-8][0-9])(\.)([0-9][0-9][0-9][0-9][0-9][0-9][0-9])/
    property var validatorValueDecimal: /^([-]|)([0-9]|[1-9][0-9]|[1][0-7][0-9])/
    property var validatorValueMinute: /^([0-9]|[1-5][0-9])/
    property var validatorValueSecond: /^([0-9]|[1-5][0-9])/
    property var validatorValueMinuteFloat: /^([0-9]|[1-5][0-9])(\.)([0-9][0-9][0-9])/
    property real value: 0
    property var lstTxt: [
        txtValue,txtValueD1,txtValueM1,txtValueS1,txtValueD2,txtValueM2]
    onFocusChanged: {
        changeAllFocus(focus);
    }
    function changeAllFocus(enable){
        for(var i =0; i < lstTxt.length; i++){
            if(lstTxt[i].focus !== enable)
                lstTxt[i].focus = enable;
        }
    }
    function updateValueDMS(){
        value = Number(
            parseFloat(txtValueD1.text) +
            parseFloat(txtValueM1.text)/60 +
            parseFloat(txtValueS1.text)/3600).toFixed(7);
    }
    function updateValueDM(){
        value = Number(
            parseFloat(txtValueD2.text) +
            parseFloat(txtValueM2.text)/60).toFixed(7);
    }
    function updateValue(){
        value = Number(txtValue.text).toFixed(7);
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
                text: focus?text:Number(root.value).toFixed(7)
                onTextChanged: {
                    if(focus){
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
                text: focus?text:parseInt(
                                      (Math.abs(
                                           (root.value-parseInt(root.value))*60)-Math.floor(
                                                  Math.abs(
                                                      (root.value-parseInt(root.value))*60)
                                                  )
                                              )*60
                                        )
                onTextChanged: {
                    if(focus){
                        if(text != "" && !isNaN(text)){
                            updateValueDMS()
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
                text: focus?text:Math.round(Math.floor(
                                             Math.abs(
                                                 (root.value-parseInt(root.value))*60)
                                             )
                                         )
                onTextChanged: {
                    if(focus){
                        if(text != "" && !isNaN(text)){
                            updateValueDMS()
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
                text: focus?text:parseInt(root.value)
                onTextChanged: {
                    if(focus){
                        if(text != "" && !isNaN(text)){
                            updateValueDMS()
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
                text: focus?text:Number((root.value-parseInt(root.value))*60).toFixed(3);
                onTextChanged: {
                    if(focus){
                        if(text != "" && !isNaN(text)){
                            updateValueDM()
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
                text: focus?text:parseInt(root.value)
                onTextChanged: {
                    if(focus){
                        if(text != "" && !isNaN(text)){
                            updateValueDM()
                        }
                    }
                }
            }
        }
    }
}
