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
    width: UIConstants.sRect * 14
    height: UIConstants.sRect * 15/2
    border.color: "gray"
    border.width: 1
    property real minValue: 85
    property real maxValue: 110
    property real currentValue: 100
    property real stepValue: 5
    property var validatorValue: /^([1-9][0-9]|[1-3][0-9][0-9]|[400])/
    signal confirmClicked()
    signal cancelClicked()
    MouseArea{
        anchors.fill: parent
        hoverEnabled: true
    }
    FlatButtonIcon{
        anchors.right: parent.right
        anchors.top: parent.top
//                anchors.rightMargin: 5
//                anchors.topMargin: 5
//                color: "white"
        width: 20
        height: 20
        isShowRect: false
        isAutoReturn: true
        isSolid: true
        iconColor: UIConstants.textColor
        iconSize: 15
        icon: UIConstants.iMouse

        onClicked: {
            cancelClicked();
            root.visible = false;
        }
    }

    Label {
        id: label
        x: 8
        y: 0
        width: 264
        height: 25
        text: qsTr("Speed editor")
        verticalAlignment: Text.AlignVCenter
        horizontalAlignment: Text.AlignHCenter
        color: UIConstants.textColor
        font.pixelSize: UIConstants.fontSize
        font.family: UIConstants.appFont
    }

    Rectangle {
        y: 75
        height: 25
        anchors.left: parent.left
        anchors.leftMargin: 81
        anchors.right: parent.right
        anchors.rightMargin: 81
        color: UIConstants.transparentColor
        border.color: UIConstants.grayColor
        border.width: 1
        radius: UIConstants.rectRadius
        TextInput {
            id: txtCurrentValue
            anchors.fill: parent
            anchors.margins: UIConstants.rectRadius/2
            clip: true
            horizontalAlignment: Text.AlignHCenter
            color: UIConstants.textColor
            text: focus?text:Number(root.currentValue).toFixed(0).toString()
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
            anchors.topMargin: 3
            validator: RegExpValidator { regExp: validatorValue }
            enabled: false
            onTextChanged: {
                if(focus){
                    root.currentValue = Number(text).toFixed(0);
                }
            }
        }
    }

    FlatButtonIcon{
        id: btnConfirm
        y: 192
        height: 30
        width: 60
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
            root.confirmClicked();
            root.visible = false;
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
            root.visible = false;
        }
    }

    Slider {
        id: sldBar
        x: 40
        y: 28
        anchors.horizontalCenter: parent.horizontalCenter
        value: pressed?value:(currentValue-minValue)/(maxValue-minValue)
        onValueChanged: {
            if(pressed){
                root.currentValue = value*(maxValue-minValue)+minValue;
                txtCurrentValue.text = Number(root.currentValue).toFixed(0).toString()
            }
        }
    }

    Label {
        id: lblMin
        y: 40
        width: 26
        height: 25
        color: UIConstants.textColor
        text: Number(root.minValue).toFixed(0).toString()
        anchors.left: parent.left
        anchors.leftMargin: 8
        horizontalAlignment: Text.AlignHCenter
        font.family: UIConstants.appFont
        font.pixelSize: UIConstants.fontSize
        verticalAlignment: Text.AlignVCenter
    }

    Label {
        id: lblMax
        x: 8
        y: 40
        width: 26
        height: 25
        color: UIConstants.textColor
        text: Number(root.maxValue).toFixed(0).toString()
        anchors.right: parent.right
        anchors.rightMargin: 8
        horizontalAlignment: Text.AlignHCenter
        font.family: UIConstants.appFont
        font.pixelSize: UIConstants.fontSize
        verticalAlignment: Text.AlignVCenter
    }
    FlatButtonIcon{
        id: btnMinus
        y: 66
        isSolid: true
        isShowRect: false
        isAutoReturn: true
        width: UIConstants.sRect*2
        height: UIConstants.sRect*2
        anchors.left: parent.left
        anchors.leftMargin: 8
        icon: UIConstants.iRemoveMarker
        onClicked: {
            if(root.currentValue - root.stepValue >= (root.stepValue+root.minValue)){
                root.currentValue -= root.stepValue;
            }
            else{
                root.currentValue = root.minValue;
            }
            txtCurrentValue.text = Number(root.currentValue).toFixed(0).toString()
        }
    }
    FlatButtonIcon{
        id: btnPlus
        x: 232
        y: 66
        isSolid: true
        isShowRect: false
        isAutoReturn: true
        width: UIConstants.sRect*2
        height: UIConstants.sRect*2
        radius: 1
        anchors.right: parent.right
        anchors.rightMargin: 8
        icon: UIConstants.iAddMarker
        onClicked: {
            if(root.currentValue + root.stepValue <= (root.maxValue - root.stepValue)){
                root.currentValue += root.stepValue;
            }
            else{
                root.currentValue = root.maxValue;
            }
            txtCurrentValue.text = Number(root.currentValue).toFixed(0).toString()
        }
    }
    Component.onCompleted: {
        txtCurrentValue.text = Number(root.currentValue).toFixed(0).toString()
    }
}

/*##^## Designer {
    D{i:10;anchors_x:11}
}
 ##^##*/
