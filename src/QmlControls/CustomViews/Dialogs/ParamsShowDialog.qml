import QtQuick 2.3
import QtQuick.Controls 1.2
import QtQuick.Dialogs 1.2
import QtQuick.Layouts 1.1
import QtQuick.Window 2.0

//---------------- Include custom libs ----------------------------------------
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0

Rectangle {
    id: root
    clip: true
    color: UIConstants.transparentBlue
    radius: UIConstants.rectRadius
    border.color: "gray"
    border.width: 1
    property bool showContent: true
    property var vehicle
    property alias title: txtDialog.text
    property color fontColor: UIConstants.textColor
    property int fontSize: UIConstants.fontSize
    signal clicked(string type,string func)
    signal died()
    width: UIConstants.sRect * 12.5
    height: UIConstants.sRect * 3/2 + listView.height
    function setFocus(enable){
        rectangle.focus = enable
    }
    PropertyAnimation{
        id: animParamsShow
        target: root
        properties: "height"
        to: !showContent ? UIConstants.sRect * 3/2 : UIConstants.sRect * 3/2 + listView.height
        duration: 800
        easing.type: Easing.InOutBack
        running: false
    }
    MouseArea {
        id: rectangle
        anchors.fill: parent
        focus: true
        Rectangle{
            id: rectMinize
            height: UIConstants.sRect * 3/2 - 8
            color:UIConstants.bgAppColor
            anchors.left: parent.left
            anchors.leftMargin: 8
            anchors.top: parent.top
            anchors.topMargin: 4
            anchors.right: parent.right
            anchors.rightMargin: 4

            Label{
                id: txtDialog
                anchors.fill: parent
                color: UIConstants.textColor
                font.pixelSize: UIConstants.fontSize
                font.family: UIConstants.appFont
                verticalAlignment: Text.AlignVCenter
                horizontalAlignment: Text.AlignHCenter
            }
            MouseArea{
                hoverEnabled: true
                anchors.fill: parent
                onPressed: {
                    rectMinize.scale = 0.9;
                }
                onReleased: {
                    rectMinize.scale = 1;
                }
                onClicked: {
                    if(vehicle.propertiesShowCount > 0){
                        showContent =!showContent;
                        animParamsShow.start()
                    }
                }
            }
        }
        ListView {
            id: listView
            clip: true
            anchors.top: rectMinize.bottom
            anchors.topMargin: 4
            height: vehicle.propertiesShowCount < 15? vehicle.propertiesShowCount * UIConstants.sRect:
                                                  15*UIConstants.sRect
            anchors.right: parent.right
            anchors.rightMargin: 8
            anchors.left: parent.left
            anchors.leftMargin: 8
            model: vehicle.propertiesModel
//            model: ListModel {
//                ListElement {selected: false;paramName: "Param A"; unit:"m/s"; value:"12"}
//                ListElement {selected: true;paramName: "Param B"; unit:"m/s"; value:"12"}
//                ListElement {selected: true;paramName: "Param C"; unit:"m/s"; value:"12"}
//                ListElement {selected: false;paramName: "Param D"; unit:"m/s"; value:"12"}
//            }
            delegate: Item {
                height: selected?UIConstants.sRect:0
                visible: selected
                width: listView.width
                Label {
                    id: lblName
                    width: UIConstants.sRect * 5
                    height: UIConstants.sRect
                    text: name
                    anchors.verticalCenter: parent.verticalCenter
                    verticalAlignment: Text.AlignVCenter
                    horizontalAlignment: Text.AlignLeft
                    color: UIConstants.textColor
                    font.pixelSize: UIConstants.fontSize
                    font.family: UIConstants.appFont
                }

                Rectangle {
                    id: rectValue
                    color:  !isNaN(parseFloat(value)) ? ((parseFloat(value) < lowerValue) ? lowerColor :
                                  ((parseFloat(value) > upperValue) ? upperColor : middleColor)): "transparent"
                        //rectMinize.color
                    anchors.bottomMargin: 2
                    anchors.topMargin: 2
                    anchors.rightMargin: 2
                    anchors.left: lblName.right
                    anchors.right: lblUnit.left
                    anchors.bottom: parent.bottom
                    anchors.top: parent.top
                    anchors.leftMargin: 2
                    clip:true

                    Label {
                        id: lblValue
                        text: parseFloat(value).toFixed(2)
                        verticalAlignment: Text.AlignVCenter
                        horizontalAlignment: Text.AlignRight
                        anchors.fill: parent
                        color: UIConstants.textColor
                        font.pixelSize: UIConstants.fontSize
                        font.family: UIConstants.appFont
                    }
                }

                Label {
                    id: lblUnit
                    width: UIConstants.sRect*1.5
                    height: UIConstants.sRect
                    text: unit
                    color: UIConstants.textColor
                    font.pixelSize: UIConstants.fontSize
                    font.family: UIConstants.appFont
                    anchors.right: parent.right
                    anchors.rightMargin: 8
                    anchors.verticalCenter: parent.verticalCenter
                    verticalAlignment: Text.AlignVCenter
                }
            }
        }


    }
    Component.onCompleted: {
        console.log("Set Focus true");
        setFocus(true)
    }
}
