/**
 * ==============================================================================
 * @file Overlay.qml
 * @Project:
 * @Author: Trung Nguyen
 * @Date: 13/02/2019
 * @Language: QML
 * @License: (c) Viettel Aerospace Institude - Viettel Group
 * ============================================================================
 */
import QtQuick 2.0
import QtQuick.Controls 2.0
import CustomViews.UIConstants 1.0
import io.qdt.dev 1.0
Item {
    id: root
    property real zoomMin: cameraController.gimbal.zoomMin
    property real zoomMax: cameraController.gimbal.zoomMax
    property real zoomRatio: cameraController.gimbal.zoom
    property real digitalZoomMax: cameraController.gimbal.digitalZoomMax
    property real zoomTarget: cameraController.gimbal.zoomTarget
    property real zoomCalculate
    property color drawColor: UIConstants.redColor
    Canvas{
        id: cvsCenter
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.verticalCenter: parent.verticalCenter
        width: UIConstants.sRect
        height: UIConstants.sRect
        onPaint: {
            var ctx = getContext("2d");
            ctx.strokeStyle = root.drawColor;
            ctx.lineWidth = 1;
            var lineLength = UIConstants.sRect / 3;
            ctx.moveTo(width/2,height/3);
            ctx.lineTo(width/2,0);
            ctx.moveTo(width/2,height*2/3);
            ctx.lineTo(width/2,height);
            ctx.moveTo(width/3,height/2);
            ctx.lineTo(0,height/2);
            ctx.moveTo(width/3,height/2);
            ctx.lineTo(0,height/2);
            ctx.moveTo(width*2/3,height/2);
            ctx.lineTo(width,height/2);
            ctx.stroke();
        }
    }
    Item {
        id: element
        width: UIConstants.sRect * 12
        height: UIConstants.sRect * 2
        anchors.bottomMargin: 8
        anchors.bottom: parent.bottom
        anchors.horizontalCenter: parent.horizontalCenter

        Rectangle {
            id: rectZoom
            color: "#00000000"
            anchors.bottomMargin: 8
            anchors.leftMargin: 0
            anchors.left: parent.left
            anchors.top: parent.top
            anchors.topMargin: 0
            anchors.right: parent.right
            border.width: 1
            border.color: "#ff0000"
            anchors.bottom: lblZoomOptical.top

            Canvas{
                id: cvsZoomTarget
                width: parent.height*2
                height: parent.height
                anchors.bottom: parent.top
                anchors.bottomMargin: 0
                x: -width/2 + (root.zoomTarget <= root.zoomMax ?
                                           root.zoomTarget / root.zoomMax * parent.width * 2 / 3:
                                           parent.width * 2 / 3 + parent.width * 1 / 3 * (root.zoomTarget / root.zoomMax - 1) / (root.digitalZoomMax - 1))
                onPaint: {
                    var ctx = getContext("2d");
                    ctx.strokeStyle = parent.border.color;
                    ctx.lineWidth = 1;
                    ctx.moveTo(0,0);
                    ctx.lineTo(width/2,height);
                    ctx.lineTo(width,0);
                    ctx.lineTo(0,0);
                    ctx.stroke();
                }
                Label{
                    text: Number(root.zoomCalculate).toFixed(2) + "/" +Number(root.zoomTarget).toFixed(2)
                    anchors.verticalCenter: parent.verticalCenter
                    verticalAlignment: Label.AlignVCenter
                    horizontalAlignment: Label.AlignLeft
                    anchors.left: parent.right
                    anchors.leftMargin: 8
                    color: root.drawColor
                    font.pixelSize: UIConstants.fontSize
                    font.family: UIConstants.appFont
                }
            }
            Canvas{
                id: cvsZoomSpacing
                anchors.fill: parent
                onPaint: {
                    var ctx = getContext("2d");
                    ctx.strokeStyle = parent.border.color;
                    ctx.lineWidth = 1;
                    ctx.moveTo(parent.width*2/3,0);
                    ctx.lineTo(parent.width*2/3,height);
                    ctx.stroke();
                }
            }
            Rectangle {
                id: rectZoomRatio
                width: root.zoomRatio <= root.zoomMax ?
                           root.zoomRatio / root.zoomMax * parent.width * 2 / 3:
                           parent.width * 2 / 3 + parent.width * 1 / 3 * (root.zoomRatio / root.zoomMax - 1) / (root.digitalZoomMax - 1)
                color: rectZoom.border.color
                anchors.top: parent.top
                anchors.topMargin: 0
                anchors.bottom: parent.bottom
                anchors.bottomMargin: 0
                anchors.left: parent.left
                anchors.leftMargin: 0
            }
        }
        Rectangle{
            anchors.top: parent.top
        }

        Label {
            id: lblZoomOptical
            text: "Zoom: "+(root.zoomRatio<= root.zoomMax?
                                Number(root.zoomRatio).toFixed(2):
                                Number(root.zoomMax).toFixed(2)) +"/"+Number(root.zoomMax).toFixed(0)
            horizontalAlignment: Text.AlignLeft
            verticalAlignment: Text.AlignVCenter
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 8
            anchors.left: parent.left
            anchors.leftMargin: 8
            color: root.drawColor
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
        }

        Label {
            id: lblZoomDigital
            text: "Digital: x"+ (root.zoomRatio > root.zoomMax ? Number(root.zoomRatio/root.zoomMax).toFixed(2): 1.00)
            anchors.horizontalCenterOffset: UIConstants.sRect * 3
            anchors.horizontalCenter: parent.horizontalCenter
            verticalAlignment: Text.AlignVCenter
            anchors.bottomMargin: 8
            anchors.bottom: parent.bottom
            horizontalAlignment: Text.AlignLeft
            color: root.drawColor
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
        }
    }

    Item {
        id: itemMode
        x: 220
        width: UIConstants.sRect * 10
        height: UIConstants.sRect
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.top: parent.top
        anchors.topMargin: 8 + UIConstants.sRect

        Label {
            id: lblLockMode
            color: root.drawColor
            text: "Lock: "+camState.lockMode
            anchors.leftMargin: 8
            verticalAlignment: Text.AlignVCenter
            anchors.bottomMargin: 8
            font.pixelSize: UIConstants.fontSize
            anchors.bottom: parent.bottom
            font.family: UIConstants.appFont
            anchors.left: parent.left
            horizontalAlignment: Text.AlignLeft
        }

        Label {
            id: lblStab
            x: 9
            y: 9
            color: root.drawColor
            text: "Stab: "+ (camState.digitalStab?"On":"Off")
            verticalAlignment: Text.AlignVCenter
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.bottomMargin: 8
            font.pixelSize: UIConstants.fontSize
            anchors.bottom: parent.bottom
            font.family: UIConstants.appFont
            anchors.horizontalCenterOffset: UIConstants.sRect * 3
            horizontalAlignment: Text.AlignLeft
        }
    }
    Row{
        anchors.horizontalCenter: parent.horizontalCenter
        spacing: 8
        visible: false
        SpinBox{
            id: spbZoom
            width: UIConstants.sRect * 5
            height: UIConstants.sRect
            editable: true
            to: 240
            from: 1
            value: 1
            onValueChanged: {
                if(value > 0)
                    cameraController.gimbal.setEOZoom("",value)
            }
        }
    }
    Component.onCompleted: {
        cvsCenter.requestPaint();
        cvsZoomTarget.requestPaint();
        cvsZoomSpacing.requestPaint();
    }
}

/*##^##
Designer {
    D{i:0;autoSize:true;height:480;width:640}D{i:2;anchors_height:17;anchors_width:170;anchors_x:51;anchors_y:0}
}
##^##*/

/*##^## Designer {
    D{i:0;autoSize:true;height:480;width:640}
}
 ##^##*/
