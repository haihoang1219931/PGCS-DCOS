//------------------ Include QT libs ------------------------------------------
import QtQuick.Window 2.2
import QtQuick 2.6
import QtQuick.Controls 2.1
import QtQuick.Layouts 1.3
import QtGraphicalEffects 1.0
import QtQuick 2.0

//------------------ Include Custom modules/plugins
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0
//import QGroundControl 1.0

//---------------- Component definition ---------------------------------------
Item{
    id: rootItem
    width: UIConstants.sRect*19
    height: UIConstants.sRect*13
    property real maxSize: 0.75
    property real minSize: 0.10
    property string layoutMax: UIConstants.layoutMaxPaneVideo
    signal focusAll()
    signal zoomIn()
    signal zoomOut()
    signal switchClicked();
    signal minimizeClicked();
    signal newSize(real newWidth,real newHeight)
    signal sensorClicked()
    signal gcsShareClicked()
    signal gcsStabClicked()
    signal gcsRecordClicked()
    signal gcsSnapshotClicked()
    state: "show"

    states: [
        State{
            name: "show"
            PropertyChanges{
                target: rootItem
                width: UIConstants.sRect*19
                height: UIConstants.sRect*13
            }
        },
        State{
            name: "hide"
            PropertyChanges{
                target: rootItem
                width: UIConstants.sRect*2
                height: UIConstants.sRect*2
            }
        }
    ]
    transitions: [
        Transition{
            from: "show"; to: "hide"
            ParallelAnimation{
                NumberAnimation {
                    target: rootItem
                    property: "width"
                    duration: 200
                    easing.type: Easing.InOutQuad
                }
                NumberAnimation {
                    target: rootItem
                    property: "height"
                    duration: 200
                    easing.type: Easing.InOutQuad
                }
            }
        },
        Transition{
            from: "hide"; to: "show"
            ParallelAnimation{
                NumberAnimation {
                    target: rootItem
                    property: "width"
                    duration: 200
                    easing.type: Easing.InOutQuad
                }
                NumberAnimation {
                    target: rootItem
                    property: "height"
                    duration: 200
                    easing.type: Easing.InOutQuad
                }
            }
        }
    ]

    FlatButtonIcon{
        id: btnMaxSize
        width: UIConstants.sRect*2
        height: UIConstants.sRect*2
        anchors.top: parent.top
        anchors.left: parent.left
        icon: UIConstants.iWindowStore
        iconSize: UIConstants.sRect
        isSolid: false
        isShowRect: false
        visible: rootItem.state === "show"
        onClicked: {
            rootItem.switchClicked();
        }
    }
    FlatButtonIcon{
        id: btnSmallSize
        width: UIConstants.sRect*2
        height: UIConstants.sRect*2
        anchors.bottom: parent.bottom
        anchors.left: parent.left
        icon: UIConstants.iHide
        rotation: rootItem.state === "show"?0:180
        iconSize: UIConstants.sRect
        isSolid: true
        isShowRect: false
        border.color: rootItem.state === "hide"?
                          UIConstants.grayColor:UIConstants.transparentColor
        color: rootItem.state === "hide"?
                   UIConstants.grayColor:UIConstants.transparentColor
        border.width: rootItem.state === "hide"?1:0
        onClicked: {
            if(rootItem.state === "show")
                rootItem.state = "hide";
            else
                rootItem.state = "show";
            minimizeClicked();
            newSize(rootItem.width,rootItem.height)
        }
    }
    Item{
        id: pipResize
        width: UIConstants.sRect*2
        height: UIConstants.sRect*2
        anchors.top: parent.top
        anchors.right: parent.right
        visible: rootItem.state === "show"
        property real initialX: 0
        property real initialWidth: 0
        Canvas{
            anchors.fill: parent
            onPaint: {
                var ctx = getContext("2d");
                // set drawing stype
                ctx.strokeStyle = UIConstants.textColor;
                ctx.lineWidth = 2;
                var space = 10;
                var lengthLine = width - 2*space;
                ctx.moveTo(space ,space);
                ctx.lineTo(width - space, space)
                ctx.lineTo(width - space, width-space);
                ctx.stroke();
            }
        }

        MouseArea{
            anchors.fill: parent
            drag.target: rootItem.width >= 150?parent:undefined
            onPressed: {


                pipResize.anchors.top = undefined // Top doesn't seem to 'detach'
                pipResize.anchors.right = undefined // This one works right, which is what we really need
            }
            onReleased: {
                pipResize.anchors.top = rootItem.top
                pipResize.anchors.right = rootItem.right
            }
            onPositionChanged: {
                if(pressed){

                    var currentRatio = rootItem.width / rootItem.height;
                    if(pipResize.x < 150 - pipResize.width){
                        pipResize.x = 150 - pipResize.width;
                    }
                    rootItem.width = pipResize.x + pipResize.width;
                    rootItem.height = rootItem.width / currentRatio;
                    pipResize.y = 0;
                    newSize(rootItem.width,rootItem.height);
                }
            }
        }
    }
    StackLayout{
        id: stkBtn
        anchors.top: pipResize.bottom
        anchors.right: parent.right
        width: UIConstants.sRect*2
        height: layoutMax === UIConstants.layoutMaxPaneVideo?
                    UIConstants.sRect*2*3:UIConstants.sRect*2*5
        currentIndex: layoutMax === UIConstants.layoutMaxPaneVideo?1:0
        visible: rootItem.height > height + pipResize.height
        ColumnLayout{
            id: groupBtnVideo
            FlatButtonIcon{
                id: btnShareVideo
                Layout.preferredWidth: UIConstants.sRect*2
                Layout.preferredHeight: UIConstants.sRect*2
                icon: UIConstants.iShare
                iconSize: UIConstants.sRect
                isSolid: true
                isShowRect: false
                iconColor: camState.gcsShare?UIConstants.greenColor:UIConstants.textColor
                onClicked: {
                    rootItem.gcsShareClicked();
                }
            }
            FlatButtonIcon {
                id: btnSensor
                Layout.preferredWidth: UIConstants.sRect*2
                Layout.preferredHeight: UIConstants.sRect*2
                icon: UIConstants.iSensor
                isAutoReturn: true
                iconSize: UIConstants.sRect
                isSolid: true
                isShowRect: false
                iconRotate: camState.sensorID === camState.sensorIDEO?0:180
                onClicked: {
                    sensorClicked();
                }
            }
            FlatButtonIcon {
                id: btnSnapshot
                Layout.preferredWidth: UIConstants.sRect*2
                Layout.preferredHeight: UIConstants.sRect*2
                icon: UIConstants.iSnapshot
                isAutoReturn: true
                isShowRect: false
                isSolid: true
                iconSize: UIConstants.sRect
                onClicked: {
                    rootItem.gcsSnapshotClicked();
                }
            }
            FlatButtonIcon {
                id: btnGcsStab
                Layout.preferredWidth: UIConstants.sRect*2
                Layout.preferredHeight: UIConstants.sRect*2
                icon: UIConstants.iGCSStab
                isAutoReturn: true
                isShowRect: false
                isSolid: true
                iconSize: UIConstants.sRect
                iconColor: camState.gcsStab?UIConstants.greenColor:UIConstants.textColor
                onClicked: {
                    rootItem.gcsStabClicked();
                }
            }
            FlatButtonIcon {
                id: btnGCSRecord
                Layout.preferredWidth: UIConstants.sRect*2
                Layout.preferredHeight: UIConstants.sRect*2
                icon: UIConstants.iGCSRecord
                isAutoReturn: true
                isShowRect: false
                isSolid: true
                iconSize: UIConstants.sRect
                iconColor: camState.gcsRecord?UIConstants.greenColor:UIConstants.textColor
                onClicked: {
                    rootItem.gcsRecordClicked();
                }
            }
        }
        ColumnLayout{
            id: groupBtnMap
            FlatButtonIcon{
                id: btnFocus
                Layout.preferredWidth: UIConstants.sRect*2
                Layout.preferredHeight: UIConstants.sRect*2
                icon: UIConstants.iCompress
                iconSize: UIConstants.sRect
                isAutoReturn: true
                isSolid: true
                isShowRect: false
                onClicked: {
                    rootItem.focusAll()
                }
            }
            FlatButtonIcon{
                id: btnZoomIn
                Layout.preferredWidth: UIConstants.sRect*2
                Layout.preferredHeight: UIConstants.sRect*2
                icon: UIConstants.iZoomIn
                iconSize: UIConstants.sRect
                isAutoReturn: true
                isSolid: true
                isShowRect: false
                onClicked: {
                    rootItem.zoomIn()
                }
            }
            FlatButtonIcon{
                id: iZoomOut
                Layout.preferredWidth: UIConstants.sRect*2
                Layout.preferredHeight: UIConstants.sRect*2
                icon: UIConstants.iZoomOut
                iconSize: UIConstants.sRect
                isAutoReturn: true
                isSolid: true
                isShowRect: false
                onClicked: {
                    rootItem.zoomOut()
                }
            }
        }
    }
}
