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
import QtQuick.Window 2.2
import QtQuick 2.6
import QtQuick.Controls 2.1
import QtQuick.Layouts 1.3
import QtGraphicalEffects 1.0
import QtQuick 2.0

//---------------- Include custom libs ----------------------------------------
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0

Rectangle {
    id: rootItem
    width: 40
    height: 40
    //--- Properties
    color: UIConstants.transparentColor
    property bool isActive: false
    property bool isNormal: true
    property alias icon: icon.text
    property alias btnText: textBtn.text
    property alias iconRotate: icon.rotation
    property int iconSize: UIConstants.fontSize * 3 / 2
    property bool isSolid: true
    property bool isEnable: true
    property color bgColor: UIConstants.transparentColor
    property alias lineThroughEnable: lineThrough.visible
    property bool isOn: true
    property bool isSync: false
    radius: UIConstants.rectRadius
    //--- Signals
    signal clicked()

    //--- Button wrapper
    Rectangle {
        id: rectangle
        anchors.fill: parent
        color: rootItem.bgColor
        radius: rootItem.radius
        Text {
            id: icon
            text: UIConstants.iAdd
            wrapMode: Text.WordWrap
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignHCenter
            font{ pixelSize: iconSize;
                weight: rootItem.isSolid?Font.Bold:Font.Normal;
                family: rootItem.isSolid?ExternalFontLoader.solidFont:ExternalFontLoader.regularFont }
            color: textBtn.color
            rotation: 0
            anchors.topMargin: 5
            anchors.top: parent.top
            anchors.left: parent.left
            anchors.right: parent.right
            height: parent.height/2
            Canvas {
                id: lineThrough
                anchors.fill: parent
                visible: false
                onPaint: {
                    var ctx = getContext("2d");
                    ctx.strokeStyle = parent.color;
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.moveTo(0,0);
                    ctx.lineTo(width, height);
                    ctx.closePath();
                    ctx.stroke();
                }
            }
        }

        //--- Text
        Text {
            id: textBtn
            text: "Text"
//                anchors.horizontalCenter: parent.horizontalCenter
            verticalAlignment: Text.AlignBottom
            horizontalAlignment: Text.AlignHCenter
            Layout.alignment: Qt.AlignHCenter | Qt.AlignVCenter
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
            anchors.top: icon.bottom
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 0
            color: isEnable ? UIConstants.textFooterColor : UIConstants.cDisableColor
        }
    }
    Text {
        id: iconSwitch
        x: 33
        text: UIConstants.iEnabled
        anchors.top: parent.top
        anchors.topMargin: 5
        anchors.right: parent.right
        anchors.rightMargin: 5
        wrapMode: Text.WordWrap
        verticalAlignment: Text.AlignVCenter
        horizontalAlignment: Text.AlignHCenter
//                anchors.horizontalCenter: parent.horizontalCenter
        Layout.alignment: Qt.AlignHCenter | Qt.AlignVCenter
        font{ pixelSize: iconSize / 2;
            weight: Font.Bold;
            family: ExternalFontLoader.solidFont}
        color: rootItem.isOn?UIConstants.greenColor:UIConstants.grayColor
        rotation: 0
    }
    //--- Button background
    Rectangle {
        id: btnBackground
        anchors.fill: parent
        color: isEnable ? UIConstants.cateOverlayBg : UIConstants.cDisableColor
        opacity: .2
        radius: rootItem.radius
//        radius: 5
        //--- Click event
        MouseArea {
            id: btnSelectedArea
            anchors.fill: parent
            enabled: isEnable ? true : false
            onPressed: {
                parent.opacity = 0.5;
                rootItem.clicked();
                if(!isSync){
                    rootItem.isOn=!rootItem.isOn;
                }
            }

            onReleased: {
                parent.opacity = 0.2;
            }
        }
    }

    //--- Js supported functions
    function setButtonActive()
    {
        btnBackground.color = UIConstants.info
        isActive = true
        isNormal = false
    }

    function setButtonNormal()
    {
        btnBackground.color = UIConstants.cateOverlayBg;
        textBtn.color = UIConstants.textFooterColor
        isNormal = true
        isActive = false
    }

    function setButtonDisable()
    {
        btnSelectedArea.hoverEnabled = true;
        btnSelectedArea.preventStealing = true;
        btnSelectedArea.enabled = false;

        btnBackground.color = UIConstants.cDisableColor
        textBtn.color = UIConstants.cDisableColor
    }
}
