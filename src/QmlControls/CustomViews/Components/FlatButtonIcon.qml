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

    //--- Properties
    width: 60
    height: 60
    color: !isShowRect?UIConstants.transparentColor:bgColor
    radius: UIConstants.rectRadius
    property bool isShowRect: true
    property bool isAutoReturn: false
    property bool isActive: false
    property bool isNormal: true
    property bool isPressed: false
    property color iconColor: UIConstants.textColor
    property alias icon: icon.text
    property alias iconRotate: icon.rotation
    property bool isEnable: true
    property bool isSolid: false
    property color bgColor: UIConstants.transparentBlue
    property int iconSize: height / 3 * 2
    property int idx: -1
    //--- Signals
    signal clicked()
    signal entered()
    signal exited()
    signal pressed()
    signal released()
    scale: isEnable?(isPressed?0.9:1):0.9
    opacity: isEnable?(isPressed?0.5:1):0.9
    //--- Click event
    MouseArea {
        id: btnSelectedArea
        anchors.fill: parent
        hoverEnabled: true
        enabled: isEnable ? true : false
        onPressed: {
            rootItem.isPressed=!rootItem.isPressed;
            rootItem.pressed();
        }

        onReleased: {
            if(isAutoReturn){
                rootItem.isPressed=!rootItem.isPressed;
                rootItem.released();
            }
        }
        onClicked: {
            rootItem.clicked();
        }

        onEntered: {
            rootItem.entered();
        }
        onExited: {
            rootItem.exited();
        }
    }
    Label {
        id: icon
        text: UIConstants.iAdd
        verticalAlignment: Text.AlignVCenter
        horizontalAlignment: Text.AlignHCenter
        anchors.fill: parent
        Layout.alignment: Qt.AlignCenter
        font{ pixelSize: rootItem.isPressed?iconSize/1.2:iconSize;
            weight: isSolid?Font.Bold:Font.Normal;
            family: isSolid?ExternalFontLoader.solidFont:ExternalFontLoader.regularFont }
        color: iconColor
//            opacity: rootItem.isPressed?1:0.5

        rotation: 0
    }



    //--- Js supported functions
    function setButtonActive()
    {
        isPressed = true
        isActive = true
        isNormal = false
    }

    function setButtonNormal()
    {
        isPressed = false
        isNormal = true
        isActive = false
    }

    function setButtonDisable()
    {
        isEnable = false;
    }
}
