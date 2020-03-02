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
    id: flatBtn
    //---------- properties
    height: 60
    property alias btnText: flatBtnText.text
    property alias btnTextColor: flatBtnText.color
    property alias btnBgColor: flatBtn.color
    property alias radius: flatBtn.radius
    property alias iconVisible: iconBtn.visible
    property alias icon: iconBtn.text
    property bool active: true
    property int idx

    //---------- Signals
    signal clicked(real idx)

    //---------- Ele attributes
    state: "normal"
    z: 10
//    radius: 1
    width: iconBtn.width + flatBtnText.width + 20

    //---------- Icon
    Text {
        id: iconBtn
        visible: false
        text: UIConstants.iX
        font{ pixelSize: 18; bold: true;  family: ExternalFontLoader.solidFont }
        anchors{ verticalCenter: parent.verticalCenter; right: flatBtnText.left;
                 rightMargin: 5 }
        color: flatBtnText.color
    }
    //---------- Text
    Text {
        id: flatBtnText
        text: "Button Text"
        font.pixelSize: 13
        font.family: UIConstants.fontComicSans
        anchors.verticalCenter: parent.verticalCenter
        x: iconBtn.width + 15
        color: UIConstants.textBlueColor
        font.capitalization: Font.AllUppercase
    }


    //--------- Background
    color: active ? UIConstants.sidebarActiveBg : UIConstants.transparentColor
    opacity: (flatBtn.state === "down") ? 0.3 : 0.7

    //--------- Active States
    Rectangle {
        id: activeStatesWrapper
        anchors.fill: parent
        color: UIConstants.transparentColor
        clip: true
        Rectangle {
            id: underLine
            width: active ? parent.width + 10 : 0
            height: 4
            color: "#2980b9"
            anchors { left: parent.left; bottom: parent.bottom; leftMargin: 7 }
            Behavior on width {
                NumberAnimation {
                    duration: 250
                    easing.type: Easing.Linear
                }
            }
        }

        //---------- Icon
        Text {
            id: iconBtnActive
            visible: iconBtn.visible
            text: iconBtn.text
            font { pixelSize: iconBtn.font.pixelSize; bold: true; family: ExternalFontLoader.solidFont }
            anchors { verticalCenter: parent.verticalCenter; right: flatBtnTextActive.left;
                     rightMargin: 5 }
            color: flatBtnTextActive.color
            opacity: flatBtnTextActive.opacity
        }
        //---------- Text
        Text {
            id: flatBtnTextActive
            text: flatBtnText.text
            font { pixelSize: flatBtnText.font.pixelSize; family: flatBtnText.font.family;
                            capitalization: Font.AllUppercase }
            anchors.verticalCenter: parent.verticalCenter
            color: UIConstants.textColor
            opacity: active ? 1 : 0
            x: iconBtn.width + 15
            Behavior on opacity {
                NumberAnimation {
                    duration: 250
                    easing.type: Easing.InExpo
                }
            }
        }
    }

    //---------- Event listener
     MouseArea {
         anchors.fill: parent
         hoverEnabled: true
         enabled: true
         onPressed: { flatBtn.state = "down"; active = true; flatBtn.clicked(idx); }
         onReleased: { flatBtn.state = "normal"; }
         onEntered: { if( !active ) { setActive() }; }
         onExited: { if( !active ) { setInactive(); } }
     }

    //----------- Js supported funcs
    function setActive()
    {
        flatBtnTextActive.opacity = 1;
        underLine.width = activeStatesWrapper.width + 10;
        flatBtn.color = UIConstants.sidebarActiveBg;

    }

    function setInactive()
    {
        flatBtnTextActive.opacity = 0;
        underLine.width = 0;
        flatBtn.color = UIConstants.transparentColor;
    }
}
