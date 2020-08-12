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
    width: UIConstants.sRect *2
    height: UIConstants.sRect
    color: UIConstants.transparentBlue
    radius: UIConstants.rectRadius
    property bool isShowRect: true
    property bool isAutoReturn: false
    property bool isActive: false
    property bool isNormal: true
    property bool isPressed: false
    property color textColor: UIConstants.textColor
    property alias text: lblText.text
    property bool isEnable: true
    property alias horizontalAlignment: lblText.horizontalAlignment
    property alias verticalAlignment: lblText.verticalAlignment
    //--- Signals
    signal clicked()
    signal entered()
    signal exited()
    signal pressed()
    signal released()
    //--- Button background
    scale: isEnable?(isPressed?0.9:1):0.9
    opacity: isEnable?(isPressed?0.5:1):0.9
    Label{
        id: lblText
        anchors.fill: parent
        anchors.margins: 5
        verticalAlignment: Text.AlignVCenter
        horizontalAlignment: Text.AlignHCenter
        font.family: UIConstants.appFont
        font.pixelSize: UIConstants.fontSize
        color: textColor
    }
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
}
