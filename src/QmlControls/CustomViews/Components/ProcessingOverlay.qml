/**
 * ==============================================================================
 * @Project:
 * @Component: Processing Overlay
 * @Breif:
 * @Author: Trung Nguyen
 * @Date: 26/02/2019
 * @Language: QML
 * @License: (c) Viettel Aerospace Institude - Viettel Group
 * ============================================================================
 */

//-------------------- Include QT libs ---------------------------------------
import QtQuick 2.5
import QtQuick.Controls 2.1
import QtQuick.Layouts 1.3
import QtGraphicalEffects 1.0

//---------------- Include custom libs ----------------------------------------
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0

Item {
    id: root
    property alias textOverlay: infoProcessingOverlay.text
    property alias background: rectBound.color
    property alias transparentLevel: rectBound.opacity
    Rectangle {
        id: rectBound
        anchors.fill: parent
        color: UIConstants.cfProcessingOverlayBg
        opacity: .4
        DotProcessingAnimation {
            id: dotProcessingAnimation
            anchors.fill: parent
        }

        Text {
            id: infoProcessingOverlay
            text: "Chưa gán thiết bị"
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.verticalCenter: parent.verticalCenter
            color: UIConstants.textColor
            opacity: .7
        }

        Timer {
            id: timer
            interval: 300; running: false; repeat: false
            onTriggered: root.opacity = 0
        }
    }
}
