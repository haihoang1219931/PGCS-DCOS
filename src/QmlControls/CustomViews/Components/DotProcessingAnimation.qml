/**
 * ==============================================================================
 * @Project: VCM01TargetViewer
 * @Component:
 * @Breif:
 * @Author: Trung Nguyen
 * @Date: 19/02/2019
 * @Language: QML
 * @License: (c) Viettel Aerospace Institude - Viettel Group
 * ============================================================================
 */

//------------------ Include QT libs ------------------------------------------
import QtQuick.Window 2.2
import QtQuick 2.6
import QtQuick.Controls 2.1
import QtQuick.Layouts 1.3
import QtGraphicalEffects 1.0

//---------------- Include custom libs ----------------------------------------
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0

Item {
    id: root
    Row {
        id: bwContent
        anchors.horizontalCenter: parent.horizontalCenter
        y: parent.height / 2 - 100

        Text {
            id: dot0
            text: "."
            font.pointSize: 50
            color: "#fff"
        }
        Text {
            id: dot1
            opacity: 0
            text: "."
            font.pointSize: 50
            color: "#fff"
        }
        Text {
            id: dot2
            opacity: 0
            text: "."
            font.pointSize: 50
            color: "#fff"
        }
        Text {
            id: dot3
            opacity: 0
            text: "."
            font.pointSize: 50
            color: "#fff"
        }
        SequentialAnimation {
            id: anim
            loops: Animation.Infinite
            running: true
            NumberAnimation {
                target: dot0
                properties: "opacity"
                from: 0
                to: 1
                duration: 400
            }
            NumberAnimation {
                target: dot1
                properties: "opacity"
                from: 0
                to: 1
                duration: 400
            }
            NumberAnimation {
                target: dot2
                properties: "opacity"
                from: 0
                to: 1
                duration: 400
            }
            NumberAnimation {
                target: dot3
                properties: "opacity"
                from: 0
                to: 1
                duration: 400
            }
            NumberAnimation {
                target: dot3
                properties: "opacity"
                from: 1
                to: 0
                duration: 300
            }
            NumberAnimation {
                target: dot2
                properties: "opacity"
                from: 1
                to: 0
                duration: 300
            }
            NumberAnimation {
                target: dot1
                properties: "opacity"
                from: 1
                to: 0
                duration: 300
            }
            NumberAnimation {
                target: dot0
                properties: "opacity"
                from: 1
                to: 0
                duration: 300
            }
        }

    }
}
