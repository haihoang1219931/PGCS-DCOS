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
import CustomViews.UIConstants 1.0

Item {
    id: overlay
    property alias color: overlayRect.color
    Rectangle {
        id: overlayRect
        anchors.fill: parent
        color: UIConstants.bgColorOverlay
        opacity: .8
    }
}
