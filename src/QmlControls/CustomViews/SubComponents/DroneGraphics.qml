/**
 * ==============================================================================
 * @Project: VCM01TargetViewer
 * @Component: Center Commander Info
 * @Breif:
 * @Author: Hai Nguyen Hoang
 * @Date: 23/05/201
9
 * @Language: QML
 * @License: (c) Viettel Aerospace Institude - Viettel Group
 * ============================================================================
 */

//----------------------- Include QT libs -------------------------------------
//------------------ Include QT libs ------------------------------------------
import QtQuick.Window 2.2
import QtQuick 2.6
import QtQuick.Controls 2.1
import QtQuick.Layouts 1.3
import QtGraphicalEffects 1.0
import QtQuick 2.0
//---------------- Include custom libs ----------------------------------------
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0

Item {
    id: rootItem
    width: UIConstants.sRect * 4
    height: UIConstants.sRect * 4

    property color bgColor: "white"
    property color iconColor: "blue"
    IconSVG{
        id: icon
        source: "qrc:/assets/images/icons/Quad.svg"
        color: iconColor
        size: parent.width*2/3
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.verticalCenter: parent.verticalCenter
    }
}
