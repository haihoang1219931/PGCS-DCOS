/**
 * ==============================================================================
 * @Project: VCM01TargetViewer
 * @Component: SidebarTitle
 * @Breif:
 * @Author: Trung Nguyen
 * @Date: 27/02/2019
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

Rectangle {
    id: rootItem
    color: UIConstants.bgColorOverlay
    opacity: .7

    //------- Properties
    property alias title: txtTitle.text
    property alias iconType: iconList.text
    property alias xPosition: iconList.x

    //------- Sidebar border
    RectBorder {
        id: borderBottom
        type: "bottom"
        thick: 2
    }
    Text {
        id: iconList
        text: UIConstants.iList
        font{ bold: true; pixelSize: UIConstants.fontSize; family: ExternalFontLoader.solidFont }
        x: parent.width * 1 / 5
        anchors.verticalCenter: parent.verticalCenter
        color: UIConstants.textColor
    }
    Text {
        id: txtTitle
        text: "Title"
        font.pixelSize: UIConstants.fontSize
        font.family: UIConstants.appFont
        anchors.left: iconList.right
        anchors.leftMargin: 10
        anchors.verticalCenter: parent.verticalCenter
        color: UIConstants.textColor
        Behavior on text {
            PropertyAnimation {
                duration: 400
                easing.type: Easing.InCurve
            }
        }
    }
    gradient: Gradient {
        GradientStop { position: 0.0; color: UIConstants.cfProcessingOverlayBg }
        GradientStop { position: 0.8; color: UIConstants.bgColorOverlay }
        GradientStop { position: 1.0; color: UIConstants.bgColorOverlay }
    }
}
