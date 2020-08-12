/**
 * ==============================================================================
 * @Project: VCM01TargetViewer
 * @Component: Icon for mutual using
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

Item {
    id: root
    property alias type: icon.text
    property alias color: icon.color
    property alias size: icon.font.pixelSize
    property alias supText: textDes.text
    property alias needSupText: textDes.visible

    signal clicked();
    //--- Main icon
    Text {
        id: icon
        font{ pixelSize: 16; weight: Font.Bold; family: ExternalFontLoader.icons }
        text: "\uf028"
        color: UIConstants.textFooterColor
        transform: Rotation { id: iconRot; angle: 0 }
    }

    //--- Sub text
    Text {
        id: textDes
        font{ pixelSize: 9; bold: true; family: ExternalFontLoader.solidFont }
        color: icon.color
        text: "VGA"
        anchors.left: icon.right
        anchors.leftMargin: 3
        y: 1
        visible: false
    }

    //--- Mouse event
    MouseArea {
        anchors.fill: parent
        enabled: true
        hoverEnabled: true
        onPressed: {
            root.clicked();
            icon.opacity = .5
        }
        onReleased: {
            icon.opacity = 1
        }
    }
}
