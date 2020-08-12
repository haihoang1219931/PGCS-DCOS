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
Label {
    id: rootItem
    property bool isSolid: true
    property alias icon: rootItem.text
    property int iconSize: 16
    width: 40
    height: 40
    text: UIConstants.iAdd
    verticalAlignment: Text.AlignVCenter
    horizontalAlignment: Text.AlignHCenter
    font{ pixelSize: rootItem.iconSize;
        weight: isSolid?Font.Bold:Font.Normal;
        family: isSolid?ExternalFontLoader.solidFont:ExternalFontLoader.regularFont }
    rotation: 0
}
