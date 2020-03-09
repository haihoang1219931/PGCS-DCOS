/**
 * ==============================================================================
 * @Project: VCM01TargetViewer
 * @Component: system time
 * @Breif:
 * @Author: Trung Nguyen
 * @Date: 26/03/2019
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
import QtQuick 2.0
//---------------- Include custom libs ----------------------------------------
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0

Rectangle {
    id: root
    width: UIConstants.sRect
    height: UIConstants.sRect
    color: UIConstants.transparentColor
    clip: true
    border.color: UIConstants.grayColor
    border.width: 1
    radius: UIConstants.rectRadius
    property alias text: label.text
    Label{
        id: label
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        color: UIConstants.textColor
        font.pixelSize:UIConstants.fontSize
        font.family: UIConstants.appFont
        text: "L"
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.verticalCenter: parent.verticalCenter
    }
}
