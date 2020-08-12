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
import QtQuick 2.6
//---------------- Include custom libs ----------------------------------------

import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0
TextInput {
    id: root
    width: UIConstants.sRect * 4
    height: UIConstants.sRect
    color: UIConstants.textColor
    horizontalAlignment: Text.AlignLeft
    verticalAlignment: Text.AlignVCenter
    leftPadding: 4
    rightPadding: 4
    font.pixelSize:UIConstants.fontSize
    font.family: UIConstants.appFont
    text: ""
    font.bold: false
    clip: true
    Rectangle{
        id: txtInput
        color: UIConstants.transparentColor
        border.color: UIConstants.grayColor
        border.width: 1
        radius: UIConstants.rectRadius
        anchors.fill: parent
    }
}
