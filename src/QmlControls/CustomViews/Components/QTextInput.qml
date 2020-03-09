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
    width: UIConstants.sRect * 4
    height: UIConstants.sRect
    color: UIConstants.transparentColor
    clip: true
    border.color: UIConstants.grayColor
    border.width: 1
    radius: UIConstants.rectRadius
//    property alias text: textInput.text
//    property alias focus: textInput.focus
//    TextInput{
//        id: textInput
//        color: UIConstants.textColor
//        horizontalAlignment: Text.AlignLeft
//        font.pixelSize:0
//        font.family: UIConstants.appFont
//        text: "Text input"
//        anchors.verticalCenter: parent.verticalCenter
//        anchors.left: parent.left
//        anchors.leftMargin: 4
//        anchors.right: parent.right
//        anchors.rightMargin: 4
//    }
}
