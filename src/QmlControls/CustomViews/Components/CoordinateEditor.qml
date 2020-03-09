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
import QtQuick.Layouts 1.3
import QtQuick.Controls 2.1

//---------------- Include custom libs ----------------------------------------
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0

Item {
    id: root
    width: UIConstants.sRect * 8.5
    height: UIConstants.sRect * 6
    property string directionLabel: "E"
    property string title: "Coordinate"
    Label{
        id: lblTitle
        text: title
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        anchors.top: parent.top
        anchors.topMargin: 0
        anchors.horizontalCenter: parent.horizontalCenter
        height: UIConstants.sRect
    }

    ColumnLayout{
        anchors.top: lblTitle.bottom
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        RowLayout{
            Layout.alignment: Qt.AlignRight | Qt.AlignVCenter
            Layout.preferredHeight: UIConstants.sRect
            layoutDirection: Qt.RightToLeft
            QLabel{
                Layout.preferredWidth: UIConstants.sRect
                Layout.preferredHeight: UIConstants.sRect
                text: directionLabel
            }
            Label{
                Layout.preferredWidth: UIConstants.sRect/4
                Layout.preferredHeight: UIConstants.sRect/4
                color: UIConstants.textColor
                text: " "
                font.family: UIConstants.appFont
                font.pixelSize: UIConstants.fontSize
            }
            QTextInput{
                id: txtValue
                Layout.preferredWidth:  UIConstants.sRect*6.5 + 15
                Layout.preferredHeight: UIConstants.sRect
            }
        }
        RowLayout{
            Layout.alignment: Qt.AlignRight | Qt.AlignVCenter
            Layout.preferredHeight: UIConstants.sRect
            layoutDirection: Qt.RightToLeft
            QLabel{
                Layout.preferredWidth: UIConstants.sRect
                Layout.preferredHeight: UIConstants.sRect
                text: directionLabel
            }
            Label{
                Layout.preferredWidth: UIConstants.sRect/4
                Layout.preferredHeight: UIConstants.sRect/4
                color: UIConstants.textColor
                text: "\""
                font.family: UIConstants.appFont
                font.pixelSize: UIConstants.fontSize
            }
            QTextInput{
                id: txtValueS1
                Layout.preferredWidth:  UIConstants.sRect*3/2
                Layout.preferredHeight: UIConstants.sRect
            }
            Label{
                Layout.preferredWidth: UIConstants.sRect/4
                Layout.preferredHeight: UIConstants.sRect/4
                color: UIConstants.textColor
                text: "'"
                font.family: UIConstants.appFont
                font.pixelSize: UIConstants.fontSize
            }
            QTextInput{
                id: txtValueM1
                Layout.preferredWidth:  UIConstants.sRect*3/2
                Layout.preferredHeight: UIConstants.sRect
            }
            Label{
                Layout.preferredWidth: UIConstants.sRect/4
                Layout.preferredHeight: UIConstants.sRect/4
                color: UIConstants.textColor
                text: "o"
                font.family: UIConstants.appFont
                font.pixelSize: UIConstants.fontSize
            }
            QTextInput{
                id: txtValueD1
                Layout.preferredWidth:  UIConstants.sRect * 2
                Layout.preferredHeight: UIConstants.sRect
            }
        }
        RowLayout{
            Layout.preferredWidth: parent.width
            Layout.preferredHeight: UIConstants.sRect
        }
    }
}
