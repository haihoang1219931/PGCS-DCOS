/**
 * ==============================================================================
 * @Project: VCM01TargetViewer
 * @Component: Map control buttons
 * @Breif: zoom in, zoom out, center at
 * @Author: Trung Nguyen
 * @Date: 27/03/2019
 * @Language: QML
 * @License: (c) Viettel Aerospace Institude - Viettel Group
 * ============================================================================
 */

//----------------------- Include QT libs -------------------------------------
import QtQuick.Window 2.2
import QtQuick 2.6
import QtQuick.Controls 2.1
import QtQuick.Layouts 1.3
import QtGraphicalEffects 1.0
import QtQuick 2.0

//---------------- Include custom libs ----------------------------------------
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0

Rectangle {
    id: rootItem
    width: UIConstants.sRect*3
    height: UIConstants.sRect*12
    color: UIConstants.transparentBlue
    border.color: UIConstants.textColor
    border.width: 1
    radius: 2
    signal focusAll()
    signal zoomIn()
    signal zoomOut()

    ColumnLayout {
        anchors.fill: parent
        spacing: 0
        z: 10
        FlatButtonIcon {
            icon: UIConstants.iCenter
            iconColor: UIConstants.textColor
            Layout.preferredHeight: parent.height / 3
            Layout.preferredWidth: parent.width
            iconSize: 30
            isAutoReturn: true
            isShowRect: false
            onClicked: {
                rootItem.focusAll()
            }
            Label{
                anchors.horizontalCenter: parent.horizontalCenter
                horizontalAlignment: Label.AlignHCenter
                anchors.bottom: parent.bottom
                text: "Center"
                color: UIConstants.textColor
                font.pixelSize: UIConstants.fontSize
            }
        }

        FlatButtonIcon {
            icon: UIConstants.iZoomIn
            iconColor: UIConstants.textColor
            Layout.preferredHeight: parent.height / 3
            Layout.preferredWidth: parent.width
            iconSize: 15
            isAutoReturn: true
            isShowRect: false
            onClicked: {
                rootItem.zoomIn()
            }
            Label{
                anchors.horizontalCenter: parent.horizontalCenter
                horizontalAlignment: Label.AlignHCenter
                anchors.bottom: parent.bottom
                anchors.bottomMargin: 5
                text: "In"
                color: UIConstants.textColor
                font.pixelSize: UIConstants.fontSize
            }
        }

        FlatButtonIcon {
            icon: UIConstants.iZoomOut
            iconColor: UIConstants.textColor
            Layout.preferredHeight: parent.height / 3
            Layout.preferredWidth: parent.width
            iconSize: 15
            isAutoReturn: true
            isShowRect: false
            onClicked: {
                rootItem.zoomOut()
            }
            Label{
                anchors.horizontalCenter: parent.horizontalCenter
                horizontalAlignment: Label.AlignHCenter
                anchors.bottom: parent.bottom
                anchors.bottomMargin: 5
                text: "Out"
                color: UIConstants.textColor
                font.pixelSize: UIConstants.fontSize
            }
        }
    }
}
