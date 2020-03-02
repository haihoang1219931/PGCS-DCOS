//------------------ Include QT libs ------------------------------------------
import QtQuick.Window 2.2
import QtQuick 2.6
import QtQuick.Controls 2.1
import QtQuick.Layouts 1.3
import QtGraphicalEffects 1.0
import QtQuick 2.0

//------------------ Include Custom modules/plugins
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0
//import QGroundControl 1.0

//---------------- Component definition ---------------------------------------
Rectangle {
    id: rootItem
    width: UIConstants.sRect*2
    height: UIConstants.sRect*12
    color: UIConstants.transparentBlue
    border.color: UIConstants.boundColor
    border.width: 1
    radius: UIConstants.rectRadius
    signal focusAll()
    signal zoomIn()
    signal zoomOut()

    ColumnLayout {
        anchors.fill: parent
        spacing: 0
        z: 10
        FlatButtonIcon {
            icon: UIConstants.iCompress
            iconColor: UIConstants.textColor
            Layout.preferredHeight: parent.height / 3
            Layout.preferredWidth: parent.width
            iconSize: 30
            isSolid: true
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
                font.family: UIConstants.appFont
            }
        }

        FlatButtonIcon {
            icon: UIConstants.iZoomIn
            iconColor: UIConstants.textColor
            Layout.preferredHeight: parent.height / 3
            Layout.preferredWidth: parent.width
            iconSize: 20
            isSolid: true
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
                font.family: UIConstants.appFont
            }
        }

        FlatButtonIcon {
            icon: UIConstants.iZoomOut
            iconColor: UIConstants.textColor
            Layout.preferredHeight: parent.height / 3
            Layout.preferredWidth: parent.width
            iconSize: 20
            isSolid: true
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
                font.family: UIConstants.appFont
            }
        }
    }
}
