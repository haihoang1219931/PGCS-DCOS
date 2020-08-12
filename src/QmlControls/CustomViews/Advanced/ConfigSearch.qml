import QtQuick 2.9
import QtQuick.Controls 2.2
import QtQuick.Layouts 1.3
//------------------ Include Custom modules/plugins
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0
Rectangle {
    id: root
    color: camState.themeColor
    width: 650
    height:450
    property color textColor: "white"

    StackLayout {
        id: swipeView
        anchors.top: tabBar.bottom
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        currentIndex: tabBar.currentIndex
        ConfigSearchClass{
        }
        ConfigSearchPlate{
        }
        ConfigSearchImage{
        }
    }
    Rectangle{
        color: camState.themeColor
        height: 40
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.right: parent.right
    }

    TabBar {
        id: tabBar
        x: 1
        y: 1
        currentIndex: swipeView.currentIndex

        spacing: 1
        TabButton {
            width: implicitWidth
            text: qsTr("Classification")
            font.family: UIConstants.appFont
            font.pixelSize: UIConstants.fontSize
            background: Rectangle{
                anchors.fill: parent
                color: UIConstants.bgAppColor
            }
        }
        TabButton {
            width: implicitWidth
            text: qsTr("Plate ID")
            font.family: UIConstants.appFont
            font.pixelSize: UIConstants.fontSize
            background: Rectangle{
                anchors.fill: parent
                color: UIConstants.bgAppColor
            }
        }
        TabButton {
            width: implicitWidth
            text: qsTr("Object Image")
            font.family: UIConstants.appFont
            font.pixelSize: UIConstants.fontSize
            background: Rectangle{
                anchors.fill: parent
                color: UIConstants.bgAppColor
            }
        }
    }
}
