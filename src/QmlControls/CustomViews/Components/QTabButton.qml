import QtQuick 2.0
import QtQuick.Controls 2.4
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0
TabButton {
    id: root
    background: Rectangle{
        color: !checked?
        UIConstants.bgAppColor:UIConstants.grayColor
    }

    contentItem: Label{
        anchors.fill: parent
        color: UIConstants.textColor
        text: root.text
        font.family: UIConstants.appFont
        font.pixelSize: UIConstants.fontSize
        horizontalAlignment: Label.AlignHCenter
        verticalAlignment: Label.AlignVCenter
    }
}
