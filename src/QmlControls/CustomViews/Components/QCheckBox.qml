import QtQuick 2.0
import QtQuick.Controls 1.2
import QtQuick.Controls.Styles 1.4
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0
CheckBox {
    id: root
    width: UIConstants.sRect* 3
    height: UIConstants.sRect
    property bool isSynced: false
    style: CheckBoxStyle{
        indicator: Rectangle{
            implicitWidth: UIConstants.sRect
            implicitHeight: UIConstants.sRect
            radius: 3
            border.color: control.activeFocus? "darkblue":"gray"
            Rectangle{
                visible: control.checked
                color: "#555"
                border.color: "#333"
                radius: 1
                anchors.margins: UIConstants.sRect / 4
                anchors.fill: parent
            }
        }
        label: Label{
            verticalAlignment: Label.AlignVCenter
            horizontalAlignment: Label.AlignLeft
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
            color: UIConstants.textColor
            text: root.text
        }
    }
    onClicked: {
        if(isSynced){
            root.checked = !root.checked;
        }
    }
}
