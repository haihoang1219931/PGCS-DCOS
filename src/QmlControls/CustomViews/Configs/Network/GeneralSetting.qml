import QtQuick 2.0
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0
Item {
    width: 630
    height: 320
    Column{
        anchors.fill: parent
        anchors.margins: 8
        spacing: 5
        QCheckBox {
            id: checkBox
            text: qsTr("Automatically connect this network when it is available")
            width: parent.width
            height: UIConstants.sRect*1.5
        }

        QCheckBox {
            id: checkBox1
            text: qsTr("Allow users may connect to this network")
            width: parent.width
            height: UIConstants.sRect*1.5
        }
    }
}
