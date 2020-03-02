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
    width: UIConstants.sRect * 27
    height: UIConstants.sRect * 2
    color: "#d75151"
    border.color: "#808080"
    property alias text: label.text
    Label {
        id: label
        text: qsTr("Label")
        color: "white"
        anchors.verticalCenter: parent.verticalCenter
        anchors.horizontalCenter: parent.horizontalCenter
    }

}

/*##^## Designer {
    D{i:0;autoSize:true;height:480;width:640}
}
 ##^##*/
