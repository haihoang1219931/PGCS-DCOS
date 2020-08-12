import QtQuick 2.9
import QtQuick.Window 2.2
import QtQuick.Controls 2.4
import QtQuick.Layouts 1.3
import QtWebEngine 1.7
import QtGraphicalEffects 1.0

import io.qdt.dev 1.0
import CustomViews.Components   1.0
import CustomViews.UIConstants  1.0
FlatRectangle {
    id: root
    property string userId_
    property double latitude_
    property double longitude_
    property double speed_
    property double angle_
    property string name_
    clip: true
    signal showFinished()
    signal hideFinished()
    function showInfo(){
        lblAngle.showInfo()
    }
    function hideInfo(){
        lblSpeed.hideInfo()
    }
    Label{
        id: lblName
        x: 8
        y: 8
        height: UIConstants.sRect
        color: UIConstants.textColor
        text: name_
        verticalAlignment: Text.AlignVCenter
        horizontalAlignment: Text.AlignHCenter
        anchors.leftMargin: 8
        anchors.rightMargin: 8
        anchors.right: parent.right
        anchors.left: parent.left
        font.family: UIConstants.appFont
        font.pixelSize: UIConstants.fontSize
    }

    ToastLabel {
        id: lblAngle
        x: 15
        y: 45
        height: UIConstants.sRect
        color: UIConstants.textColor
        text: "Angle: "+Number(angle_).toFixed(2)
        anchors.leftMargin: 8
        anchors.rightMargin: 8
        anchors.right: parent.right
        anchors.left: parent.left
        font.family: UIConstants.appFont
        font.pixelSize: UIConstants.fontSize
        onShowFinished:{
            lblLat.showInfo()
        }
        onHideFinished: {
            root.hideFinished();
        }
    }


    ToastLabel {
        id: lblLat
        x: 8
        y: 67
        height: UIConstants.sRect
        color: UIConstants.textColor
        font.family: UIConstants.appFont
        font.pixelSize: UIConstants.fontSize
        text: qsTr("Lat: ") + Number(latitude_).toFixed(7)
        anchors.right: parent.right
        anchors.rightMargin: 8
        anchors.left: parent.left
        anchors.leftMargin: 8
        onShowFinished:{
            lblLon.showInfo()
        }
        onHideFinished: {
            lblAngle.hideInfo()
        }
    }



    ToastLabel {
        id: lblLon
        x: 16
        y: 89
        height: UIConstants.sRect
        color: UIConstants.textColor
        font.family: UIConstants.appFont
        font.pixelSize: UIConstants.fontSize
        text: qsTr("Lon: ") + Number(longitude_).toFixed(7)
        anchors.leftMargin: 8
        anchors.rightMargin: 8
        anchors.right: parent.right
        anchors.left: parent.left
        onShowFinished:{
            lblSpeed.showInfo()
        }
        onHideFinished: {
            lblLat.hideInfo()
        }
    }



    ToastLabel {
        id: lblSpeed
        x: 11
        y: 111
        height: UIConstants.sRect
        color: UIConstants.textColor
        font.family: UIConstants.appFont
        font.pixelSize: UIConstants.fontSize
        text: qsTr("Speed: ") + speed_ + " km/h"
        anchors.leftMargin: 8
        anchors.rightMargin: 8
        anchors.right: parent.right
        anchors.left: parent.left
        onHideFinished: {
            lblLon.hideInfo()
        }
        onShowFinished: {
            root.showFinished()
        }
    }
}

/*##^## Designer {
    D{i:10;anchors_width:134;anchors_x:8}
}
 ##^##*/
