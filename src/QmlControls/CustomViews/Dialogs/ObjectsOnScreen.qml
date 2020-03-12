import QtQuick 2.9
import QtQuick.Window 2.2
import QtQuick.Controls 2.4
import QtQuick.Layouts 1.3
import QtWebEngine 1.7
import QtGraphicalEffects 1.0

import io.qdt.dev 1.0
import CustomViews.Components   1.0
import CustomViews.UIConstants  1.0
Item {
    id: root
    property var player
    Repeater {
        id: listObjects
        model: player.listTrackObjectInfos
        delegate: Rectangle{
            id: object
            width: rect.width / sourceSize.width * root.width
            height: rect.height / sourceSize.height * root.height
            x: rect.x / sourceSize.width * root.width
            y: rect.y / sourceSize.height * root.height
            property string userId_: userId
            property double latitude_: latitude
            property double longitude_: longitude
            property double speed_: speed
            property double angle_: angle
            property string name_: name
            property int screenX_: 0
            property int screenY_: 0
            color: UIConstants.transparentColor
            border.color: UIConstants.textColor
            property bool animFinish: false
            MouseArea{
                id: mouse
                anchors.fill: objectInfo
                onClicked: {
                    if(object.animFinish){
                        player.updateTrackObjectInfo("Object","SELECTED",!isSelected);
                        object.animFinish = false;
                    }
                }
            }
            ObjectInformation{
                id: objectInfo
                anchors.left: parent.right
                anchors.leftMargin: 20
                anchors.verticalCenter: parent.verticalCenter
                userId_: userId
                latitude_: latitude
                longitude_: longitude
                speed_: speed
                angle_: angle
                name_: name
                width: 0
                property bool _isSeleted: isSelected
                SequentialAnimation on width {
                    id: animShow
                    running: false
                    NumberAnimation {
                        to: UIConstants.sRect * 7
                        duration: 1000
                        easing.type: Easing.InOutBack
                    }
                    onStopped: {
                        objectInfo.showInfo();
                    }
                }
                SequentialAnimation on width {
                    id: animHide
                    running: false
                    NumberAnimation {
                        to: 0
                        duration: 1000
                        easing.type: Easing.OutInBack
                    }
                    onStopped: {
                        object.animFinish = true;
                    }
                }
                onHideFinished: {
                    animHide.start()
                }
                onShowFinished:{
                    object.animFinish = true;
                }

                on_IsSeletedChanged: {
                    if(_isSeleted)
                        animShow.start()
                    else{
                        objectInfo.hideInfo()
                    }
                }
            }

        }
    }
}
