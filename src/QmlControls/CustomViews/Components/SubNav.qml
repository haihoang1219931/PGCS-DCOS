/**
 * ==============================================================================
 * @Project: VCM01TargetViewer
 * @Component: List Data for control pane
 * @Breif:
 * @Author: Trung Nguyen
 * @Date: 27/02/2019
 * @Language: QML
 * @License: (c) Viettel Aerospace Institude - Viettel Group
 * ============================================================================
 */

//-------------------- Include QT libs ---------------------------------------
import QtQuick 2.6
import QtQuick.Controls 2.1
import QtQuick.Layouts 1.3
import QtGraphicalEffects 1.0

//---------------------Include custom modules
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0

//---------------------------------------------------------------------------

ListView {
    id: rootItem
    height: childrenRect.height
    property string choosedItem
    property string prevItem
    property int prevIndex
    signal listViewClicked(string choosedItem)
    property color color: UIConstants.transparentColor
    clip: true
    function setCurrentText(text){
//        console.log("SubNav flightmodes.length = "+model.length);
        for(var index=0; index< model.length; index++){
//            console.log("model["+index+"] = "+model[index]);
            if(model[index] === text){
                currentIndex = index;
                break;
            }
        }
    }
    model:[]

    delegate: Item {
        width: parent.width
        height: UIConstants.sRect
        Rectangle {
            id: rectBound
            anchors.fill: parent
            color: rootItem.color
            opacity: 0.6
            Label {
                id: txt
                verticalAlignment: Text.AlignVCenter
                horizontalAlignment: Text.AlignLeft
                anchors.left: parent.left
                anchors.leftMargin: 10
                text: modelData
                color: rootItem.currentIndex === index ?
                           UIConstants.bgAppColor:UIConstants.textColor
                font.family: UIConstants.appFont
                font.pixelSize: UIConstants.fontSize
            }

            MouseArea {
                anchors.fill: parent
                hoverEnabled: true
                onEntered: {
                    rectBound.opacity = 1;
                }
                onExited: {
                    rectBound.opacity = 0.6;
                }
                onPressed: {
                    rectBound.opacity = 1;
                }

                onReleased: {
                    rectBound.opacity = 0.6;
                }

                onClicked: {
//                    console.log("Subnav model["+currentIndex+"] = "+rootItem.model[currentIndex]);
                    prevItem = rootItem.model[currentIndex];
                    rootItem.currentIndex = index;
                    rootItem.listViewClicked(txt.text);
                    rootItem.choosedItem = txt.text;
                }
            }
        }
    }
    highlight: Rectangle { color: UIConstants.textColor }
    focus: true
}
