/**
 * ==============================================================================
 * @Project: VCM01TargetViewer
 * @Component:
 * @Breif:
 * @Author: Trung Nguyen
 * @Date: 19/02/2019
 * @Language: QML
 * @License: (c) Viettel Aerospace Institude - Viettel Group
 * ============================================================================
 */

//------------------ Include QT libs ------------------------------------------
import QtQuick.Window 2.2
import QtQuick 2.6
import QtQuick.Controls 2.1
import QtQuick.Layouts 1.3
import QtGraphicalEffects 1.0

//---------------- Include custom libs ----------------------------------------
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0

//---------------- Component definition ----------------------------------------
Rectangle {
    id: rootItem

    //-------------- properties
    property alias currentIndex: sidebarElements.currentIndex
    property alias model: sidebarElements.model
    property var itemListName:
        UIConstants.itemTextMultilanguages["PRECHECK"]
    //-------------- signal
    signal displayActiveConfigBoard( real boardId_ )

    //------------------- Set background
    color: UIConstants.transparentColor

    //-------------- Sidebar title
    SidebarTitle {
        id: sidebarTitle
        anchors { top: parent.top; left: parent.left; right: parent.right }
        height: UIConstants.sRect * 2
        visible: true
        title: itemListName["MENU_TITTLE"]
               [UIConstants.language[UIConstants.languageID]]
    }

    //--------------- Sidebar list / Sidebar content
    ListView {
        id: sidebarElements
        anchors { top: sidebarTitle.bottom; left: parent.left; right: parent.right; bottom: parent.bottom }        
        currentIndex: 0
        delegate: CheckListEle {
            width: parent.width
            height: visible?60:0
            stateE: state_
            textSide: text_
            visible: showed_
            actived: sidebarElements.currentIndex === index
        }
    }

    //---------------- Js supported functions
    function next()
    {
        for(var i=currentIndex+1; i < sidebarElements.model.count ; i++){
            if(sidebarElements.model.get(i).showed_){
                sidebarElements.currentIndex = i;
                break;
            }
        }
    }

    function prev()
    {
        for(var i=currentIndex-1; i >= 0 ; i--){
            if(sidebarElements.model.get(i).showed_){
                sidebarElements.currentIndex = i;
                break;
            }
        }
    }

    function doCheck()
    {
        if(sidebarElements.currentIndex >= 0 &&
            sidebarElements.currentIndex < sidebarElements.model.count){
            sidebarElements.model.get(currentIndex).state_ = "passed";
        }
    }
}
