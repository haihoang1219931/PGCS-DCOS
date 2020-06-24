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
    property int currentItem: 0
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
    Column {
        anchors { top: sidebarTitle.bottom; left: parent.left; right: parent.right; bottom: parent.bottom }
        //--- Sidebar Elements
        Repeater {
            id: sidebarElements
            anchors.fill: parent            
            delegate: CheckListEle {
                width: parent.width
                height: 60
                stateE: state_
                textSide: text_
                eleId: index
                onActiveSide: {
                    currentItem = eleId_;
                    rootItem.displayActiveConfigBoard( eleId_);
                    for( var i = 0; i < sidebarElements.model.count; i++ )
                    {
                        if( i === eleId_)
                        {
                            sidebarElements.itemAt(i).setActive();
                        }
                        else
                        {
                            sidebarElements.itemAt(i).setDeactive();
                        }
                    }
                }
                //--- Set default active
                Component.onCompleted: {
                    sidebarElements.itemAt(0).setActive();
                }
            }
        }
    }

    //--------------- Mouse Area
    MouseArea {
        id: sidebarMouseArea
        anchors.fill: parent
        enabled: false
        hoverEnabled: false
        preventStealing: false

    }

    //---------------- Js supported functions
    function next()
    {
        if( currentItem + 1 > sidebarElements.model.count ){
            return;
        }else
            currentItem += 1;
        rootItem.displayActiveConfigBoard(currentItem);
        for( var i = 0; i < sidebarElements.model.count; i++ )
        {
            if( i === currentItem)
            {
                sidebarElements.itemAt(i).setActive();
            }
            else
            {
                sidebarElements.itemAt(i).setDeactive();
            }
        }

    }

    function prev()
    {
        if( currentItem -1 < 0 ){
            return;
        }else
            currentItem -= 1;
        rootItem.displayActiveConfigBoard( currentItem);
        for( var i = 0; i < sidebarElements.model.count; i++ )
        {
            if( i === currentItem)
            {
                sidebarElements.itemAt(i).setActive();
            }
            else
            {
                sidebarElements.itemAt(i).setDeactive();
            }
        }
    }

    function doCheck()
    {
        if(currentItem < sidebarElements.model.count){
            sidebarElements.itemAt(currentItem).doCheck();
            sidebarMouseArea.hoverEnabled = true;
            sidebarMouseArea.preventStealing = true;
            sidebarMouseArea.enabled = !sidebarMouseArea.enabled
        }
    }
}
