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
    property int currentItem: 1

    //-------------- signal
    signal displayActiveConfigBoard( real boardId_ )

    //------------------- Set background
    color: UIConstants.transparentColor

    //-------------- Sidebar title
    SidebarTitle {
        id: sidebarTitle
        anchors { top: parent.top; left: parent.left; right: parent.right }
        height: 70
        visible: true
        title: "Check List"
    }

    //--------------- Sidebar list / Sidebar content
    Column {
        anchors { top: sidebarTitle.bottom; left: parent.left; right: parent.right; bottom: parent.bottom }
        //--- Sidebar Elements
        Repeater {
            id: sidebarElements
            anchors.fill: parent
            model: ListModel {
                id: listElesData
                Component.onCompleted: {
                    append({id_: 1, state_: "uncheck", text_: "ModeCheck" });
                    append({id_: 2, state_: "uncheck", text_: "Propellers" });
                    append({id_: 3, state_: "uncheck", text_: "Steering" });
                    append({id_: 4, state_: "uncheck", text_: "Pitot" });
                    append({id_: 5, state_: "uncheck", text_: "Laser" });
                    append({id_: 6, state_: "uncheck", text_: "GPS" });
                    append({id_: 7, state_: "uncheck", text_: "RPM" });
                    append({id_: 8, state_: "uncheck", text_: "Payload" });
                }
            }
            CheckListEle {
                width: parent.width
                height: 60
                stateE: state_
                textSide: text_
                eleId: id_
                onActiveSide: {
                    currentItem = eleId_;
                    rootItem.displayActiveConfigBoard( eleId_ - 1);
                    for( var i = 0; i < listElesData.count; i++ )
                    {
                        if( i == eleId_ - 1 )
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
        if( currentItem == (listElesData.count+1) )
            return
        currentItem = currentItem + 1;
        rootItem.displayActiveConfigBoard( currentItem - 1);
        for( var i = 0; i < listElesData.count; i++ )
        {
            if( i == currentItem - 1 )
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
        if( currentItem == 1 )
            return

        currentItem = currentItem - 1;
        rootItem.displayActiveConfigBoard( currentItem - 1);
        for( var i = 0; i < listElesData.count; i++ )
        {
            if( i == currentItem - 1 )
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
        if(currentItem - 1 < listElesData.count){
            sidebarElements.itemAt(currentItem - 1).doCheck();
            sidebarMouseArea.hoverEnabled = true;
            sidebarMouseArea.preventStealing = true;
            sidebarMouseArea.enabled = !sidebarMouseArea.enabled
        }
    }
}
