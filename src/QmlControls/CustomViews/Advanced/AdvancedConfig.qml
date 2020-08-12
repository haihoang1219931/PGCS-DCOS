import QtQuick 2.9
import QtQuick.Controls 2.2
import QtQuick.Layouts 1.3
//------------------ Include Custom modules/plugins
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0
Rectangle {
    id: root
    color: UIConstants.bgAppColor
    width: 650
    height:500
    border.color: "gray"
    property var buttonListName:
    {
        "TAB_CONFIG_EO":["Config EO","Ảnh ngày"],
        "TAB_CONFIG_IR":["Config IR","Ảnh nhiệt"],
        "TAB_CONFIG_VIDEO":["Video","Hiển thị"],
        "TAB_CONFIG_SEARCH":["Search","Tìm kiếm"],
        "TAB_CONFIG_MOTIONC":["MotionC","MotionC"],
    }
//    MouseArea{
//        anchors.fill: parent
//        hoverEnabled: true
//    }
    MouseArea{
        hoverEnabled: true
        anchors.top: tabBar.bottom
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
    }
    StackLayout {
        id: swipeView
        anchors.top: tabBar.bottom
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        currentIndex: tabBar.currentIndex
        ConfigEO{
        }
        ConfigIR{
        }
        ConfigOverlay{
        }
        ConfigMotionCParams{
        }
        ConfigSearch{
        }
    }
    Rectangle{
        color: UIConstants.bgAppColor
        anchors.bottom: tabBar.bottom
        anchors.bottomMargin: 0
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.right: parent.right
        border.color: "gray"
        border.width: 1
    }
//    MouseArea {
//        x: 1
//        y: 1
//        width: parent.width
//        height: 50
//        //! [drag]
//        drag.target: root
//        drag.axis: Drag.XAndYAxis
//        drag.minimumX: 0
//        drag.minimumY: 0
//    }
    RowLayout{
        id: tabBar
        x: 1
        y: 1
//        anchors.fill: parent
//        color: "transparent"
        height: 47
        spacing: 0
        property int currentIndex
        property int count:navMenuEles.model.count
        onCurrentIndexChanged: {
            for(var i = 0; i < navMenuEles.count; i++ )
            {
                if( i != currentIndex)
                {
                    navMenuEles.itemAt(i).active = false;
                    navMenuEles.itemAt(i).setInactive();
                }else{
                    navMenuEles.itemAt(i).active = true;
                    navMenuEles.itemAt(i).setActive();
                }
            }
        }
      Repeater {
            id: navMenuEles
            model: ListModel {
                Component.onCompleted: {
                    append({id_: 0, btnText_: buttonListName["TAB_CONFIG_EO"][camState.language[camState.fd_icon]],
                               icon_: "\uf185", active_: true ,info_:"config_eo"});
                    append({id_: 1, btnText_: buttonListName["TAB_CONFIG_IR"][camState.language[camState.fd_icon]],
                               icon_: "\uf186", active_: false ,info_:"config_ir" });
                    append({id_: 2, btnText_: buttonListName["TAB_CONFIG_VIDEO"][camState.language[camState.fd_icon]],
                               icon_: "\uf02b", active_: false ,info_:"video" });
                    append({id_: 3, btnText_: buttonListName["TAB_CONFIG_MOTIONC"][camState.language[camState.fd_icon]],
                               icon_: "\uf0b2", active_: false ,info_:"motionc" });
                    append({id_: 4, btnText_: buttonListName["TAB_CONFIG_SEARCH"][camState.language[camState.fd_icon]],
                               icon_: "\uf4fe", active_: false ,info_:"other-functions" });

                }
            }

            FlatButton {
                btnText: btnText_
                btnTextColor: UIConstants.textFooterColor
                Layout.preferredHeight: parent.height
                Layout.preferredWidth: width
                iconVisible: true
                icon: icon_
                color: active_ ? UIConstants.sidebarActiveBg : UIConstants.transparentColor
                radius: 5
                active: tabBar.currentIndex === idx
                idx: id_
                onClicked: {
                    tabBar.currentIndex = idx;
                }
            }
        }
    }
//    TabBar {
//        id: tabBar
//        x: 1
//        y: 1
//        currentIndex: swipeView.currentIndex
//        spacing: 1
//        TabButton {
//            width: implicitWidth
//            text: qsTr("Config EO")
//            font.bold: true
//        }
//        TabButton {
//            width: implicitWidth
//            text: qsTr("Config IR")
//            font.bold: true
//        }
//        TabButton {
//            width: implicitWidth
//            text: qsTr("Object Search")
//            font.bold: true
//        }
//    }
    Button{
        id: btnCancel
        x: 600
        width: 49
        height: 47
        text: "x"
        anchors.top: parent.top
        anchors.topMargin: 1
        anchors.right: parent.right
        anchors.rightMargin: 1
        onClicked: {
            configPane.visible = false
        }
    }
}

/*##^## Designer {
    D{i:13;anchors_y:250}
}
 ##^##*/
