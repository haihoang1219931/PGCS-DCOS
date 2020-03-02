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
Item {
    id: rootItem
    width: UIConstants.sRect*6*2
    height: UIConstants.sRect*2
    property alias model_: navMenuEles.model
    property alias visible_: rootItem.visible
    property string type
    property int size: 40
    signal listViewClicked(string choosedItem)

    RowLayout {
        id: listView
//        anchors.fill: parent
        anchors.right: parent.right
        anchors.top: parent.top
        anchors.bottom: parent.bottom
        spacing: 2
        Repeater{
            id: navMenuEles
            model: ListModel{
                ListElement{_icon:"\uf024";_btnText:"Flag"}
                ListElement{_icon:"\uf072";_btnText:"PLane"}
                ListElement{_icon:"\uf7d2";_btnText:"Tank"}
                ListElement{_icon:"\uf140";_btnText:"Target"}
                ListElement{_icon:"\uf21a";_btnText:"Ship"}
                ListElement{_icon:"\uf0C0";_btnText:"Military"}
            }
            delegate: FooterButton {
                Layout.preferredHeight: parent.height
                Layout.preferredWidth: parent.height
                icon: _icon
                bgColor: UIConstants.bgAppColor
                btnText: _btnText
                onClicked: {
                    listViewClicked(btnText);
                }
            }
        }


        focus: true
    }
}
