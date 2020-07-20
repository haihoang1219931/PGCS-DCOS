/**
 * ==============================================================================
 * @Project: FCS-Groundcontrol-based
 * @Module: PreflightCheck page
 * @Breif:
 * @Author: Hai Nguyen Hoang
 * @Date: 14/05/2019
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
import QtQuick 2.0

//------------------ Include Custom modules/plugins
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0
Rectangle{
    id: root
    width: 600
    height: 600
    color: "transparent"
    property var itemListName:
        UIConstants.itemTextMultilanguages["PRECHECK"]["RESULT"]
//    property var model
    property int totalTest: 0
    property int passedTest: 0

    function showResult(model){
        totalTest = 0;
        passedTest = 0;
        for(var i=0; i < model.count-1; i++){
            if(model.get(i).showed_){
                totalTest++;
                if(model.get(i).state_ === "passed"){
                    passedTest++;
                }
            }
        }
    }
//    ListView{
//        id: lstResult
//        width: UIConstants.sRect * 10
//        height: UIConstants.sRect * 1.5 * model.count
//        spacing: 5
//        delegate: Row{
//            width: UIConstants.sRect * 10
//            height: UIConstants.sRect * 1.5
//            visible: showed_ && index < lstResult.model.count-1
//            FlatIcon{
//                anchors.verticalCenter: parent.verticalCenter
//                width: parent.height
//                height: width
//                icon: state_ === "passed"?
//                    UIConstants.iChecked:UIConstants.iClose
//                color: state_ === "passed"?
//                    UIConstants.greenColor:UIConstants.redColor
//            }
//            Label{
//                anchors.verticalCenter: parent.verticalCenter
//                horizontalAlignment: Text.AlignHCenter
//                verticalAlignment: Text.AlignLeft
//                color: UIConstants.textColor
//                font.pixelSize:UIConstants.fontSize
//                font.family: UIConstants.appFont
//                text: text_
//            }
//        }
//    }

    Rectangle {
        id: rectangle
        x: 120
        y: 280
        width: 520
        height: 49
        color: "#00000000"
        anchors.horizontalCenterOffset: 0
        anchors.horizontalCenter: parent.horizontalCenter
        Label {
            id: label
            height: UIConstants.sRect * 2
            text: itemListName["TITTLE"]
                  [UIConstants.language[UIConstants.languageID]]
            font.pixelSize: UIConstants.fontSize
            font.family: UIConstants.appFont
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 0
            color: UIConstants.grayColor
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignHCenter
            anchors.right: parent.right
            anchors.rightMargin: 0
            anchors.left: parent.left
            anchors.leftMargin: 0
        }
    }

    Rectangle {
        width: UIConstants.sRect * 4
        height: width
        color: "#00000000"
        radius: 20
        anchors.horizontalCenterOffset: -UIConstants.sRect * 2.5
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.top: parent.top
        anchors.topMargin: 195
        border.width: 4
        border.color: UIConstants.greenColor
        Label {
            text: passedTest
            font{ pixelSize: parent.width / 2; bold: true; family: UIConstants.appFont }
            color: parent.border.color
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignHCenter
            anchors.fill: parent
        }
        Label{
            text: itemListName["PASSED"]
                  [UIConstants.language[UIConstants.languageID]]
            anchors.top: parent.bottom
            anchors.topMargin: UIConstants.sRect / 2
            anchors.horizontalCenter: parent.horizontalCenter
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignHCenter
            color: parent.border.color
            font.family: UIConstants.appFont
            font.pixelSize: UIConstants.fontSize
        }
    }

    Rectangle {
        width: UIConstants.sRect * 4
        height: width
        color: "#00000000"
        radius: 20
        border.width: 4
        anchors.horizontalCenterOffset: UIConstants.sRect * 2.5
        anchors.horizontalCenter: parent.horizontalCenter
        border.color: UIConstants.redColor
        anchors.top: parent.top
        anchors.topMargin: 195
        Label {
            color: parent.border.color
            text: totalTest - passedTest
            horizontalAlignment: Text.AlignHCenter
            anchors.fill: parent
            font.bold: true
            font.pixelSize: parent.width / 2
            font.family: UIConstants.appFont
            verticalAlignment: Text.AlignVCenter
        }
        Label{
            text: itemListName["FAILED"]
                  [UIConstants.language[UIConstants.languageID]]
            anchors.top: parent.bottom
            anchors.topMargin: UIConstants.sRect / 2
            anchors.horizontalCenter: parent.horizontalCenter
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignHCenter
            color: parent.border.color
            font.family: UIConstants.appFont
            font.pixelSize: UIConstants.fontSize
        }

    }
}
