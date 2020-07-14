/**
 * ==============================================================================
 * @Project: GCS-FCS
 * @Component: Flat Button
 * @Breif:
 * @Author: Hai Nguyen Hoang
 * @Date: 17/05/2019
 * @Language: QML
 * @License: (c) Viettel Aerospace Institude - Viettel Group
 * ============================================================================
 */

//----------------------- Include QT libs -------------------------------------
import QtQuick.Window 2.2
import QtQuick 2.6
import QtQuick.Controls 2.1
import QtQuick.Layouts 1.3
import QtGraphicalEffects 1.0
import QtQuick 2.0

//---------------- Include custom libs ----------------------------------------
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0

Item {
    id: rootItem
    width:  100
    height: 80
    //--- Properties
    property bool isActive: false
    property bool isNormal: true
    property alias topValue: lblTopValue.text
    property alias middleValue: lblMiddleValue.text
    property alias bottomValue: lblBottomValue.text
    property bool isEnable: true
    property bool fontBold: false
    property int fontSize: UIConstants.fontSize * 3 / 2
    property var midColor:  UIConstants.textFooterValueColor

    property var middleColor: UIConstants.textFooterValueColor
    property var disableColor: UIConstants.textFooterValueColorDisable

    //--- Signals
    signal clicked()

    //--- Button wrapper
    Rectangle {
        anchors.fill: parent
        color: UIConstants.transparentColor
        radius: UIConstants.rectRadius
        ColumnLayout {
            width: parent.width
            height: parent.height
            spacing: 0

            //--- Top value
            Label {
                id: lblTopValue
                text: "Label"
                horizontalAlignment: Text.AlignLeft
                verticalAlignment: Text.AlignVCenter
                Layout.alignment: Qt.AlignCenter
                Layout.preferredHeight: parent.height/3
                font.pixelSize: rootItem.fontSize
                font.bold: rootItem.fontBold
                font.family: UIConstants.appFont
                color: isEnable ? UIConstants.textFooterColor : UIConstants.cDisableColor
            }

            //--- Middle value
            Label {
                id: lblMiddleValue
                text: "Label"
                verticalAlignment: Text.AlignVCenter
                Layout.alignment: Qt.AlignCenter
                Layout.preferredHeight: parent.height/3
                font.pixelSize: rootItem.fontSize
                font.bold: rootItem.fontBold
                font.family: UIConstants.appFont
                color: isEnable ? middleColor : disableColor
            }

            //--- Bottom value
            Label {
                id: lblBottomValue
                text: "Label"
                verticalAlignment: Text.AlignVCenter
                Layout.alignment: Qt.AlignCenter
                Layout.preferredHeight: parent.height/3
                font.pixelSize: rootItem.fontSize
                font.bold: rootItem.fontBold
                font.family: UIConstants.appFont
                color: isEnable ? UIConstants.textFooterColor : UIConstants.cDisableColor
            }
        }
    }

    //--- Button background
    Rectangle {
        id: btnBackground
        anchors.fill: parent
        color: isEnable ? UIConstants.cateOverlayBg : UIConstants.cDisableColor
        opacity: .2
        radius: UIConstants.rectRadius
//        radius: 5
        //--- Click event
        MouseArea {
            id: btnSelectedArea
            anchors.fill: parent
            enabled: isEnable ? true : false
            onPressed: {
                parent.opacity = 0.5;
                rootItem.clicked();
            }

            onReleased: {
                parent.opacity = 0.2;
            }
        }
    }

    //--- Js supported functions
    function setButtonActive()
    {
        btnBackground.color = UIConstants.info
        isActive = true
        isNormal = false
    }

    function setButtonNormal()
    {
        btnBackground.color = UIConstants.cateOverlayBg;
        textBtn.color = UIConstants.textFooterColor
        isNormal = true
        isActive = false
    }

    function setButtonDisable()
    {
        btnSelectedArea.hoverEnabled = true;
        btnSelectedArea.preventStealing = true;
        btnSelectedArea.enabled = false;
        btnBackground.color = UIConstants.cDisableColor
        isEnable = false;
    }
}
