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

Item {
    id: rootItem

    //--------- Properties
    property alias textSide: textSide.text
    property int eleId
    property string stateE: "uncheck"
    property alias icon: iconSide.text

    //--------- Signals
    signal activeSide(real eleId_)

    //--------- Element children
    Rectangle {
        id: sideEleWrapper
        anchors.fill: parent
        color: UIConstants.transparentColor
        RowLayout {
            opacity: 0.8
            anchors.fill: parent
            spacing: 15

            Rectangle {
                color: UIConstants.transparentColor
                Layout.fillWidth: true
                Layout.minimumWidth: 30
                Layout.preferredWidth: 40
                Layout.minimumHeight: parent.height
                Layout.alignment: Qt.AlignVCenter
                Layout.leftMargin: 10

                Text {
                    id: iconSide
                    visible: true
                    text: (stateE === "uncheck") ? UIConstants.iInfo : ( (stateE === "pass") ? UIConstants.iChecked : UIConstants.iX )
                    font { pixelSize: UIConstants.fontSize; bold: true;family: ExternalFontLoader.solidFont }
                    color: textSide.color
                    anchors.centerIn: parent
                    RotationAnimation on rotation {
                        id: iconLoadingRotate
                        from: 0
                        to: 360
                        duration: 1000
                        loops: Animation.Infinite
                        running: false
                        easing.type: Easing.Linear
                    }
                }
            }

            Rectangle {
                color: UIConstants.transparentColor
                Layout.fillWidth: true
                Layout.minimumWidth: 100
                Layout.preferredWidth: 200
                Layout.minimumHeight: parent.height
                Layout.alignment: Qt.AlignVCenter

                Text {
                    id: textSide
                    text: "Lưu trữ"
                    font {pixelSize: UIConstants.fontSize}
                    font.family: UIConstants.appFont
                    color: UIConstants.textFooterColor
                    anchors.left: parent.left
                    anchors.verticalCenter: parent.verticalCenter
                }
            }
        }
        RectBorder {
            thick: 1
            type: "bottom"
        }
    }

    //---- Js supported functions
    function setActive()
    {
        sideEleWrapper.color = UIConstants.sidebarActiveBg;
        textSide.color = UIConstants.textColor;
    }

    function setDeactive()
    {
        sideEleWrapper.color = UIConstants.transparentColor;
        textSide.color = UIConstants.textFooterColor;
    }

    function doCheck()
    {
        stateE = "pass";
//        iconSide.text = UIConstants.iSuccess
//        iconSide.text = UIConstants.iLoading
//        iconLoadingRotate.start();
    }
}
