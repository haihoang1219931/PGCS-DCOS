/**
 * ==============================================================================
 * @Project: VCM01TargetViewer
 * @Component: system time
 * @Breif:
 * @Author: Trung Nguyen
 * @Date: 26/03/2019
 * @Language: QML
 * @License: (c) Viettel Aerospace Institude - Viettel Group
 * ============================================================================
 */

//----------------------- Include QT libs -------------------------------------
//------------------ Include QT libs ------------------------------------------
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
    id: item1
    ColumnLayout {
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.verticalCenter: parent.verticalCenter
        opacity: .7
        Text {
            id: headTitle
            text: "System Time"
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignHCenter
            font {pixelSize: UIConstants.fontSize;}
            font.family: UIConstants.appFont
            color: UIConstants.textFooterColor
            Layout.alignment: Qt.AlignCenter
            Layout.topMargin: 5
        }

        Text {
            id: dateTime
            visible: true
            color: UIConstants.textFooterColor
            smooth: true
            Layout.alignment: Qt.AlignCenter
            text: "10:56:06"
            font {pixelSize: UIConstants.fontSize;}
            font.family: UIConstants.appFont
            function updateDateTime()
            {
                dateTime.text = Qt.formatDateTime(new Date(), "hh:mm:ss");
            }
        }
    }

    Timer {
       id: textTimer
       interval: 1000
       repeat: true
       running: true
       triggeredOnStart: true
       onTriggered: dateTime.updateDateTime()
    }
}
