/**
 * ==============================================================================
 * @Project: VCM01TargetViewer
 * @Module: Rectangle Border
 * @Breif:
 * @Author: Trung Nguyen
 * @Date: 18/02/2019
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

//---------------- Component content
Item {
    id: root
    anchors.fill: parent
    property string type: "top"
    property string color: UIConstants.sidebarBorderColor
    property int thick: 2
    Canvas {
        id: canvs
        anchors.fill: parent
        onPaint: {
            var ctx = getContext("2d");
            ctx.strokeStyle = root.color;
            ctx.lineWidth = root.thick;
            ctx.beginPath();
            switch(root.type) {
                case "top": { ctx.moveTo(0, 0); ctx.lineTo(width, 0);break; }
                case "right": { ctx.moveTo(width, 0); ctx.lineTo(width, height); break;}
                case "left" : { ctx.moveTo(0, 0); ctx.lineTo(0, height); break; }
                case "bottom": { ctx.moveTo(0, height); ctx.lineTo(width, height); break; }
            }
            ctx.closePath();
            ctx.stroke();
        }
    }

    function requestPaint()
    {
        canvs.requestPaint();
    }
}
