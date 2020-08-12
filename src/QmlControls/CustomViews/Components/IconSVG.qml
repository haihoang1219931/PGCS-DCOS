import QtQuick 2.0
import QtQuick.Window 2.2
import QtGraphicalEffects 1.0

Item{
    id: root
    property alias color: colorOverlay.color
    property int size: 60  // default
    property alias source: img.source

    ShaderEffectSource {
        id: src
        anchors.fill: root
        sourceItem: Image {
            id: img
            anchors.fill: parent
            ColorOverlay {
                id: colorOverlay
                anchors.fill: parent
                source: img
                color: "#000000ff"
            }
        }
        mipmap: true
    }
}
