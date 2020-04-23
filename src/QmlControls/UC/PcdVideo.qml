/**
 * ==============================================================================
 * @Project: UC-FCS
 * @Module: PcdVideo
 * @Breif:
 * @Author: Trung Ng
 * @Date: 26/08/2019
 * @Language: QML
 * @License: (c) Viettel Aerospace Institude - Viettel Group
 * ============================================================================
 */

import QtQuick 2.9
import QtQuick.Window 2.2
import QtQuick.Controls 2.4
import QtQuick.Layouts 1.3
import QtWebEngine 1.1
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0
import io.qdt.dev   1.0
Rectangle {
    id: rootComponent
    color: UIConstants.transparentBlue
    property bool sharing: false
    property string pcdActive: ""
    property var pcdVideo: pcdVideoView
    z: 100001

    //-------- :WebEngineView
    WebEngineView {
        id: pcdVideoView
        //--- Dimension
        anchors.fill: parent

        //--- Attributes
        visible: true
        opacity: 1
        url: Qt.resolvedUrl(UcApiConfig.value("Settings:UCPCDVideoSource:Value:data"))
        //--- Set permision to access video/audio
        onFeaturePermissionRequested: {
            grantFeaturePermission(securityOrigin, feature, true);
        }

        //--- Ignore ssl certificate error
        onCertificateError: error.ignoreCertificateError()

        //--- Message in console.log
        onJavaScriptConsoleMessage: {
            console.log(message);
        }
//        Component.onCompleted: {
//            console.log("pcdVideoView.url:"+pcdVideoView.url);
//        }

        //--- Button group actions
        Row {
            anchors.right: parent.right
            anchors.top: parent.top
             z: 2
            Rectangle {
                id: mediaSharingAction
                width: 30
                height: 30
                color: "#4b6584"
                radius: 1
                border {width: 1; color: "#a5b1c2"}
                z: 2
                Text {
                    text: rootComponent.sharing ? "\uf04c" : "\uf064"
                    color: "#fff"
                    font{ pixelSize: 13;
                        weight: Font.Bold;
                        family: ExternalFontLoader.solidFont}
                    anchors.centerIn: parent
                }
                MouseArea {
                    anchors.fill: parent
                    onClicked: {
                        an1.target = mediaSharingAction;
                        an2.target = mediaSharingAction;
                        clickAnimation.running = true;

                        if( rootComponent.sharing ) {
                            UcApi.stopSharePcdVideoFromRoom(rootComponent.pcdActive);
//                            rootComponent.stopSharePcdVideoFromRoom(rootComponent.pcdActive);
                        }
                        else {
                            UcApi.sharePcdVideoToRoom(rootComponent.pcdActive);
//                            rootComponent.sharePcdVideoToRoom(rootComponent.pcdActive);
                        }

                    }
                }
            }
            Rectangle {
                id: closeWindow
                width: 30
                height: 30
                color: "#4b6584"
                radius: 1
                border {width: 1; color: "#a5b1c2"}
                z: 2
                Text {
                    text: UIConstants.iChatClose
                    color: "#fff"
                    font{ pixelSize: 13;
                        weight: Font.Bold;
                        family: ExternalFontLoader.solidFont}
                    anchors.centerIn: parent
                }
                MouseArea {
                    anchors.fill: parent
                    onClicked: {
                        an1.target = closeWindow;
                        an2.target = closeWindow;
                        clickAnimation.running = true;
                        rootComponent.visible = false;
                        rootComponent.sharing = false;
                        UcApi.closePcdVideo(rootComponent.pcdActive);
                    }
                }
            }
            SequentialAnimation {
                id: clickAnimation
                running: false
                NumberAnimation {
                    id: an1
                    duration: 100
                    properties: "opacity"
                    to: 0.5
                }

                NumberAnimation {
                    id: an2
                    duration: 100
                    properties: "opacity"
                    to: 1
                }
            }

            PropertyAnimation {
                targets: pcdVideoView
                properties: "z"
                duration: 200
                easing.type: Easing.InExpo
            }
        }
    }

    function reload() {
        pcdVideoView.reload();
    }
}
