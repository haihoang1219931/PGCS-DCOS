/****************************************************************************
 *
 *   (c) 2009-2016 QGROUNDCONTROL PROJECT <http://www.qgroundcontrol.org>
 *
 * QGroundControl is licensed according to the terms in the file
 * COPYING.md in the root of the source code directory.
 *
 ****************************************************************************/


import QtQuick          2.3
import QtQuick.Controls 1.2
import QtQuick.Dialogs  1.2
import QtQuick.Layouts 1.3
import QtQuick.Window 2.11
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0
/// ConfigPage
Rectangle {
    id: root
    color: UIConstants.transparentColor
    width: 1376
    height: 768
    property string type: ""
    signal clicked(string type, string action)
    ColumnLayout{
        anchors.topMargin: 8
        anchors.right: parent.right
        anchors.rightMargin: 8
        anchors.top: parent.top
        spacing: 3
        layoutDirection: Qt.RightToLeft
        RowLayout{
            Layout.preferredHeight: UIConstants.sRect * 2
            Layout.fillWidth: true
            layoutDirection: Qt.RightToLeft
            FlatButtonIcon{
                Layout.preferredWidth: UIConstants.sRect * 2
                Layout.preferredHeight: UIConstants.sRect * 2
                width: UIConstants.sRect * 2
                height: width
                icon: mainWindow.visibility === ApplicationWindow.FullScreen?
                          UIConstants.iWindowsMinimize:UIConstants.iWindowStore
                isSolid: true
                isShowRect: false
                iconSize: UIConstants.sRect*3/2 - 5
                iconColor: UIConstants.textColor
                color: UIConstants.transparentBlue
                radius: UIConstants.rectRadius
                Layout.alignment: Qt.AlignRight | Qt.AlignVCenter
                isAutoReturn: true
                onClicked: {
                    if(mainWindow.visibility === ApplicationWindow.FullScreen)
                        mainWindow.visibility = ApplicationWindow.Maximized;
                    else
                        mainWindow.visibility = ApplicationWindow.FullScreen;
                }
            }
            Label{
                font.family: UIConstants.appFont
                font.pixelSize: UIConstants.fontSize
                text: mainWindow.visibility === ApplicationWindow.FullScreen?
                          "Minize windows":"Full windows"
                color: UIConstants.textColor
            }
        }
        RowLayout{
            Layout.preferredHeight: UIConstants.sRect * 2
            Layout.fillWidth: true
            layoutDirection: Qt.RightToLeft
            FlatButtonIcon{
                Layout.preferredWidth: UIConstants.sRect * 2
                Layout.preferredHeight: UIConstants.sRect * 2
                width: UIConstants.sRect * 2
                height: width
                icon: UIConstants.iClose
                isSolid: true
                isShowRect: false
                iconSize: UIConstants.sRect*3/2 - 5
                iconColor: UIConstants.textColor
                color: UIConstants.redColor
                radius: UIConstants.rectRadius
                Layout.alignment: Qt.AlignRight | Qt.AlignVCenter
                isAutoReturn: true
                onClicked: {
                    root.clicked(root.type,"QUIT_APP");
                }
            }
            Label{
                font.family: UIConstants.appFont
                font.pixelSize: UIConstants.fontSize
                text: "Quit Application"
                color: UIConstants.textColor
            }
        }
        RowLayout{
            Layout.fillWidth: true
            Layout.preferredHeight: UIConstants.sRect * 2
            layoutDirection: Qt.RightToLeft
            FlatButtonIcon{
                Layout.preferredWidth: UIConstants.sRect * 2
                Layout.preferredHeight: UIConstants.sRect * 2
                width: UIConstants.sRect * 2
                height: width
                icon: UIConstants.iPowerOff
                isSolid: true
                isShowRect: false
                iconSize: UIConstants.sRect*3/2 - 5
                iconColor: UIConstants.textColor
                color: UIConstants.redColor
                radius: UIConstants.rectRadius
                Layout.alignment: Qt.AlignRight | Qt.AlignVCenter
                isAutoReturn: true
                onClicked: {
                    root.clicked(root.type,"QUIT_COM");
                }
            }
            Label{
                font.family: UIConstants.appFont
                font.pixelSize: UIConstants.fontSize
                text: "Shutdown Computer"
                color: UIConstants.textColor
            }
        }
    }


} // ConfigPage



