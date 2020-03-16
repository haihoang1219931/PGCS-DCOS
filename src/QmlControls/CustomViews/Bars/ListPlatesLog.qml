import QtQuick 2.7
import QtQuick 2.2
import QtQuick.Dialogs 1.1
import QtQuick.Controls 2.0
import QtQuick.Controls 1.4
import QtQuick.Layouts 1.0
import QtQuick.Layouts 1.3
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0
import io.qdt.dev 1.0
Rectangle {
    id: root
    width: UIConstants.sRect * 30
    height: UIConstants.sRect * 16
    color: UIConstants.transparentBlue
    radius: 3
    border.width: 1
    border.color: UIConstants.grayColor
    property alias plateLog: plateLog
    function addPlate(time,plateNumber,plateSourceImage){
        lstPlateLog.model.append({
                    "time": time,
                    "plateNumber":plateNumber,
                    "plateSourceImage":plateSourceImage
                    });
    }
    function addPlateLogLine(logLine){
        var items = logLine.split(",");
        var time = items[0];
        var plateNumber = items[1];
        var plateSourceImage = items[2];
        addPlate(time,plateNumber,plateSourceImage);
    }

    function loadPlateTimeline(plateNumber,plateSourceImage){
        lblPlateID01.text = plateNumber;
        lblPlateID02.text = plateNumber;
        var imgPlateSource = "file://"+applicationDirPath+"/plates/"+plateSourceImage;
//        console.log("imgPlateSource = "+imgPlateSource);
        imgPlate.source = imgPlateSource;
        lstPlateTimeline.model.clear();
        for(var i=0; i<lstPlateLog.model.count; i++){
            if(lstPlateLog.model.get(i).plateNumber === plateNumber){
                lstPlateTimeline.model.append({
                    "time":lstPlateLog.model.get(i).time,
                    "plateSourceImage":lstPlateLog.model.get(i).plateSourceImage
                    });
            }
        }
    }
    PlateLog{
        id: plateLog
        onPlateReaded: {
            addPlateLogLine(logLine);
        }
    }

    MouseArea{
        anchors.fill: parent
        drag.target: parent
        drag.axis: Drag.XAndYAxis
        drag.minimumX: 0
        drag.minimumY: 0
    }
    Rectangle{
        id: headerPlatesLog
        width: UIConstants.sRect * 10
        height: UIConstants.sRect
        color: UIConstants.bgAppColor
        anchors.top: parent.top
        anchors.topMargin: 8
        anchors.right: parent.right
        anchors.rightMargin: 8
        RowLayout{
            anchors.fill: parent
            spacing: 1
            Label{
                Layout.preferredWidth: UIConstants.sRect*4
                Layout.preferredHeight: parent.height
                text: "Plate"
                color: UIConstants.textColor
                verticalAlignment: Text.AlignVCenter
                horizontalAlignment: Text.AlignHCenter
                font.family: UIConstants.appFont
                font.pixelSize: UIConstants.fontSize
            }
            Label{
                Layout.preferredWidth: UIConstants.sRect*6
                Layout.preferredHeight: parent.height
                text: "Date time"
                color: UIConstants.textColor
                verticalAlignment: Text.AlignVCenter
                horizontalAlignment: Text.AlignHCenter
                font.family: UIConstants.appFont
                font.pixelSize: UIConstants.fontSize
            }
        }
    }

    ListView {
        id: lstPlateLog
        width: UIConstants.sRect * 10
        anchors.top: headerPlatesLog.bottom
        anchors.topMargin: 8
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 8
        anchors.right: parent.right
        anchors.rightMargin: 8
        clip: true
        model: ListModel {
        }
        delegate: Rectangle {
            id: rectPlateLog
            width: parent.width
            height: visible?30:0
            visible: txtPlateSearch.text === "" || plateNumber.includes(txtPlateSearch.text.toUpperCase())
            color: UIConstants.transparentColor
            MouseArea{
                anchors.fill: parent
                hoverEnabled: true
                onEntered: {
                   parent.opacity = 0.8
                }
                onExited: {
                    parent.opacity = 1
                }

                onClicked: {
                    loadPlateTimeline(plateNumber,plateSourceImage)
                }
            }

            RowLayout{
                anchors.fill: parent
                spacing: 1
                Rectangle{
                    Layout.preferredWidth: UIConstants.sRect*4
                    Layout.preferredHeight: parent.height - 8
                    color: UIConstants.textColor
                    Label{
                        anchors.fill: parent
                        text: plateNumber
                        color: UIConstants.blackColor
                        verticalAlignment: Text.AlignVCenter
                        horizontalAlignment: Text.AlignHCenter
                        font.family: UIConstants.appFont
                        font.pixelSize: UIConstants.fontSize
                    }
                }

                Label{
                    Layout.preferredWidth: UIConstants.sRect*6
                    Layout.preferredHeight: parent.height
                    text: time
                    color: UIConstants.textColor
                    verticalAlignment: Text.AlignVCenter
                    horizontalAlignment: Text.AlignHCenter
                    font.family: UIConstants.appFont
                    font.pixelSize: UIConstants.fontSize
                }
            }
        }
    }

    Rectangle {
        id: rectInfomation
        color: UIConstants.transparentColor
        anchors.right: lstPlateLog.left
        anchors.rightMargin: 8
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 8
        anchors.left: parent.left
        anchors.leftMargin: 8
        anchors.top: parent.top
        anchors.topMargin: 8

        Rectangle {
            id: rectSearch

            width: UIConstants.sRect * 6
            height: UIConstants.sRect
            color: UIConstants.transparentColor
            clip: true

            TextField {
                id: txtPlateSearch
                text: qsTr("")
                placeholderText: "License number"
                font.capitalization: Font.AllUppercase
                anchors.fill: parent
                font.family: UIConstants.appFont
                font.pixelSize: UIConstants.fontSize
            }
        }
        FlatButtonText {
            id: btnReload
            anchors.left: rectSearch.right
            anchors.leftMargin: 8
            width: UIConstants.sRect * 4
            height: UIConstants.sRect
            text: qsTr("Read log")
            isAutoReturn: true
            color: UIConstants.transparentBlueDarker
            border.color: UIConstants.grayColor
            border.width: 1
            onClicked: {
                lstPlateLog.model.clear();
                plateLog.readLogFile("plates/plate_log.csv");
            }
        }
        Rectangle {
            id: rectImage
            color: UIConstants.transparentColor
            anchors.bottom: rectPlate.top
            anchors.bottomMargin: 8
            anchors.top: parent.top
            anchors.topMargin: 58
            anchors.right: parent.right
            anchors.rightMargin: 8
            anchors.left: parent.left
            anchors.leftMargin: 8
            Rectangle{
                id: rectHeaderPlate
                width: UIConstants.sRect * 6
                height: UIConstants.sRect
                anchors.right: parent.right
                anchors.rightMargin: 8
                anchors.top: parent.top
                anchors.topMargin: 0
                color: UIConstants.transparentBlueDarker
                Label {
                    id: label
                    anchors.fill: parent
                    text: qsTr("Time line")
                    verticalAlignment: Text.AlignVCenter
                    horizontalAlignment: Text.AlignHCenter
                    color: UIConstants.textColor
                    font.family: UIConstants.appFont
                    font.pixelSize: UIConstants.fontSize
                }
            }

            ListView {
                id: lstPlateTimeline
                width: UIConstants.sRect * 6
                anchors.bottom: parent.bottom
                anchors.bottomMargin: 8
                anchors.right: parent.right
                anchors.rightMargin: 8
                anchors.top: rectHeaderPlate.bottom
                anchors.topMargin: 8
                model: ListModel {

                }
                delegate: Rectangle {
                    width: parent.width
                    height: 20
                    color: UIConstants.transparentBlue
                    MouseArea{
                        anchors.fill: parent
                        hoverEnabled: true
                        onEntered: {
                           parent.opacity = 0.8
                        }
                        onExited: {
                            parent.opacity = 1
                        }

                        onClicked: {
                            var imgPlateSource = "file://"+applicationDirPath+"/plates/"+plateSourceImage;
//                            console.log("imgPlateSource = "+imgPlateSource);
                            imgPlate.source = imgPlateSource;
                        }
                    }

                    Label{
                        anchors.fill: parent
                        text: time
                        color: UIConstants.orangeColor
                        verticalAlignment: Text.AlignVCenter
                        horizontalAlignment: Text.AlignHCenter
                        font.family: UIConstants.appFont
                        font.pixelSize: UIConstants.fontSize
                    }
                }
            }

            Rectangle {
                id: rectangle6
                color: UIConstants.transparentColor
                border.width: 2
                border.color: UIConstants.orangeColor
                radius: UIConstants.rectRadius
                anchors.bottom: parent.bottom
                anchors.bottomMargin: 8
                anchors.right: rectHeaderPlate.left
                anchors.rightMargin: 8
                anchors.left: parent.left
                anchors.leftMargin: 0
                anchors.top: parent.top
                anchors.topMargin: 0

                Image {
                    id: imgPlate
                    x: 8
                    y: 8
                    anchors.right: parent.right
                    anchors.rightMargin: 8
                    anchors.left: parent.left
                    anchors.leftMargin: 8
                    anchors.bottom: parent.bottom
                    anchors.bottomMargin: 8
                    anchors.top: parent.top
                    anchors.topMargin: 8
                }
            }
        }

        Rectangle {
            id: rectPlate
            height: UIConstants.sRect * 4
            color: UIConstants.transparentColor
            border.width: 1
            border.color: UIConstants.grayColor
            radius: UIConstants.rectRadius
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 8
            anchors.right: parent.right
            anchors.rightMargin: 8
            anchors.left: parent.left
            anchors.leftMargin: 8

            Rectangle {
                id: rectPlateNumberColor
                width: UIConstants.sRect * 9
                height: UIConstants.sRect * 3
                color: UIConstants.textColor
                anchors.left: parent.left
                anchors.leftMargin: UIConstants.sRect
                anchors.verticalCenter: parent.verticalCenter

                Label {
                    id: lblPlateID01
                    text: qsTr("XXX-XXXXX")
                    font.pointSize: 20
                    anchors.fill: parent
                    verticalAlignment: Text.AlignVCenter
                    horizontalAlignment: Text.AlignHCenter
                    font.family: UIConstants.appFont
                    font.pixelSize: UIConstants.fontSize
                }
            }
            ColumnLayout{
                id: rectOtherInfo
                anchors.top: parent.top
                anchors.right: parent.right
                anchors.bottom: parent.bottom
                anchors.left:rectPlateNumberColor.right
                anchors.leftMargin: UIConstants.sRect
                spacing: 0
                RowLayout{
                    Layout.preferredHeight: parent.height / 3
                    Layout.fillWidth: true

                    Label {
                        text: qsTr("Plate")
                        Layout.preferredWidth: UIConstants.sRect * 3
                        color: UIConstants.textColor
                        font.family: UIConstants.appFont
                        font.pixelSize: UIConstants.fontSize
                    }

                    Rectangle {
                        id: rectangle5
                        width: UIConstants.sRect * 4
                        height: UIConstants.sRect * 1
                        Layout.preferredHeight: height
                        Layout.preferredWidth: width
                        color: UIConstants.transparentColor
                        border.color: UIConstants.grayColor
                        border.width: 1

                        Label {
                            id: lblPlateID02
                            text: qsTr("XXX-XXXXX")
                            anchors.fill: parent
                            color: UIConstants.textColor
                            verticalAlignment: Text.AlignVCenter
                            horizontalAlignment: Text.AlignHCenter
                            font.family: UIConstants.appFont
                            font.pixelSize: UIConstants.fontSize
                        }
                    }
                }
                RowLayout{
                    Layout.preferredHeight: parent.height / 3
                    Layout.preferredWidth: parent.width
                    Label {
                        id: label2
                        Layout.preferredWidth: UIConstants.sRect * 3
                        text: qsTr("Speed")
                        color: UIConstants.textColor
                        font.family: UIConstants.appFont
                        font.pixelSize: UIConstants.fontSize
                    }

                    Label {
                        id: lblSpeed
                        Layout.fillWidth: true
                        color: UIConstants.textColor
                        text: qsTr("0 km/h")
                        font.family: UIConstants.appFont
                        font.pixelSize: UIConstants.fontSize
                    }
                }
                RowLayout{
                    Layout.preferredHeight: parent.height / 3
                    Layout.preferredWidth: parent.width
                    Label {
                        id: label4
                        Layout.preferredWidth: UIConstants.sRect * 3
                        text: qsTr("Count")
                        color: UIConstants.textColor
                        font.family: UIConstants.appFont
                        font.pixelSize: UIConstants.fontSize
                    }

                    Label {
                        id: lblPlateCount
                        Layout.fillWidth: true
                        text: lstPlateTimeline.model.count
                        color: UIConstants.textColor
                        font.family: UIConstants.appFont
                        font.pixelSize: UIConstants.fontSize
                    }
                }
            }
        }


    }
}

/*##^## Designer {
    D{i:6;anchors_height:560;anchors_y:84}
}
 ##^##*/
