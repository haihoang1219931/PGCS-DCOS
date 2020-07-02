import QtQuick 2.0

//---------------- Include custom libs ----------------------------------------
import CustomViews.Components 1.0
import CustomViews.UIConstants 1.0

Rectangle {
    id: rect_scrollWP
    signal scrollUpWpClicked();
    signal scrollDownWpClicked();

    color: "transparent"
    width: btnScrollUp.width
    height: btnScrollUp.height * 2 + 15

    FlatButtonIcon{
        id: btnScrollUp
        width: UIConstants.sRect*2
        height: UIConstants.sRect*2
        anchors.top: parent.top
        anchors.left: parent.left
        icon: UIConstants.iArrowUp
        iconSize: UIConstants.wpScrollRect
        isSolid: false
        isShowRect: true
        visible: true
        isAutoReturn : true
        onClicked: {
            scrollUpWpClicked()
        }
    }

    FlatButtonIcon{
        id: btnScrollDown
        width: UIConstants.sRect*2
        height: UIConstants.sRect*2
        anchors.bottom : parent.bottom
        anchors.left: parent.left
        icon: UIConstants.iArrowDown
        iconSize: UIConstants.wpScrollRect
        isSolid: false
        isShowRect: true
        visible: true
        isAutoReturn : true
        onClicked: {
            scrollDownWpClicked()
        }
    }
}

