/**
 * ==============================================================================
 * @Project: VCM01TargetViewer
 * @Module: UIConstants
 * @Breif:
 * @Author: Trung Nguyen
 * @Date: 13/02/2019
 * @Language: QML
 * @License: (c) Viettel Aerospace Institude - Viettel Group
 * ============================================================================
 */

pragma Singleton
import QtQuick 2.0
import QtQuick.Window 2.11
QtObject {
    //---------------------- Size of smallest rect  --------------------
    property int sRect:                    Screen.pixelDensity*5.5
    //------------------------------ App Font --------------------------
    property bool isMobile:                         false
    property int fontSize:                          Screen.pixelDensity*3.5
    property string appFont:                        "monospace"
    property int rectRadius:                        5
    property color transparentRed:                  "#b3ff0000"
    property color transparentGreen:                "#b300FF5e"
    property color transparentBlue:                 "#b334495e"
    property color transparentBlueDarker:           "#e334495e"
    property color transparentColor:                "transparent"
    property color boundColor:                      "gray"
    property string degreeSymbol :                  "\u00B0"
    //------------------------------ Flight Mode -----------------------
    property string monitorMode:                    monitorModeFlight // "MISSION","FLIGHT"
    property string monitorModeFlight:              "FLIGHT"
    property string monitorModeMission:             "MISSION"
    property string layoutMaxPane:                  layoutMaxPaneMap // "VIDEO","MAP"
    property string layoutMaxPaneMap:               "MAP"
    property string layoutMaxPaneVideo:             "VIDEO"
    property string mouseOnMapMode:                 mouseOnMapModeWaypoint
    property string mouseOnMapModeWaypoint:         "WAYPOINT" // ""
    property string mouseOnMapModeMeasure:          "MEASURE" // ""
    //------- Background
    property color bgPanelColor:                    "#283441"
    property color bgAppColor:                      "#34495e"
    property color bgColorOverlay:                  "#222f3e"
    property color sidebarBgColor:                  "#34495e"
    property color flatButtonColor:                 "#3c6382"
    property color categoryEleBgColor:              "#4b6584"
    property color categoryCirColor:                "#4b6584"
    property color cateOverlayBg:                   "#808e9b"
    property color sidebarBgItemLevel1:             "#2B3E52"
    property color sidebarBgItemLevel2:             "#1D2833"
    property color cfProcessingOverlayBg:           "#2D3436"
    property color sidebarConfigBg:                 "#2D3C4B"
    property color sidebarActiveBg:                 "#283441"

    //------------------ Notification type messages color --------------
    readonly property color success:                "#1dd1a1"
    readonly property color error:                  "#ff7675"
    readonly property color warning:                "#fdcb6e"
    readonly property color info:                   "#0984e3"

    //------------------ Icons notification -------------------------
    readonly property string iChecked:              "\uf00c"
    readonly property string iInfoCircle:           "\uf05a"
    readonly property string iLoading:              "\uf110"
    readonly property string iArrow:                "\uf124"
    readonly property string iError:                "\uf06a"

    //------------------- Nav / Categrory Icons --------------------
    readonly property string iHomepage:             "\uf015"
    readonly property string iLiveView:             "\uf03d"
    readonly property string iPlayback:             "\uf144"
    readonly property string iDeviceMgmt:           "\uf109"
    readonly property string iGeneralConfigs:       "\uf085"
    readonly property string iBullhorn:             "\uf0a1"
    readonly property string iSatellite:            "\uf7bf"
    readonly property string iTrash:                "\uf2ed"

    //------------------- Sidebar Icons ---------------------------
    readonly property string iList:                 "\uf0ca"
    readonly property string iDevice:               "\uf7d9"
    readonly property string iCaretDown:            "\uf0d7"
    readonly property string iCaretLeft:            "\uf0d9"
    readonly property string iLeftHand:             "\uf0a5"
    readonly property string iRightHand:            "\uf0a4"
    readonly property string iSearch:               "\uf00e"

    //------------------- Live View icons -------------------------
    readonly property string iSetting:              "\uf013"
    readonly property string iUpload:               "\uf093"
    readonly property string iVoiceControl:         "\uf028"
    readonly property string iX:                    "\uf00d"
    readonly property string iRocketWarning:        "\uf071"
    readonly property string iBandwidth:            "\uf362"
    readonly property string iResolution:           "\uf26c"
    readonly property string iWindowGrid:           "\uf17a"

    //------------------- Playback icons -------------------------
    readonly property string iFiles:                "\uf0c5"
    readonly property string iDoubleLeft:           "\uf100"
    readonly property string iDoubleRight:          "\uf101"
    readonly property string iPlayState:            "\uf7a5"
    readonly property string iStopState:            "\uf04b"
    readonly property string iChoosedRight:         "\uf061"

    //-------------------General configs icons ------------------
    readonly property string iStorage:              "\uf0a0"
    readonly property string iDetectParams:          "\uf11c"
    readonly property string iSequenceTasks:         "\uf0ae"
    readonly property string iUsers:                 "\uf0c0"

    //------------------- Device Mgmt Icons ----------------------
    readonly property string iSave:                 "\uf0c7"

    //------------------- Preflight check Icons ----------------------
    readonly property string iSuccess:              "\uf00c"

    //------------------ Usual Color -------------------------------
    //-------Text
    readonly property color textColor:             "#fff"
    readonly property color textFooterColor:       "#b2bec3"
    readonly property color textFooterValueColor:  "#e23636"
    readonly property color textFooterValueColorDisable:  "#d15757"
    readonly property color cateCirTextColor:      "#778ca3"
    readonly property color cateDescTextColor:     "#CAD3C8"
    readonly property color textBlueColor:         "#0a3d62"
    readonly property color textSidebarColor:      "#ADACAA"
    readonly property color cfWarningColor:        "#ffd32a"
    readonly property color tableHeaderColor:      "#3c6382"
    readonly property color blackColor:            "#2f3640"
    readonly property color textSideActive:        "#23B99A"
    //------- Button
    readonly property color btnCateColor:          "#CAD3C8"
    readonly property color btnSelectedColor:      "#38729A"
    //------- Shadow
    readonly property color dropshadowColor:       "#57606f"
    readonly property color headerTxtShadowColor:  "#3c6382"
    //------- Border
    readonly property color sidebarBorderColor:    "#636e72"
    readonly property color borderCircleBtnColor:  "#718093"
    readonly property color sidebarHeaderBorderColor: "#c8d6e5"
    readonly property color liveViewGridHighlight:  "#487eb0"
    readonly property color borderGreen:            "#0984e3"

    //------- Button
    readonly property color savingBtn:             "#487eb0"
    //------- Others
    readonly property color grayColor:             "gray"
    readonly property color greyColor:             "grey"
    readonly property color blueColor:             "blue"
    readonly property color orangeColor:             "orange"
    readonly property color redColor:             "red"
    readonly property color grayLighterColor:      "#808080"
    readonly property color greenColor:            "#4cd137"
    readonly property color activeNav:             "#222f3e"
    property int defaultFontPixelWidth:             5
    property int defaultFontPixelHeight:            UIConstants.fontSize
    property int defaultTextWidth:                  UIConstants.fontSize / 2
    property int defaultTextHeight:                 16
    property int defaultFontPointSize:              8
    property bool demiboldFontFamily:               false

    //------------------ Fonts -----------------------------------
    readonly property string fontComicSans:         "Comic Sans MS"
    readonly property string fontMonospace:         "Monospace"
    readonly property string iWindowsMaximize:      "\uf20d"
    readonly property string iWindowsMinimize:      "\uf2d2"
    readonly property string iPayload:              "\uf03d"
    readonly property string iInternet:             "\uf1eb"
    readonly property string iGPS:                  "\uf276"
    readonly property string iPower:                "\uf244"
    readonly property string inActive:              "#57606f"
    readonly property string iAdd:                  "\uf067"
    readonly property string iOpenFolder:           "\uf07c"
    readonly property string iSaveFolder:           "\uf0c7"
    readonly property string iNewFolder:            "\uf65e"
    readonly property string iDownload:             "\uf019"
    readonly property string iPattern:              "\uf542"
    readonly property string iDeparture:            "\uf5b0"
    readonly property string iLeftOff:              "\uf5af"
    readonly property string iRtl:                  "\uf14d"
    readonly property string iCircle:               "\uf111" //"\uf1ce"
    readonly property string iCircleClockWise:      "\uf2f9"
    readonly property string iCircleCounterClock:   "\uf2ea"
    readonly property string iWaypoint:             "\uf08d"
    readonly property string iAuto:                 "\uf0fb"
    readonly property string iManual:               "\uf44b"
    readonly property string iArmed:                "\uf023"
    readonly property string iDisarmed:             "\uf3c1"
    readonly property string iRunning:              "\uf7a5"
    readonly property string iStop:                 "\uf0da"
    readonly property string iTurnRight:            "\uf064"
    readonly property string iInfo:                 "\uf129"
    readonly property color  cDisableColor:         "#636e72"
    readonly property color  cSelectedColor:        "#b2bec3"
    readonly property color  cMapControlsColor:     "#57606f"
    readonly property string iCompressAlt:          "\uf78c"
    readonly property string iCompress:             "\uf066"
    readonly property string iCenter:               "\uf05b"
    readonly property string iZoomIn:               "\uf00e"
    readonly property string iZoomOut:              "\uf010"
    readonly property string iLockGeo:              "\uf689"
    readonly property string iCar:                  "\uf1b9"
    readonly property string iBolt:                 "\uf0e7"
    readonly property color  btnDefaultColor:       "#40739e"
    readonly property string iFlag:                 "\uf024"
    readonly property string iBatteryThreeQuaters:  "\uf241"
    readonly property string iBatteryHalf:          "\uf242"
    readonly property string iBatteryQuater:        "\uf243"
    readonly property string iBatteryFull:          "\uf240"
    readonly property string iBatteryEmpty:         "\uf244"
    readonly property string iWarning:              "\uf06a"
    readonly property string iDelete:               "\ufe2d"
    readonly property string iDeleteWP:             "\uf057"
    readonly property string iNextWP:               "\uf35a"
    readonly property string iMarker:               "\uf3c5"
    readonly property string iTracker:              "\uf519"
    readonly property string iAvianex:              "\uf374"
    readonly property string iAddWaypoint:          "\uf055"
    readonly property string iPowerOff:             "\uf011"
    readonly property string iClose:                "\uf00d"
    //------- Marker
    readonly property string iMouse:                "\uf00d"
    readonly property string iAddMarker:            "\uf0fe"
    readonly property string iEditMarker:           "\uf044"
    readonly property string iRemoveMarker:         "\uf146"
    readonly property string iClearMarkers:         "\uf05e"
    readonly property string iMarkerFlag:           "\uf024"
    readonly property string iMarkerPlane:          "\uf072"
    readonly property string iMarkerTank:           "\uf7d2"
    readonly property string iMarkerShip:           "\uf21a"
    readonly property string iMarkerTarget:         "\uf140"
    readonly property string iMarkerMilitary:       "\uf0C0"
    //------- UAV and Payload info
    readonly property string iUAVLink:              "\uf7c0"
    readonly property string iGPSLink:              "\uf7bf"
    readonly property string iBattery:              "\uf240"
    readonly property string iMessage:              "\uf071"
    readonly property string iFlightMode:           "\uf072"
    readonly property string iCamLink:              "\uf03d"
    readonly property string iCamMode:              "\uf037"
    readonly property string iSystemConfig:         "\uf085"
    //------- payload button
    readonly property string iSensor:               "\uf042"
    readonly property string iObserve:              "\uf06e"
    readonly property string iPreset:               "\uf0b2"
    readonly property string iSnapshot:             "\uf030"
    readonly property string iMinorInfo:            "\uf141"
    readonly property string iMeasureDistance:      "\uf547"
    readonly property string iGimbalRecord:         "\uf03d"
    readonly property string iGCSRecord:            "\uf03d"
    readonly property string iGimbalStab:           "\uf2b5"
    readonly property string iVisualLock:           "\uf05b"
    readonly property string iFree:                 "\uf067"
    readonly property string iDigitalStab:          "\uf529"
    readonly property string iGCSStab:              "\uf5cb"
    readonly property string iHUD:                  "\uf037"
    readonly property string iInvertPan:            "\uf337"
    readonly property string iInvertTilt:           "\uf338"
    readonly property string iVideoConfig:          "\uf1de"
    readonly property string iAdvanced:             "\uf085"
    readonly property string iDisplay:              "\uf26c"
    readonly property string iBack:                 "\uf3e5"
    readonly property string iEnabled:              "\uf111"
    //------- sharing button
    readonly property string iConnect:              "\uf796"
    readonly property string iVisible:              "\uf06e"
    readonly property string iInvisible:            "\uf070"
    readonly property string iChatIcon:             "\uf27a"
    readonly property string iChatClose:            "\uf00d"
    readonly property string iChatDown:             "\uf078"
    readonly property string iChatUp:               "\uf077"
    readonly property string iChat:                 "\uf086"
    readonly property string iSend:                 "\uf054"
    readonly property string iSendPaperPlane:       "\uf1d8"
    readonly property string iHide:                 "\uf100"
    readonly property string iShare:                "\uf14d"
    readonly property string iMax:                  "\uf2d0"
    readonly property string iClone:                "\uf24d"
    readonly property string iWindowStore:          "\uf2d2"
    readonly property string iDown:                 "\uf0dd"
    readonly property string iUp:                   "\uf151"
    //------- drone  button
    readonly property string iDrone:                "\uf55b"
    readonly property string iPatrolMan:            "\uf007"
    readonly property string iCenterCommander:      "\uf1ad"
    readonly property string iValue:                "\uf129"
    readonly property string iConnection:           "\uf796"
    readonly property string iHealth:               "\uf477"
    readonly property string iVibration:            "\uf478"
    readonly property string iGear:                 "\uf013"
    //------- color
    readonly property color cDroneMajor:            "#1f6cd1"
    readonly property color cDroneOther:            "#5a5d60"
    readonly property color cConnected:             "#4cd137"
    readonly property color cDisConnected:          "#afacac"
    readonly property color cPatrolManNormal:       "#00976e"
    readonly property color cPatrolManNeedHelp:     "#e84118"
    readonly property color cCenterCommander:       "#5d4037"
    //------- payload
    readonly property string iChevronDown:          "\uf078"
    readonly property string iChevronLeft:          "\uf053"
    readonly property string iChevronRight:         "\uf054"
    readonly property string themeNormal:           "normal"
    readonly property string themeGreen:            "green"
    readonly property string themeBlack:            "black"
    readonly property string themeLight:            "light"
    //------- UC
    readonly property string iUserMinus:            "\uf503"
    readonly property string iMinus:                "\uf068"
    readonly property string iAddUser:              "\uf055"
    readonly property string iShareVideo:           "\uf064"
    readonly property string iRemoveUser:           "\uf056"
    function changeTheme(themeType){
        switch(themeType){
        case UIConstants.themeNormal:
            UIConstants.transparentBlue = "#b334495e";
            UIConstants.bgPanelColor = "#283441";
            UIConstants.bgAppColor = "#34495e";
            UIConstants.bgColorOverlay = "#222f3e";
            UIConstants.cfProcessingOverlayBg = "#2d3436";
            UIConstants.sidebarActiveBg = "#283441";
            break;
        case UIConstants.themeGreen:
            UIConstants.transparentBlue = "#b334995e";
            UIConstants.bgPanelColor = "#283441";
            UIConstants.bgAppColor = "#34995e";
            UIConstants.bgColorOverlay = "#50613F";
            UIConstants.cfProcessingOverlayBg = "#40552C";
            UIConstants.sidebarActiveBg = "#34995e";
            break;
        case UIConstants.themeBlack:
            break;
        case UIConstants.themeLight:
            break;
        }
    }

    //nhatdn1 define
    //------- waypoint type
    readonly property int waypointType : 16
    readonly property int loitertimeType : 19
    readonly property int landType : 21
    readonly property int takeoffType : 22
    readonly property int vtoltakeoffType : 84
    readonly property int vtollandType : 85
    readonly property int dojumpType : 177

    //---------z_index
    readonly property int z_targetPolygon: 0
    readonly property int z_mouseArrow: 1
    readonly property int z_tracjactory: 2
    readonly property int z_tracjactoryPlane: 3
    readonly property int z_marker: 4
    readonly property int z_waypoint: 5
    readonly property int z_plane: 5000
    readonly property int z_gotohere: 6
    readonly property int z_ruler: 7

    readonly property int z_map: 0
    readonly property int z_profilePath: 1
    readonly property int z_dialog: 2

    readonly property color planeTrajactoryColor: "#ffee44"




}
