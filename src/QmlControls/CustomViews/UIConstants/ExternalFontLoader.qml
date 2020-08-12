/**
 * ==============================================================================
 * @Project: PGCS
 * @Module: UIConstants
 * @Breif:
 * @Author: Hai Nguyen Hoang
 * @Date: 20/05/2019
 * @Language: QML
 * @License: (c) Viettel Aerospace Institude - Viettel Group
 * ============================================================================
 */

pragma Singleton
import QtQuick 2.0

//------------------------------Font Loader Component ------------------------
QtObject {
    id: fontsManager
    readonly property FontLoader fontAwesomeRegular: FontLoader {
        source: "qrc:/assets/fonts/FontAwesomeRegular-400.otf"
    }
    readonly property FontLoader fontAwesomeSolid: FontLoader {
        source: "qrc:/assets/fonts/FontAwesomeSolid-900.otf"
    }

    readonly property string solidFont: fontsManager.fontAwesomeSolid.name
    readonly property string regularFont: fontsManager.fontAwesomeRegular.name
}

