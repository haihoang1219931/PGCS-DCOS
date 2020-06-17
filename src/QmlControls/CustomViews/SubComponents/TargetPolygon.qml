import QtQuick 2.0
import QtLocation 5.9
import QtPositioning 5.5
import QtQuick.Window 2.0
import io.qdt.dev 1.0
import CustomViews.UIConstants 1.0


MapPolygon{
    id:gcs_target_polygon
    color: "green"
    opacity: 0.5
    border.color: "gray"
    border.width: 2
    antialiasing: true
    smooth: true
    path:[]

    function changeCoordinate(coord1,coord2,coord3,coord4)
    {
        //if(gcs_target_polygon.path.length >= 4)
        {
            path[0] = coord1;
            path[1] = coord2;
            path[2] = coord3;
            path[3] = coord4;
        }
    }
}

//MapQuickItem {
//    id: gcs_target_polygon

//}

