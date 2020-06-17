function sleep(milliseconds) {
  var start = new Date().getTime();
  for (var i = 0; i < 1e7; i++) {
    if ((new Date().getTime() - start) > milliseconds){
      break;
    }
  }
}


function calculateScale(_map,_scaleLine,_scaleImage,_scaleImageLeft,_scaleLengths,_scaleText)
{
    var coord1, coord2, dist, text, f
    f = 0
    coord1 = _map.toCoordinate(Qt.point(0,_scaleLine.y))
    coord2 = _map.toCoordinate(Qt.point(0+_scaleImage.sourceSize.width,_scaleLine.y))
    dist = Math.round(coord1.distanceTo(coord2))

    if (dist === 0) {
        // not visible
    } else {
        for (var i = 0; i < _scaleLengths.length-1; i++) {
            if (dist < (_scaleLengths[i] + _scaleLengths[i+1]) / 2 ) {
                f = _scaleLengths[i] / dist
                dist = _scaleLengths[i]
                break;
            }
        }
        if (f === 0) {
            f = dist / _scaleLengths[i]
            dist = _scaleLengths[i]
        }
    }

    text = formatDistance(dist)
    _scaleImage.width = (_scaleImage.sourceSize.width * f) - 2 * _scaleImageLeft.sourceSize.width
    _scaleText.text = text
}

function formatDistance(meters)
{
    var dist = Math.round(meters)
    if (dist > 1000 ){
        if (dist > 100000){
            dist = Math.round(dist / 1000)
        }
        else{
            dist = Math.round(dist / 100)
            dist = dist / 10
        }
        dist = dist + " km"
    }
    else{
        dist = dist + " m"

    }
    return dist
}

function getScale(map)//pixel per m
{
    var coord1 = map.toCoordinate(Qt.point(0,0))
    var coord2 = map.toCoordinate(Qt.point(100,0))
    var dist = Math.round(coord1.distanceTo(coord2))
    return 1000/dist
}

function unselect_all(map,list_symbol)
{
    for(var i=0;i<list_symbol.length;i++)
    {
         list_symbol[i].isSelected=false
    }
    map.gesture.enabled = true
}

function convert_coordinator2screen(_coord,_map)
{
    var w = _map.width
    var h = _map.height
    var coord1 = _map.toCoordinate(Qt.point(0,0))
    var coord2 = _map.toCoordinate(Qt.point(w,h))

    var p = Qt.point(0,0)

    p.x = w*(_coord.longitude - coord1.longitude)/(coord2.longitude - coord1.longitude)
    p.y = h*(_coord.latitude - coord1.latitude)/(coord2.latitude - coord1.latitude)

    return p
}


function convertUrltoPath(url)
{
    return url.toString().replace(/^(file:\/{3})|(qrc:\/{2})|(http:\/{2})/,"/")
}
