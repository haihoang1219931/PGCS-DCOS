var rMajor = 6378137.0; //Equatorial Radius, WGS84
var shift  = Math.PI * rMajor;
function mercatorToLatLon(mercX, mercY) {
    var lon    = mercX / shift * 180.0;
    var lat    = mercY / shift * 180.0;
    lat = 180.0 / Math.PI * (2 * Math.atan(Math.exp(lat * Math.PI / 180.0)) - Math.PI / 2.0);
    while(lon > 180)
        lon-=360;
    while(lon < -180)
        lon+=360;
    while(lat > 90)
        lat-=180;
    while(lat < -90)
        lat+=180;
    return { 'lon': lon, 'lat': lat };
}
function latlongToMercator(lat,lon){
    var x = lon * shift / 180.0;
    var y = shift / Math.PI * Math.log(Math.tan(Math.PI * lat / 360.0+ Math.PI / 4.0));
    return { 'x': x, 'y': y };
}
function pad(num, size) {
    var s = "00000000000000" + num;
    return s.substr(s.length-size);
}
