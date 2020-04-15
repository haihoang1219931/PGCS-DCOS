#include "FirmwarePlugin.h"
#include "../Params/Fact.h"
#include "../Vehicle/Vehicle.h"
FirmwarePlugin::FirmwarePlugin(QObject *parent) : QObject(parent)
{
    loadFromFile("conf/Properties.conf");
}
void FirmwarePlugin::loadFromFile(QString fileName){
    if(_listParamShow.size() != 0){
        _listParamShow.clear();
    }
    XMLDocument m_doc;
    XMLError res = m_doc.LoadFile(fileName.toStdString().c_str());
    if(res == XML_SUCCESS){
        XMLElement * pElement = m_doc.FirstChildElement("ArrayOfProperties");
        XMLElement * pListElement = pElement->FirstChildElement("Property");
        //int i = 0;
        while(pListElement!= nullptr){
            //printf("item %d\r\n",i);

            XMLElement * pSelected = pListElement->FirstChildElement("Selected");
            XMLElement * pName = pListElement->FirstChildElement("Name");
            XMLElement * pUnit = pListElement->FirstChildElement("Unit");
            Fact *tmp = new Fact(QString::fromLocal8Bit(pSelected->GetText()).contains("true"),
                                 QString::fromLocal8Bit(pName->GetText()),
                                 "",
                                 QString::fromLocal8Bit(pUnit->GetText()));
            pListElement = pListElement->NextSiblingElement("Property");
            //i++;
            _listParamShow.append(tmp);
        }
    }
    if(_listParamShow.size() <  37){
        _listParamShow.clear();
        _listParamShow.append(new Fact(false,"AirSpeed","0","m"));
        _listParamShow.append(new Fact(false,"AltHome","0","m"));
        _listParamShow.append(new Fact(false,"AltitudeAGL","0","m"));
        _listParamShow.append(new Fact(false,"AltitudeAMSL","0","m"));
        _listParamShow.append(new Fact(false,"ClimbSpeed","0","m"));
        _listParamShow.append(new Fact(false,"Current","0","m"));
        _listParamShow.append(new Fact(false,"Current2","0","m"));
        _listParamShow.append(new Fact(false,"DisttoWP","0","m"));
        _listParamShow.append(new Fact(false,"EkfCompass","0",""));
        _listParamShow.append(new Fact(false,"EkfPosH","0",""));
        _listParamShow.append(new Fact(false,"EkfPosV","0",""));
        _listParamShow.append(new Fact(false,"EkfVel","0",""));
        _listParamShow.append(new Fact(false,"GPSHdop","0","m"));
        _listParamShow.append(new Fact(false,"GPSSatCount","0","m"));
        _listParamShow.append(new Fact(false,"GroundLevel","0","m"));
        _listParamShow.append(new Fact(false,"GroundSpeed","0","m"));
        _listParamShow.append(new Fact(false,"Landed","False",""));
        _listParamShow.append(new Fact(false,"LatHome","0","deg"));
        _listParamShow.append(new Fact(false,"LongHome","0","deg"));
        _listParamShow.append(new Fact(false,"Latitude","0","deg"));
        _listParamShow.append(new Fact(false,"Longitude","0","deg"));
        _listParamShow.append(new Fact(true,"PitchDeg","0","m"));
        _listParamShow.append(new Fact(true,"PMU_IBatA","0","A"));
        _listParamShow.append(new Fact(true,"PMU_IBatB","0","A"));
        _listParamShow.append(new Fact(true,"PMU_Rpm","0","rpm"));
        _listParamShow.append(new Fact(true,"PMU_Temp","0","deg"));
        _listParamShow.append(new Fact(true,"PMU_vBatt12S","0","V"));
        _listParamShow.append(new Fact(true,"PMU_vBatA","0","V"));
        _listParamShow.append(new Fact(true,"PMU_vBatB","0","V"));
        _listParamShow.append(new Fact(false,"PTU_Alt","0","m"));
        _listParamShow.append(new Fact(false,"PTU_Heading","0","m"));
        _listParamShow.append(new Fact(false,"PTU_Press","0","m"));
        _listParamShow.append(new Fact(false,"PTU_RSSI","0","m"));
        _listParamShow.append(new Fact(false,"PTU_Temperature","0","m"));
        _listParamShow.append(new Fact(false,"RollDeg","0","m"));
        _listParamShow.append(new Fact(false,"RPM1","0","m"));
        _listParamShow.append(new Fact(false,"RPM2","0","m"));
        _listParamShow.append(new Fact(false,"Sonarrange","0","m"));
        _listParamShow.append(new Fact(false,"TargetAlt","0","m"));
        _listParamShow.append(new Fact(false,"VibeX","0",""));
        _listParamShow.append(new Fact(false,"VibeY","0",""));
        _listParamShow.append(new Fact(false,"VibeZ","0",""));
        _listParamShow.append(new Fact(false,"Voltage","0","V"));
        _listParamShow.append(new Fact(false,"Voltage2","0","V"));
        _listParamShow.append(new Fact(false,"WindHeading","0","Deg"));
        _listParamShow.append(new Fact(false,"WindSpeed","0","km/h"));
        _listParamShow.append(new Fact(false,"YawDeg","0","Deg"));
        saveToFile("conf/Properties.conf",_listParamShow);
    }
}
void FirmwarePlugin::saveToFile(QString fileName,QList<Fact*> _listParamShow){
    if(fileName.contains(".conf")){
        XMLDocument xmlDoc;
        XMLNode * pRoot = xmlDoc.NewElement("ArrayOfProperties");
        xmlDoc.InsertFirstChild(pRoot);
        for(int i=0; i< _listParamShow.size(); i++){
            Fact *tmp = _listParamShow[i];
            XMLElement * pElement = xmlDoc.NewElement("Property");
            XMLElement * pSelected = xmlDoc.NewElement("Selected");
            pSelected->SetText(tmp->selected()?"true":"false");
            pElement->InsertEndChild(pSelected);
            XMLElement * pName = xmlDoc.NewElement("Name");
            pName->SetText(tmp->name().toStdString().c_str());
            pElement->InsertEndChild(pName);
            XMLElement * pUnit = xmlDoc.NewElement("Unit");
            pUnit->SetText(tmp->unit().toStdString().c_str());
            pElement->InsertEndChild(pUnit);
            pRoot->InsertEndChild(pElement);
        }
        XMLError eResult = xmlDoc.SaveFile(fileName.toStdString().c_str());
    }
}
QList<Fact*> FirmwarePlugin::listParamsShow(){
    return _listParamShow;
}
QString FirmwarePlugin::rtlAltParamName(){
    return m_rtlAltParamName;
}
QString FirmwarePlugin::airSpeedParamName(){
    return m_airSpeedParamName;
}
QString FirmwarePlugin::loiterRadiusParamName(){
    return m_loiterRadiusParamName;
}
QString FirmwarePlugin::flightMode(int flightModeId){
    Q_UNUSED(flightModeId);
    return "UNDEFINED";
}
bool FirmwarePlugin::flightModeID(QString flightMode,int* base_mode,int* custom_mode){
    Q_UNUSED(flightMode);
    Q_UNUSED(base_mode);
    Q_UNUSED(custom_mode);
    return false;
}
void FirmwarePlugin::initializeVehicle(Vehicle* vehicle)
{
    Q_UNUSED(vehicle);
}
QStringList FirmwarePlugin::flightModes(){
    return m_mapFlightModeOnGround.values();
}
QStringList FirmwarePlugin::flightModesOnAir(){
    return m_mapFlightModeOnAir.values();
}
QString FirmwarePlugin::gotoFlightMode(void) const
{
    return QStringLiteral("Guided");
}
bool FirmwarePlugin::hasGimbal(Vehicle* vehicle, bool& rollSupported, bool& pitchSupported, bool& yawSupported)
{
    Q_UNUSED(vehicle);
    rollSupported = false;
    pitchSupported = false;
    yawSupported = false;
    return false;
}
bool FirmwarePlugin::isVtol(const Vehicle* vehicle) const
{
    switch (vehicle->vehicleType()) {
    case MAV_TYPE_VTOL_DUOROTOR:
    case MAV_TYPE_VTOL_QUADROTOR:
    case MAV_TYPE_VTOL_TILTROTOR:
    case MAV_TYPE_VTOL_RESERVED2:
    case MAV_TYPE_VTOL_RESERVED3:
    case MAV_TYPE_VTOL_RESERVED4:
    case MAV_TYPE_VTOL_RESERVED5:
        return true;
    default:
        return false;
    }
}
void FirmwarePlugin::sendHomePosition(Vehicle* vehicle,QGeoCoordinate location){

}
bool FirmwarePlugin::setFlightMode(const QString& flightMode, uint8_t* base_mode, uint32_t* custom_mode)
{
    Q_UNUSED(flightMode);
    Q_UNUSED(base_mode);
    Q_UNUSED(custom_mode);

    qWarning() << "FirmwarePlugin::setFlightMode called on base class, not supported";

    return false;
}

bool FirmwarePlugin::iscommand(const Vehicle* vehicle) const
{
    // Not supported by generic vehicle
    Q_UNUSED(vehicle);
    return false;
}

void FirmwarePlugin::setcommand(Vehicle* vehicle, bool command)
{
    Q_UNUSED(vehicle);
    Q_UNUSED(command);
}

void FirmwarePlugin::pauseVehicle(Vehicle* vehicle)
{
    // Not supported by generic vehicle
    Q_UNUSED(vehicle);
}
void FirmwarePlugin::commandRTL(void)
{

}
void FirmwarePlugin::commandLand(void){

}

void FirmwarePlugin::commandTakeoff(Vehicle* vehicle,double altitudeRelative){
    Q_UNUSED(vehicle);
    Q_UNUSED(altitudeRelative);
}

double FirmwarePlugin::minimumTakeoffAltitude(void){
    return 0;
}

void FirmwarePlugin::commandGotoLocation(Vehicle *vehicle,const QGeoCoordinate& gotoCoord){
    Q_UNUSED(gotoCoord);
}

void FirmwarePlugin::commandChangeAltitude(double altitudeChange){
    Q_UNUSED(altitudeChange);
}

void FirmwarePlugin::commandSetAltitude(Vehicle *vehicle,double newAltitude){
    Q_UNUSED(vehicle);
    Q_UNUSED(newAltitude);
}

void FirmwarePlugin::commandChangeSpeed(Vehicle* vehicle,double speedChange){
    Q_UNUSED(vehicle);
    Q_UNUSED(speedChange);
}

void FirmwarePlugin::commandOrbit(const QGeoCoordinate& centerCoord,
                                     double radius, double amslAltitude){
    Q_UNUSED(centerCoord);
    Q_UNUSED(radius);
    Q_UNUSED(amslAltitude);
}

void FirmwarePlugin::pauseVehicle(void){

}

void FirmwarePlugin::emergencyStop(void){

}

void FirmwarePlugin::abortLanding(double climbOutAltitude){
    Q_UNUSED(climbOutAltitude);
}

void FirmwarePlugin::startMission(Vehicle* vehicle){
    Q_UNUSED(vehicle);
}

void FirmwarePlugin::setCurrentMissionSequence(Vehicle* vehicle, int seq){
    Q_UNUSED(vehicle);
    Q_UNUSED(seq);
}

void FirmwarePlugin::rebootVehicle(){

}

void FirmwarePlugin::clearMessages(){

}

void FirmwarePlugin::triggerCamera(void){

}
void FirmwarePlugin::sendPlan(QString planFile){
    Q_UNUSED(planFile);
}

/// Used to check if running current version is equal or higher than the one being compared.
//  returns 1 if current > compare, 0 if current == compare, -1 if current < compare

int FirmwarePlugin::versionCompare(QString& compare){
    Q_UNUSED(compare);
    return 0;
}
int FirmwarePlugin::versionCompare(int major, int minor, int patch){
    Q_UNUSED(major);
    Q_UNUSED(minor);
    Q_UNUSED(patch);
    return 0;
}

void FirmwarePlugin::motorTest(Vehicle* vehicle,int motor, int percent){
    Q_UNUSED(vehicle);
    Q_UNUSED(motor);
    Q_UNUSED(percent);
}
void FirmwarePlugin::setHomeHere(Vehicle* vehicle,float lat, float lon, float alt){
    Q_UNUSED(vehicle);
    Q_UNUSED(lat);
    Q_UNUSED(lon);
    Q_UNUSED(alt);
}
