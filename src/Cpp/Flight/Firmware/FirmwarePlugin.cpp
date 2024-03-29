#include "FirmwarePlugin.h"
#include "../Params/Fact.h"
#include "../Vehicle/Vehicle.h"
FirmwarePlugin::FirmwarePlugin(Vehicle* vehicle)
{
    setVehicle(vehicle);
    loadFromFile("conf/Properties.conf");
}
void FirmwarePlugin::setVehicle(Vehicle *vehicle){
    m_vehicle = vehicle;
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

        //pw
        _listParamShow.append(new Fact(true,"PW_VBattA","0","V"));
        _listParamShow.append(new Fact(true,"PW_IBattA","0","A"));
        _listParamShow.append(new Fact(true,"PW_EBattA","0","mAh"));
        _listParamShow.append(new Fact(true,"PW_VBattB","0","V"));
        _listParamShow.append(new Fact(true,"PW_IBattB","0","A"));
        _listParamShow.append(new Fact(true,"PW_EBattB","0","mAh"));
        _listParamShow.append(new Fact(true,"PW_VBat12S","0","V"));
        _listParamShow.append(new Fact(true,"PW_Temp","0","°C"));

        //ecu
        _listParamShow.append(new Fact(true,"ECU_Throttle","0","%"));
        _listParamShow.append(new Fact(true,"ECU_FuelUsed","0","l"));
        _listParamShow.append(new Fact(true,"ECU_CHT","0","°C"));
        _listParamShow.append(new Fact(true,"ECU_FuelPressure","0","Bar"));
        _listParamShow.append(new Fact(true,"ECU_Hobbs","0","s"));
        _listParamShow.append(new Fact(true,"ECU_CPULoad","0","%"));
        _listParamShow.append(new Fact(true,"ECU_ChargeTemp","0","°C"));
        _listParamShow.append(new Fact(true,"ECU_FlowRate","0"," "));
        _listParamShow.append(new Fact(true,"ECU_Rpm","0","RPM"));
        _listParamShow.append(new Fact(true,"ECU_ThrottlePulse","0"," "));

        //aux_adc
        _listParamShow.append(new Fact(true,"ADC_FuelLevel","0"," "));
        _listParamShow.append(new Fact(true,"ADC_RawFuelLevel","0"," "));
        _listParamShow.append(new Fact(true,"ADC_EnvTemp","0"," "));
        _listParamShow.append(new Fact(true,"ADC_EnvRH","0"," "));


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
bool FirmwarePlugin::pic(){
    return false;
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
void FirmwarePlugin::initializeVehicle()
{
}
QStringList FirmwarePlugin::flightModes(){
    return m_mapFlightMode.values();
}
QStringList FirmwarePlugin::flightModesOnGround(){
    return m_mapFlightModeOnGround.values();
}
QStringList FirmwarePlugin::flightModesOnAir(){
    return m_mapFlightModeOnAir.values();
}
QString FirmwarePlugin::gotoFlightMode() const
{
    return QStringLiteral("Guided");
}
bool FirmwarePlugin::hasGimbal( bool& rollSupported, bool& pitchSupported, bool& yawSupported)
{

    rollSupported = false;
    pitchSupported = false;
    yawSupported = false;
    return false;
}
bool FirmwarePlugin::isVtol() const
{
    if(m_vehicle!= nullptr){
        switch (m_vehicle->vehicleType()) {
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
    else
        return false;
}
void FirmwarePlugin::sendHomePosition(QGeoCoordinate location){

}
bool FirmwarePlugin::setFlightMode(const QString& flightMode, uint8_t* base_mode, uint32_t* custom_mode)
{
    Q_UNUSED(flightMode);
    Q_UNUSED(base_mode);
    Q_UNUSED(custom_mode);

    qWarning() << "FirmwarePlugin::setFlightMode called on base class, not supported";

    return false;
}

bool FirmwarePlugin::iscommand() const
{
    // Not supported by generic vehicle

    return false;
}

void FirmwarePlugin::setcommand( bool command)
{

    Q_UNUSED(command);
}

void FirmwarePlugin::pauseVehicle()
{
    // Not supported by generic vehicle

}
void FirmwarePlugin::commandRTL()
{

}
void FirmwarePlugin::commandLand(){

}

void FirmwarePlugin::commandTakeoff(double altitudeRelative){

    Q_UNUSED(altitudeRelative);
}

double FirmwarePlugin::minimumTakeoffAltitude(){
    return 0;
}

void FirmwarePlugin::commandGotoLocation(const QGeoCoordinate& gotoCoord){
    Q_UNUSED(gotoCoord);
}

void FirmwarePlugin::commandChangeAltitude(double altitudeChange){
    Q_UNUSED(altitudeChange);
}

void FirmwarePlugin::commandSetAltitude(double newAltitude){

    Q_UNUSED(newAltitude);
}

void FirmwarePlugin::commandChangeSpeed(double speedChange){

    Q_UNUSED(speedChange);
}

void FirmwarePlugin::commandOrbit(const QGeoCoordinate& centerCoord,
                                  double radius, double amslAltitude){
    Q_UNUSED(centerCoord);
    Q_UNUSED(radius);
    Q_UNUSED(amslAltitude);
}

void FirmwarePlugin::emergencyStop(){

}

void FirmwarePlugin::abortLanding(double climbOutAltitude){
    Q_UNUSED(climbOutAltitude);
}

void FirmwarePlugin::startMission(){

}

void FirmwarePlugin::startEngine()
{
}

void FirmwarePlugin::setCurrentMissionSequence(int seq){
    Q_UNUSED(seq);
}

void FirmwarePlugin::rebootVehicle(){

}

void FirmwarePlugin::clearMessages(){

}

void FirmwarePlugin::triggerCamera(){

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

void FirmwarePlugin::motorTest(int motor, int percent){

    Q_UNUSED(motor);
    Q_UNUSED(percent);
}
void FirmwarePlugin::handleJSButton(int id, bool clicked){
    Q_UNUSED(id);
    Q_UNUSED(clicked);
}
void FirmwarePlugin::handleUseJoystick(bool enable){
    Q_UNUSED(enable);
}
void FirmwarePlugin::setHomeHere(float lat, float lon, float alt){

    Q_UNUSED(lat);
    Q_UNUSED(lon);
    Q_UNUSED(alt);
}

void FirmwarePlugin::setGimbalRate(float pan, float tilt)
{
    Q_UNUSED(pan);
    Q_UNUSED(tilt);
}

void FirmwarePlugin::setGimbalMode(QString mode)
{
    Q_UNUSED(mode);
}

void FirmwarePlugin::changeGimbalCurrentMode()
{

}

QString FirmwarePlugin::getGimbalCurrentMode()
{

}

void FirmwarePlugin::setGimbalAngle(float pan, float tilt)
{
    Q_UNUSED(pan);
    Q_UNUSED(tilt);
}
