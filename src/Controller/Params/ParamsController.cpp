#include "ParamsController.h"
#include <QVariant>
#include "../Vehicle/Vehicle.h"
class Vehicle;
ParamsController::ParamsController(Vehicle *vehicle)
{
    _vehicle = vehicle;
    _waitingParamTimeoutTimer.setSingleShot(false);
    _waitingParamTimeoutTimer.setInterval(600);
    connect(&_waitingParamTimeoutTimer, SIGNAL(timeout()), this, SLOT(_waitingParamTimeout()));
    _updateParamTimer.setInterval(1000);
    _updateParamTimer.setSingleShot(false);
//    connect(&_updateParamTimer, SIGNAL(timeout()), this, SLOT(_updateParamTimeout()));
}
ParamsController::~ParamsController(){

}

void ParamsController::refreshAllParameters(uint8_t componentId)
{
    printf("%s\r\n",__func__);
    _parametersReady = false;
    _loadRetry = 0;
    _totalParamCount = 0;
    _paramsReceivedCount = 0;
    _paramCountMap.clear();
    _debugCacheMap.clear();
    _missingParameters = true;
    _startLoadAllParams = true;
    _waitingParamTimeoutTimer.setInterval(600);
    _waitingParamTimeoutTimer.start();
    connect(_vehicle,SIGNAL(mavlinkMessageReceived(mavlink_message_t)),
            this,SLOT(_handleMessageReceived(mavlink_message_t)));
    if(_vehicle->m_firmwarePlugin != nullptr)
        _readParameterRaw(_vehicle->m_firmwarePlugin->rtlAltParamName(),-1);
    mavlink_message_t msg;
    mavlink_msg_param_request_list_pack_chan(_vehicle->m_com->systemId(),
                                             _vehicle->m_com->componentId(),
                                             _vehicle->m_com->mavlinkChannel(),
                                             &msg,
                                             static_cast<uint8_t>(_vehicle->id()),
                                             componentId);
    _vehicle->sendMessageOnLink(_vehicle->m_com, msg);

}
QVariant ParamsController::getParam(QString paramName){
    QVariant result;
    if(_paramMap.keys().contains(paramName)){
        result = QVariant(_paramMap[paramName].param_value);
    }
    return result;
}
void ParamsController::_handleMessageReceived(mavlink_message_t msg)
{
#ifdef DEBUG_FUNC
//    printf("%s MsgID: %d\r\n",__func__,msg.msgid);
#endif
    switch (msg.msgid) {
    case MAVLINK_MSG_ID_PARAM_VALUE:
        _handleParamRequest(msg);
        break;
    case MAVLINK_MSG_ID_PARAM_REQUEST_LIST:
        _handleParamRequestList(msg);
        break;
    case MAVLINK_MSG_ID_PARAM_REQUEST_READ:
        _handleParamRequestRead(msg);
        break;
    default:
        break;
    }
}
void ParamsController::_waitingParamTimeout(void){
    printf("%s Loading %d/%d params\r\n",__func__,_paramsReceivedCount,_totalParamCount);
    if(_lastParamsReceivedCount < _paramsReceivedCount)
    {
        _lastParamsReceivedCount = _paramsReceivedCount;
    }
    else
    {
        if(_totalParamCount > 0){
            if( _paramsReceivedCount == _totalParamCount){
                _totalLoopParamFail = 0;
                // stop waiting params
                printf("Full param received\r\n");
                _missingParameters = false;
                Q_EMIT missingParametersChanged(_missingParameters);
                _setLoadProgress(1);
                _waitingParamTimeoutTimer.stop();
            }else{
                _totalLoopParamFail++;
                _initialRequestMissingParams();
            }
        }else{
            if(_loadRetry < _maxRetry){
                _loadRetry++;
                if(_startLoadAllParams){
                    mavlink_message_t msg;
                    mavlink_msg_param_request_list_pack_chan(_vehicle->m_com->systemId(),
                                                             _vehicle->m_com->componentId(),
                                                             _vehicle->m_com->mavlinkChannel(),
                                                             &msg,
                                                             static_cast<uint8_t>(_vehicle->id()),
                                                             _vehicle->m_com->componentId());
                    _vehicle->sendMessageOnLink(_vehicle->m_com, msg);
                }
            }else{
                _missingParameters = true;
                Q_EMIT missingParametersChanged(_missingParameters);
                _waitingParamTimeoutTimer.stop();
            }
        }
    }
}
void ParamsController::_updateParamTimeout(void){
    Q_EMIT _vehicle->paramsModelChanged();
}
void ParamsController::_initialRequestMissingParams(void){
    printf("%s\r\n",__func__);
    QList<int> missParams;
    for(int componentId=0; componentId< _totalParamCount; componentId++){
        // check params loaded in every single component
        if(!_debugCacheMap.keys().contains(componentId)){
            missParams.push_back(componentId);
        }
    }
    printf("Missing %d/%d = %f params\r\n",missParams.size(),_totalParamCount,
           static_cast<float>(missParams.size())/static_cast<float>(_totalParamCount));
    if(static_cast<float>(missParams.size()) < static_cast<float>(_totalParamCount) * 0.25f){
        //_waitingParamTimeoutTimer.setInterval(600);
        printf("Reload single params\r\n");
        for(int i=0; i< missParams.size(); i++){
            _readParameterRaw("",missParams[i]);
            if(i==10) break;
        }
        sleep(1);
    }else{

        if(_totalLoopParamFail>=16)//~10s
        {
            _totalLoopParamFail = 0;
            printf("Download missing params\r\n");
            _missingParameters = true;
            Q_EMIT missingParametersChanged(_missingParameters);

            printf("Reload all params\r\n");
            mavlink_message_t msg;
            mavlink_msg_param_request_list_pack_chan(_vehicle->m_com->systemId(),
                                                     _vehicle->m_com->componentId(),
                                                     _vehicle->m_com->mavlinkChannel(),
                                                     &msg,
                                                     static_cast<uint8_t>(_vehicle->id()),
                                                     _vehicle->m_com->componentId());
            _vehicle->sendMessageOnLink(_vehicle->m_com, msg);
        }
    }
}
void ParamsController::_readParameterRaw(const QString& paramName, int paramIndex)
{
#ifdef DEBUG
    printf("%s [%d]\r\n",__func__,paramIndex);
#endif
    mavlink_message_t msg;

    char fixedParamName[MAVLINK_MSG_PARAM_REQUEST_READ_FIELD_PARAM_ID_LEN];

    strncpy(fixedParamName, paramName.toStdString().c_str(), sizeof(fixedParamName));
    mavlink_msg_param_request_read_pack_chan(_vehicle->m_com->systemId(),
                                             _vehicle->m_com->componentId(),
                                             _vehicle->m_com->mavlinkChannel(),
                                             &msg,                           // Pack into this mavlink_message_t
                                             _vehicle->id(),                 // Target system id
                                             _vehicle->m_com->componentId(),                    // Target component id
                                             fixedParamName,                 // Named parameter being requested
                                             paramIndex);                    // Parameter index being requested, -1 for named
    _vehicle->sendMessageOnLink(_vehicle->m_com, msg);
}

void ParamsController::_writeParameterRaw(const QString& paramName, const QVariant& value)
{
    mavlink_param_set_t     p;
    mavlink_param_union_t   union_value;

    memset(&p, 0, sizeof(p));
    bool foundParam = false;

    for(int i : _debugCacheMap.keys()){
        if(_debugCacheMap[i].first  == paramName){
            foundParam = true;
            p.param_type = _debugCacheMap[i].second.param_type;
            break;
        }
    }

    printf("foundParam = %s [%d] [%f] value=%f\r\n",
           foundParam?"true":"false",p.param_type,
           static_cast<double>(p.param_value),
           static_cast<double>(value.toFloat()));
    if(!foundParam) return;

    p.param_value = value.toString().replace(",",".").toFloat();

    p.target_system = static_cast<uint8_t>(_vehicle->id());
    p.target_component = static_cast<uint8_t>(_vehicle->communication()->componentId());

    strncpy(p.param_id, paramName.toStdString().c_str(), sizeof(p.param_id));
    printf("p.param_id=%s p.param_value = %f\r\n",
           p.param_id,
           static_cast<double>(p.param_value));
    mavlink_message_t msg;
    mavlink_msg_param_set_encode_chan(_vehicle->communication()->systemId(),
                                      _vehicle->communication()->componentId(),
                                      _vehicle->communication()->mavlinkChannel(),
                                      &msg,
                                      &p);
    _vehicle->sendMessageOnLink(_vehicle->communication(), msg);
}
void ParamsController::_handleParamRequest(mavlink_message_t msg)
{
    mavlink_param_value_t rawValue;
    mavlink_msg_param_value_decode(&msg, &rawValue);
//    printf("Param[%s] received index=%d\r\n",rawValue.param_id,rawValue.param_index);
    if(_totalParamCount == 0) {
        _totalParamCount = rawValue.param_count;
        _paramCountMap[msg.compid] = _totalParamCount;
    }else{

    }
    if(!_parametersReady && rawValue.param_index != 65535){

//        printf("%s Param[%s] received\r\n",__func__,rawValue.param_id);
        if(!_debugCacheMap.keys().contains(rawValue.param_index)){
            QString paramID = _convertParamID(rawValue.param_id);
            if(_vehicle->m_firmwarePlugin != nullptr &&
                    paramID == _vehicle->m_firmwarePlugin->rtlAltParamName()){
//                printf("Received param[%s] = %d (%f)\r\n",
//                       paramID.toStdString().c_str(),
//                       _convertParamValue(rawValue).toInt(),
//                       static_cast<double>(rawValue.param_value));
               if(_vehicle != nullptr){
                   QGeoCoordinate homePosition = _vehicle->homePosition();
                   homePosition.setAltitude(static_cast<double>(rawValue.param_value)/100);
                    _vehicle->setHomePosition(homePosition);
                }
            }else if(_vehicle->m_firmwarePlugin != nullptr &&
                     paramID == _vehicle->m_firmwarePlugin->airSpeedParamName()){
                _vehicle->setParamAirSpeed(static_cast<float>(rawValue.param_value)/100);
            }else if(_vehicle->m_firmwarePlugin != nullptr &&
                     paramID == _vehicle->m_firmwarePlugin->loiterRadiusParamName()){
                _vehicle->setParamLoiterRadius(static_cast<float>(rawValue.param_value));
            }
            ParamTypeVal param = QPair<QString,mavlink_param_value_t>(paramID,rawValue);
            _debugCacheMap[rawValue.param_index] = param;
            _paramMap[paramID] = rawValue;
            Q_EMIT paramChanged(paramID);
            _paramsReceivedCount++;
            _setLoadProgress(static_cast<float>(_paramsReceivedCount)/
                             static_cast<float>(_totalParamCount));
#ifdef DEBUG_FUNC
            printf("%s [_paramsReceivedCount=%d] param_index:[%04d] param_id:%s param_count:%d\r\n",
                   __func__,_paramsReceivedCount,rawValue.param_index,
                   rawValue.param_id,rawValue.param_count);
#endif
            _vehicle->_setParamValue(paramID,
                                             QString::fromStdString(
                                                 std::to_string(static_cast<double>(rawValue.param_value))),
                                             "");

        }else{

        }
        if(_paramsReceivedCount == _totalParamCount){
            printf("Full param received\r\n");
            _missingParameters = false;
            Q_EMIT missingParametersChanged(_missingParameters);
            _setLoadProgress(1);
            _waitingParamTimeoutTimer.stop();
            _parametersReady = true;
//            _updateParamTimer.start();
            Q_EMIT _vehicle->paramsModelChanged();
        }
    }else{
//        printf("Current param [%s] update\r\n",rawValue.param_id);
        QString paramID = _convertParamID(rawValue.param_id);
        _paramMap[paramID] = rawValue;
        Q_EMIT paramChanged(paramID);
        printf("Received param update [%s] = %d (%f)\r\n",
               paramID.toStdString().c_str(),
               _convertParamValue(rawValue).toInt(),
               static_cast<double>(rawValue.param_value));
        if(paramID == _vehicle->m_firmwarePlugin->rtlAltParamName()){
            if(_vehicle != nullptr){
               QGeoCoordinate homePosition = _vehicle->homePosition();
               homePosition.setAltitude(static_cast<double>(rawValue.param_value)/100);
                _vehicle->setHomePosition(homePosition);
            }
        }else if(_vehicle->m_firmwarePlugin != nullptr &&
                 paramID == _vehicle->m_firmwarePlugin->airSpeedParamName()){
            _vehicle->setParamAirSpeed(static_cast<float>(rawValue.param_value)/100);
        }else if(_vehicle->m_firmwarePlugin != nullptr &&
                 paramID == _vehicle->m_firmwarePlugin->loiterRadiusParamName()){
            _vehicle->setParamLoiterRadius(static_cast<float>(rawValue.param_value));
        }
        _vehicle->_setParamValue(paramID,
                                         QString::fromStdString(
                                             std::to_string(static_cast<double>(rawValue.param_value))),
                                         "",true);
    }
}
void ParamsController::_handleParamRequestList(mavlink_message_t msg)
{
    mavlink_param_request_list_t request;
    mavlink_msg_param_request_list_decode(&msg, &request);
#ifdef DEBUG_FUNC
    printf("%s target_system:%d target_component:%d\r\n",__func__,request.target_system,request.target_component);
#endif
}
void ParamsController::_handleParamRequestRead(mavlink_message_t msg)
{
    mavlink_param_request_read_t request;
    mavlink_msg_param_request_read_decode(&msg, &request);
    const QString paramName(QString::fromLocal8Bit(request.param_id, strnlen(request.param_id, MAVLINK_MSG_PARAM_REQUEST_READ_FIELD_PARAM_ID_LEN)));
//#ifdef DEBUG_FUNC
    printf("%s target_system:%d target_component:%d param_id:%s param_index:%d \r\n",
           __func__,request.target_system,request.target_component,
           request.param_id,request.param_index);
//#endif
}
QVariant ParamsController::_convertParamValue(mavlink_param_value_t rawValue){
    QVariant paramValue;
    mavlink_param_union_t paramUnion;
    paramUnion.param_float = rawValue.param_value;
    paramUnion.type = rawValue.param_type;
    // Insert with correct type

    switch (rawValue.param_type) {
        case MAV_PARAM_TYPE_REAL32:
            paramValue = QVariant(paramUnion.param_float);
            break;

        case MAV_PARAM_TYPE_UINT8:
            paramValue = QVariant(paramUnion.param_uint8);
            break;

        case MAV_PARAM_TYPE_INT8:
            paramValue = QVariant(paramUnion.param_int8);
            break;

        case MAV_PARAM_TYPE_UINT16:
            paramValue = QVariant(paramUnion.param_uint16);
            break;

        case MAV_PARAM_TYPE_INT16:
            paramValue = QVariant(paramUnion.param_int16);
            break;

        case MAV_PARAM_TYPE_UINT32:
            paramValue = QVariant(paramUnion.param_uint32);
            break;

        case MAV_PARAM_TYPE_INT32:
            paramValue = QVariant(paramUnion.param_int32);
            break;

        //-- Note: These are not handled above:
        //
        //   MAV_PARAM_TYPE_UINT64
        //   MAV_PARAM_TYPE_INT64
        //   MAV_PARAM_TYPE_REAL64
        //
        //   No space in message (the only storage allocation is a "float") and not present in mavlink_param_union_t

        default:
            qCritical() << "INVALID DATA TYPE USED AS PARAMETER VALUE: " << rawValue.param_type;
    }
    return paramValue;
}
QString ParamsController::_convertParamID(char* param_id){
    QByteArray bytes(param_id, MAVLINK_MSG_PARAM_VALUE_FIELD_PARAM_ID_LEN);
    // Construct a string stopping at the first NUL (0) character, else copy the whole
    // byte array (max MAVLINK_MSG_PARAM_VALUE_FIELD_PARAM_ID_LEN, so safe)
    QString parameterName(bytes);
    return parameterName;
}
void ParamsController::_setLoadProgress(float loadProgress)
{
    _loadProgress = loadProgress;
    Q_EMIT loadProgressChanged(loadProgress);
}

QList<int> ParamsController::componentIds(void)
{
    return _paramCountMap.keys();
}
