#include "PlanController.h"
Q_LOGGING_CATEGORY(PlanManagerLog, "PlanManagerLog");
PlanController::PlanController(QObject *parent) : QObject(parent)
{
    m_planType = MAV_MISSION_TYPE_MISSION;
    m_ackTimeoutTimerDownload = new QTimer(this);
    m_ackTimeoutTimerDownload->setInterval(_ackTimeoutMilliseconds);
    m_ackTimeoutTimerDownload->setSingleShot(true);

    m_ackTimeoutTimerUpload = new QTimer(this);
    m_ackTimeoutTimerUpload->setInterval(_ackTimeoutMilliseconds);
    m_ackTimeoutTimerUpload->setSingleShot(true);
    system("/bin/mkdir -p missions");
}
Vehicle* PlanController::vehicle(){
    return m_vehicle;
}
void PlanController::setVehicle(Vehicle* vehicle){
    m_vehicle = vehicle;
}
void PlanController::_startAckTimeout(AckType_t ack)
{
    switch (ack) {
    case AckMissionItem:
        // We are actively trying to get the mission item, so we don't want to wait as long.
        m_ackTimeoutTimerDownload->setInterval(_retryTimeoutMilliseconds);
        break;
    case AckNone:
        // FALLTHROUGH
    case AckMissionCount:
        // FALLTHROUGH
    case AckMissionRequest:
        // FALLTHROUGH
    case AckMissionClearAll:
        // FALLTHROUGH
    case AckGuidedItem:
        m_ackTimeoutTimerDownload->setInterval(_ackTimeoutMilliseconds);
        break;
    }

    m_expectedAck = ack;
    m_ackTimeoutTimerDownload->start();
}
bool PlanController::_checkForExpectedAck(AckType_t receivedAck)
{
    if (receivedAck == m_expectedAck) {
        m_expectedAck = AckNone;
        m_ackTimeoutTimerDownload->stop();
        return true;
    } else {
        if (m_expectedAck == AckNone){

        } else {

        }
        return false;
    }
}

void PlanController::_connectToMavlink(void)
{
    connect(m_vehicle, SIGNAL(mavlinkMessageReceived(mavlink_message_t)), this, SLOT(_mavlinkMessageReceived(mavlink_message_t)));
}

void PlanController::_disconnectFromMavlink(void)
{
    printf("PlanController %s\r\n",__func__);
    disconnect(m_vehicle, SIGNAL(mavlinkMessageReceived( mavlink_message_t)), this, SLOT(_mavlinkMessageReceived(mavlink_message_t)));
}
QString PlanController::_planTypeString(void)
{
    switch (m_planType) {
    case MAV_MISSION_TYPE_MISSION:
        return QStringLiteral("T:Mission");
    case MAV_MISSION_TYPE_FENCE:
        return QStringLiteral("T:GeoFence");
    case MAV_MISSION_TYPE_RALLY:
        return QStringLiteral("T:Rally");
    default:
        qWarning() << "Unknown plan type" << m_planType;
        return QStringLiteral("T:Unknown");
    }
}

void PlanController::_mavlinkMessageReceived(mavlink_message_t message)
{
//    printf("%s message.msgid=%d\r\n",__func__,message.msgid);
    switch (message.msgid) {
    case MAVLINK_MSG_ID_MISSION_COUNT:
        _handleMissionCount(message);
        break;

    case MAVLINK_MSG_ID_MISSION_ITEM:
        _handleMissionItem(message, false /* missionItemInt */);
        break;

    case MAVLINK_MSG_ID_MISSION_ITEM_INT:
        _handleMissionItem(message, true /* missionItemInt */);
        break;

    case MAVLINK_MSG_ID_MISSION_REQUEST:
        _handleMissionRequest(message, false /* missionItemInt */);
        break;

    case MAVLINK_MSG_ID_MISSION_REQUEST_INT:
        _handleMissionRequest(message, true /* missionItemInt */);
        break;

    case MAVLINK_MSG_ID_MISSION_ACK:
        _handleMissionAck(message);
        break;
    }
}
void PlanController::_sendError(ErrorCode_t errorCode, const QString& errorMsg)
{
    qCDebug(PlanManagerLog) << QStringLiteral("Sending %1 error").arg(_planTypeString()) << errorCode << errorMsg;

   error(errorCode, errorMsg);
}


void PlanController::_handleMissionAck(const mavlink_message_t& message){    
    Q_UNUSED(message);
    mavlink_mission_ack_t missionAck;
    mavlink_msg_mission_ack_decode(&message, &missionAck);
//#ifdef DEBUG
    printf("_handleMissionAck %d\r\n",missionAck.type);
//#endif
    if(missionAck.type == MAV_MISSION_ACCEPTED){
        m_ackTimeoutTimerDownload->stop();
        uploadMissionDone(true);
        _disconnectFromMavlink();
    }
}
void PlanController::_requestList(void)
{
    m_handleMissionCount = true;
    m_requestNextMissionItem = -1;
    mavlink_message_t message;
    mavlink_msg_mission_request_list_pack_chan(m_vehicle->communication()->systemId(),
                                               m_vehicle->communication()->componentId(),
                                               m_vehicle->communication()->mavlinkChannel(),
                                               &message,
                                               static_cast<uint8_t>(m_vehicle->id()),
                                               MAV_COMP_ID_AUTOPILOT1,
                                               static_cast<uint8_t>(m_planType));
//#ifdef DEBUG
    printf("Send message after mavlink_msg_mission_request_list_pack_chan\r\n");
//#endif
    m_vehicle->sendMessageOnLink(m_vehicle->m_com, message);
//    _startAckTimeout(AckMissionCount);
}
void PlanController::loadFromVehicle(void){
//    if (inProgress()) {
//        qCDebug(PlanManagerLog) << QStringLiteral("loadFromVehicle %1 called while transaction in progress").arg(_planTypeString());
//        return;
//    }
//    m_retryCount = 0;
//    _setTransactionInProgress(TransactionRead);
    Q_EMIT progressPct(0);
    m_retryCount = 0;
    m_ackTimeoutTimerDownload->stop();
    disconnect(m_ackTimeoutTimerDownload, SIGNAL(timeout()), this, SLOT(_ackDownloadTimeout()));
    m_ackTimeoutTimerDownload->setInterval(_ackTimeoutMilliseconds);
    m_ackTimeoutTimerDownload->start();
    connect(m_ackTimeoutTimerDownload, SIGNAL(timeout()), this, SLOT(_ackDownloadTimeout()));
    _connectToMavlink();
    _requestList();
}
void PlanController::_handleMissionCount(const mavlink_message_t& message){
#ifdef DEBUG
    printf("_handleMissionCount\r\n");
#endif
    if(m_handleMissionCount) m_handleMissionCount = false;
    else return;
    mavlink_mission_count_t missionCount;
    m_itemIndicesToRead.clear();
    m_missionItems.clear();
    mavlink_msg_mission_count_decode(&message, &missionCount);
//#ifdef DEBUG
    printf("missionCount.mission_type = %d\r\n",missionCount.mission_type);
    printf("missionCount.count = %d\r\n",missionCount.count);
    printf("missionCount.target_system = %d\r\n",missionCount.target_system);
    printf("missionCount.target_component = %d\r\n",missionCount.target_component);
//#endif
    if (missionCount.count == 0) {
        _readTransactionComplete();
    }else{
        for(int i=0; i< missionCount.count; i++){
            m_itemIndicesToRead.push_back(i);
        }
#ifdef DEBUG
        printf("m_itemIndicesToRead:");
        for(int i=0; i< m_itemIndicesToRead.size(); i++){
            printf(" %d",m_itemIndicesToRead.at(i));
        }
        printf("\r\n");
#endif
        if(m_itemIndicesToRead.size() > 0){
            _requestNextMissionItem(0);
        }
    }
}

void PlanController::_requestNextMissionItem(int sequence)
{
    if (m_itemIndicesToRead.count() == 0) {
        return;
    }
    if(m_requestNextMissionItem < sequence){
        m_requestNextMissionItem = sequence;
        m_handleMissionItem = 0;
    }else{
        return;
    }
//    printf("m_itemIndicesToRead:");
//    for(int i=0; i< m_itemIndicesToRead.size(); i++){
//        printf(" %d",m_itemIndicesToRead.at(i));
//    }
//    printf("\r\n");
#ifdef DEBUG
    printf("_requestNextMissionItem[%d]\r\n",m_itemIndicesToRead[0]);
#endif
    mavlink_message_t message;
    mavlink_msg_mission_request_int_pack_chan(m_vehicle->communication()->systemId(),
                                              m_vehicle->communication()->componentId(),
                                              m_vehicle->communication()->mavlinkChannel(),
                                              &message,
                                              static_cast<uint8_t>(m_vehicle->id()),
                                              MAV_COMP_ID_AUTOPILOT1,
                                              static_cast<uint16_t>(sequence),
                                              static_cast<uint8_t>(m_planType));
    m_vehicle->sendMessageOnLink(m_vehicle->m_com, message);
}
void PlanController::_handleMissionItem(const mavlink_message_t& message, bool missionItemInt){
    m_retryCount = 0;

    MAV_CMD     command;
    MAV_FRAME   frame;
    float      param1;
    float      param2;
    float      param3;
    float      param4;
    float      param5;
    float      param6;
    float      param7;
    bool        autoContinue;
    bool        isCurrentItem;
    int         seq;
    if (missionItemInt) {
        mavlink_mission_item_int_t missionItem;
        mavlink_msg_mission_item_int_decode(&message, &missionItem);

        command =       static_cast<MAV_CMD>(missionItem.command);
        frame  =        static_cast<MAV_FRAME>(missionItem.frame);
        param1 =        missionItem.param1;
        param2 =        missionItem.param2;
        param3 =        missionItem.param3;
        param4 =        missionItem.param4;
        param5 =        static_cast<float>(static_cast<double>(missionItem.x) / qPow(10.0, 7.0));
        param6 =        static_cast<float>(static_cast<double>(missionItem.y) / qPow(10.0, 7.0));
        param7 =        static_cast<float>(missionItem.z);
        autoContinue =  missionItem.autocontinue;
        isCurrentItem = missionItem.current;
        seq =           missionItem.seq;
//        printf("missionItem[%d].x=%f\r\n",seq,param5);
//        printf("missionItem[%d].y=%f\r\n",seq,param6);
//        printf("missionItem[%d].z=%f\r\n",seq,param7);
    } else {
        mavlink_mission_item_t missionItem;
        mavlink_msg_mission_item_decode(&message, &missionItem);

        command =       static_cast<MAV_CMD>(missionItem.command);
        frame =         static_cast<MAV_FRAME>(missionItem.frame);
        param1 =        missionItem.param1;
        param2 =        missionItem.param2;
        param3 =        missionItem.param3;
        param4 =        missionItem.param4;
        param5 =        missionItem.x;
        param6 =        missionItem.y;
        param7 =        missionItem.z;
        autoContinue =  missionItem.autocontinue;
        isCurrentItem = missionItem.current;
        seq =           missionItem.seq;
    }
#ifdef DEBUG
    printf("Received mission item seq[%d] \r\n",seq);
#endif
    if(m_requestNextMissionItem == seq){
        m_handleMissionItem ++;
        if(m_handleMissionItem>1){
#ifdef DEBUG
            printf("handleMissionItem[%d] %d time\r\n",seq,m_handleMissionItem);
#endif
            return;
        }else{
            #ifdef DEBUG
            printf("continue request MissionItem[%d]\r\n",seq);
#endif
        }
    }else{
        #ifdef DEBUG
        printf("continue request MissionItem[%d] because m_requestNextMissionItem != seq\r\n",seq);
#endif
    }
#ifdef DEBUG
    printf("_handleMissionItem[%d]\r\n",seq);
#endif
//    if(m_requestNextMissionItem == -1) m_requestNextMissionItem = seq;
//    else if(m_requestNextMissionItem == seq){
//        return;
//    }
    if (seq == 0) {
//        QGeoCoordinate newHomePosition(static_cast<double>(param5),
//                                       static_cast<double>(param6),
//                                       static_cast<double>(param7));
//        m_vehicle->setHomePosition(newHomePosition);
//        return;
    }
    if (m_itemIndicesToRead.contains(seq)) {
        m_itemIndicesToRead.removeOne(seq);

        if(seq == 0){

            printf("Home altitude = %f\r\n",static_cast<double>(param7));
            m_vehicle->setHomeAltitude(param7);
            m_vehicle->setHomePosition(QGeoCoordinate(
                                           static_cast<double>(param5),
                                           static_cast<double>(param6),
                                           static_cast<double>(m_vehicle->homePosition().altitude())));
            param7 = static_cast<float>(m_vehicle->homePosition().altitude());
        }
        MissionItem* item = new MissionItem(seq,
                                            command,
                                            frame,
                                            param1,
                                            param2,
                                            param3,
                                            param4,
                                            param5,
                                            param6,
                                            param7,
                                            autoContinue,
                                            isCurrentItem,
                                            this);

        m_missionItems.append(item);
        Q_EMIT progressPct(static_cast<float>(m_missionItems.count())/
                           static_cast<float>(m_missionItems.count()+m_itemIndicesToRead.count()));
    }
    if (m_itemIndicesToRead.count() == 0) {
        _readTransactionComplete();
    } else {
//        sleep(1);
        m_retryCount = 0;
        m_ackTimeoutTimerDownload->start();
        _requestNextMissionItem(seq+1);
    }
}
void PlanController::_readTransactionComplete(void)
{
#ifdef DEBUG
    printf("_readTransactionComplete read sequence complete\r\n");
#endif
    mavlink_message_t message;
    mavlink_msg_mission_ack_pack_chan(m_vehicle->communication()->systemId(),
                                      m_vehicle->communication()->componentId(),
                                      m_vehicle->communication()->mavlinkChannel(),
                                      &message,
                                      static_cast<uint8_t>(m_vehicle->id()),
                                      MAV_COMP_ID_AUTOPILOT1,
                                      MAV_MISSION_ACCEPTED,
                                      static_cast<uint8_t>(m_planType));
    m_vehicle->sendMessageOnLink(m_vehicle->m_com, message);
    Q_EMIT requestMissionDone(true);
    m_ackTimeoutTimerDownload->stop();
    _disconnectFromMavlink();
}
void PlanController::sendToVehicle(void){
    Q_EMIT progressPct(0);
    m_retryCount = 0;
    m_itemsSendCount = 0;
    m_itemsSendAccepted = 0;
    m_handleMissionWrite = true;
    m_writeNextMissionItem = 0;
    m_ackTimeoutTimerUpload->stop();
    disconnect(m_ackTimeoutTimerUpload, SIGNAL(timeout()), this, SLOT(_ackUploadTimeout()));
    m_ackTimeoutTimerUpload->setInterval(_ackTimeoutMilliseconds);
    m_ackTimeoutTimerUpload->start();
    connect(m_ackTimeoutTimerUpload, SIGNAL(timeout()), this, SLOT(_ackUploadTimeout()));
    _connectToMavlink();
    _writeMissionCount();
}
void PlanController::_writeMissionCount(void)
{
    m_itemsSendCount ++;
    if(m_handleMissionWrite) m_handleMissionWrite = false;
    else return;
    qCDebug(PlanManagerLog) << QStringLiteral("_writeMissionCount %1 count:_retryCount").arg(_planTypeString()) << m_writeMissionItems.count() << m_retryCount;
    mavlink_message_t message;
    mavlink_msg_mission_count_pack_chan(m_vehicle->communication()->systemId(),
                                        m_vehicle->communication()->componentId(),
                                        m_vehicle->communication()->mavlinkChannel(),
                                        &message,
                                        static_cast<uint8_t>(m_vehicle->id()),
                                        MAV_COMP_ID_AUTOPILOT1,
                                        static_cast<uint16_t>(m_writeMissionItems.count()),
                                        static_cast<uint8_t>(m_planType));

    m_vehicle->sendMessageOnLink(m_vehicle->communication(), message);
//    _startAckTimeout(AckMissionRequest);
}
void PlanController::_handleMissionRequest(const mavlink_message_t& message, bool missionItemInt){

//    if(m_itemsSendCount == 1){
//        m_itemsSendCount = 0;
//    }

    mavlink_mission_request_t missionRequest;
    mavlink_msg_mission_request_decode(&message, &missionRequest);
//    printf("%s missionRequest.seq=%d\r\n",__func__,missionRequest.seq);
    if (missionRequest.seq > m_writeMissionItems.count() - 1) {
        _sendError(RequestRangeError, tr("Vehicle requested item outside range, count:request %1:%2. Send to Vehicle failed.").arg(m_writeMissionItems.count()).arg(missionRequest.seq));
//        _finishTransaction(false);
        return;
    }

    if(missionRequest.seq == m_writeNextMissionItem){
        m_handleWriteMissionItem++;
    }else{
        return;
    }
    if(m_handleWriteMissionItem == 1) {
        m_handleWriteMissionItem = 0;
        m_writeNextMissionItem = missionRequest.seq + 1;
    }else{
        return;
    }
    m_itemsSendAccepted ++;
    #ifdef DEBUG
    printf("_handleMissionRequest [%d] %d times\r\n",missionRequest.seq,m_handleWriteMissionItem);
#endif

    _uploadNextMissionItem(missionRequest.seq,missionItemInt);
}
void PlanController::_uploadNextMissionItem(int sequence,bool missionItemInt){
//    printf("%s m_writeNextMissionItem=%d\r\n",__func__,m_writeNextMissionItem);
    MissionItem* item = m_writeMissionItems[sequence];
//    printf("%s missionItem[%d] sequence[%d]\r\n",__func__,
//           item->sequence(),
//           sequence);
    mavlink_message_t   messageOut;
    if (missionItemInt) {
        mavlink_msg_mission_item_int_pack_chan(m_vehicle->communication()->systemId(),
                                               m_vehicle->communication()->componentId(),
                                               m_vehicle->communication()->mavlinkChannel(),
                                               &messageOut,
                                               static_cast<uint8_t>(m_vehicle->id()),
                                               MAV_COMP_ID_AUTOPILOT1,
                                               sequence,
                                               static_cast<uint8_t>(item->frame()),
                                               static_cast<uint8_t>(item->command()),
                                               sequence == 0,
                                               item->autoContinue(),
                                               item->param1(),
                                               item->param2(),
                                               item->param3(),
                                               item->param4(),
                                               item->param5() * qPow(10.0, 7.0),
                                               item->param6() * qPow(10.0, 7.0),
                                               isnan(item->param7())?50:item->param7(),
                                               static_cast<uint8_t>(m_planType));
    }else {
        mavlink_msg_mission_item_pack_chan(m_vehicle->communication()->systemId(),
                                           m_vehicle->communication()->componentId(),
                                           m_vehicle->communication()->mavlinkChannel(),
                                           &messageOut,
                                           static_cast<uint8_t>(m_vehicle->id()),
                                           MAV_COMP_ID_AUTOPILOT1,
                                           sequence,
                                           static_cast<uint8_t>(item->frame()),
                                           static_cast<uint8_t>(item->command()),
                                           sequence == 0,
                                           item->autoContinue(),
                                           item->param1(),
                                           item->param2(),
                                           item->param3(),
                                           item->param4(),
                                           item->param5(),
                                           item->param6(),
                                           isnan(item->param7())?50:item->param7(),
                                           static_cast<uint8_t>(m_planType));
    }
    m_vehicle->sendMessageOnLink(m_vehicle->communication(), messageOut);
    m_itemsSendCount ++;
    m_ackTimeoutTimerUpload->setInterval(_ackTimeoutMilliseconds);
    m_ackTimeoutTimerUpload->start();
    Q_EMIT progressPct(static_cast<float>(m_writeNextMissionItem)/
                       static_cast<float>(m_writeMissionItems.count()));
}
void PlanController::_ackDownloadTimeout(void){
//    #ifdef DEBUG
    printf("_ackTimeout[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]] m_retryCount = %d\r\n",m_retryCount);
//    #endif
//        Q_EMIT readyToRequest();
//        disconnect(m_ackTimeoutTimerDownload, SIGNAL(timeout()), this, SLOT(_ackDownloadTimeout()));
    if (m_itemIndicesToRead.count() != 0){
        if(m_retryCount < _maxRetryCount){
            m_retryCount ++;
//#ifdef DEBUG
            printf("Retry =======================[%d] with %d\r\n",m_retryCount,m_itemIndicesToRead.count());
//#endif

//            _requestList();
            if(m_itemIndicesToRead.count() > 0){
                m_requestNextMissionItem = m_itemIndicesToRead[0]-1;
                m_ackTimeoutTimerDownload->setInterval(_retryTimeoutMilliseconds);
                m_ackTimeoutTimerDownload->start();
//                    connect(m_ackTimeoutTimerDownload, &QTimer::timeout, this, &PlanController::_ackTimeout);
                _requestNextMissionItem(m_itemIndicesToRead[0]);
            }
        }else{
            printf("Failed to get plan line 523\r\n");
//            _disconnectFromMavlink();
            Q_EMIT requestMissionDone(false);
        }
    }else{
//        printf("Failed to get plan line 527\r\n");
//        _disconnectFromMavlink();
//        Q_EMIT requestMissionDone(false);
        #ifdef DEBUG

        printf("m_itemIndicesToRead.count() == 0\r\n");
#endif
        if(m_retryCount < _maxRetryCount){
            m_retryCount ++;
            _requestList();
            m_ackTimeoutTimerDownload->setInterval(_ackTimeoutMilliseconds);
            m_ackTimeoutTimerDownload->start();
        }else{
            Q_EMIT requestMissionDone(false);
        }
    }
}
void PlanController::_ackUploadTimeout(void){
//    printf("%s m_writeNextMissionItem = %d , m_itemsSendAccepted = %d\r\n",__func__,
//           m_writeNextMissionItem,
//           m_itemsSendAccepted);
    if(m_itemsSendAccepted < m_writeNextMissionItem){
        if(m_retryCount < _maxRetryCount){
            m_retryCount ++;
            if(m_itemsSendCount <= 1){
                m_ackTimeoutTimerUpload->setInterval(_retryTimeoutMilliseconds);
                m_ackTimeoutTimerUpload->start();
                writeMissionItemsCount();
            }else{
                _uploadNextMissionItem(m_writeNextMissionItem,false);
            }
        }else{
            printf("Failed to upload plan time out\r\n");
//            _disconnectFromMavlink();
            Q_EMIT uploadMissionDone(false);
        }
    }else{
//        printf("Failed to upload plan\r\n");
//        Q_EMIT uploadMissionDone(false);
    }

}
void PlanController::readWaypointFile(QString file)
{
    int wp_count = 0;
    bool error = false;
    QVector<MissionItem*> cmds;
    FILE* fp = fopen(file.toStdString().c_str(), "r");
    if (fp == NULL){
        Q_EMIT requestMissionDone(false);
        return;
    }
    char* tmpLine = NULL;
    size_t len = 0;

    QString header;
    if(getline(&tmpLine, &len, fp)!= -1){
        header = QString::fromLocal8Bit(tmpLine);
    }else{
        Q_EMIT requestMissionDone(false);
        return;
    }

    while ((getline(&tmpLine, &len, fp)) != -1)
    {
        QString line = QString::fromLocal8Bit(tmpLine);
        // waypoints

        if (line.startsWith("#"))
            continue;
        line.replace(',','.');
        //seq/cur/frame/mode
        QRegExp separator("[(\t|,|\s|)]");
        QStringList items = line.split(separator);
//#ifdef DEBUG
        printf("Line[%d] ",wp_count+1);
        for(int i=0; i < items.size(); i++){
            printf("%s ",items[i].toStdString().c_str());
        }
        printf("\r\n");
//#endif
        if (items.size() <= 9)
            continue;

        try
        {
            // check to see if the first wp is index 0/home.
            // if it is not index 0, add a blank home point
            if (wp_count == 0 && items[0] != "0")
            {
                cmds.push_back(new MissionItem());
            }

            MissionItem *temp = new MissionItem();
            temp->setSequence(items[0].toInt());
            temp->setFrame(items[2].toInt());
            if (items[2] == "3")
            {
                // abs MAV_FRAME_GLOBAL_RELATIVE_ALT=3
                temp->setOption(1);
            }
            else if (items[2] == "10")
            {
                temp->setOption(8);
            }
            else
            {
                temp->setOption(0);
            }

            temp->setCommand(items[3].toInt());

            temp->setParam1(items[4].toFloat());
            temp->setParam2(items[5].toFloat());
            temp->setParam3(items[6].toFloat());
            temp->setParam4(items[7].toFloat());

            temp->setParam5(items[8].toFloat());
            temp->setParam6(items[9].toFloat());
            temp->setParam7(items[10].toFloat());
            temp->setAutoContinue(items[11].toInt());
            cmds.push_back(temp);

            wp_count++;
        }
        catch (const std::bad_alloc &)
        {

        }
    }
    fclose(fp);
    if (tmpLine)
        free(tmpLine);
    m_missionItems = cmds;
    Q_EMIT requestMissionDone(true);
    return;
}
void PlanController::writeWaypointFile(QString file){
    FILE *fptr;
    fptr = fopen(file.toStdString().c_str(), "w");
    fprintf(fptr,"QGC WPL 110\n");
    for(int i=0; i< m_missionItems.size(); i++){
        MissionItem *tmp = m_missionItems[i];
        char line[1024];
        sprintf(line,"%d\t%d\t%d\t%d\t%.08f\t%.08f\t%.08f\t%.08f\t%.08f\t%.08f\t%.06f\t%d\n",
                tmp->sequence(),tmp->sequence()==0?1:0,tmp->frame(),tmp->command(),
                tmp->param1(),tmp->param2(),tmp->param3(),tmp->param4(),
                tmp->param5(),tmp->param6(),tmp->param7(),tmp->autoContinue()?1:0);
        QString lineStr(line);
        lineStr.replace(',','.');
        fprintf(fptr,"%s",lineStr.toStdString().c_str());
    }
    fclose(fptr);
}
