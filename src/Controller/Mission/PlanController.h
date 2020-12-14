#ifndef PLANCONTROLLER_H
#define PLANCONTROLLER_H

#include <QObject>
#include <mavlink_types.h>
#include <mavlink.h>
#include <QQmlListProperty>
#include <QVector>
#include <QLoggingCategory>
#include <QtQml>
#include <QtGui>
#include <QTimer>
class MissionItem;
class Vehicle;
class PlanController : public QObject
{
    Q_OBJECT
    Q_PROPERTY(QQmlListProperty<MissionItem> missionItems READ missionItems NOTIFY missionItemsChanged)
    Q_PROPERTY(QQmlListProperty<MissionItem> writeMissionItems READ writeMissionItems NOTIFY writeMissionItemsChanged)
public:
    PlanController(Vehicle* vehicle);
    // Mission Item download from AP
    QQmlListProperty<MissionItem> missionItems(){
        return QQmlListProperty<MissionItem>(this, this,
                                                &PlanController::appendMissionItem,
                                                &PlanController::missionItemsCount,
                                                &PlanController::missionItem,
                                                &PlanController::clearMissionItems);
    }
    void appendMissionItem(MissionItem* p) {
        m_missionItems.append(p);
        Q_EMIT missionItemsChanged();
    }
    int missionItemsCount() const{return m_missionItems.count();}
    MissionItem *missionItem(int index) const{ return m_missionItems.at(index);}
    void clearMissionItems() {
        m_missionItems.clear();
        Q_EMIT missionItemsChanged();
    }
    // Mission Item upload to AP
    QQmlListProperty<MissionItem> writeMissionItems(){
        return QQmlListProperty<MissionItem>(this, this,
                                                &PlanController::appendWriteMissionItem,
                                                &PlanController::writeMissionItemsCount,
                                                &PlanController::writeMissionItem,
                                                &PlanController::clearWriteMissionItems);
    }

    void appendWriteMissionItem(MissionItem* p) {
        m_writeMissionItems.append(p);
        Q_EMIT writeMissionItemsChanged();
    }
    int writeMissionItemsCount() const {return m_writeMissionItems.count();}
    MissionItem *writeMissionItem(int index) const{ return m_writeMissionItems.at(index);}
    void clearWriteMissionItems() {
        m_writeMissionItems.clear();
        Q_EMIT writeMissionItemsChanged();
    }
    bool inProgress(void) const {
        return m_transactionInProgress != TransactionNone;
    }
    /// Error codes returned in error signal
    typedef enum {
        InternalError,
        AckTimeoutError,        ///< Timed out waiting for response from vehicle
        ProtocolOrderError,     ///< Incorrect protocol sequence from vehicle
        RequestRangeError,      ///< Vehicle requested item out of range
        ItemMismatchError,      ///< Vehicle returned item with seq # different than requested
        VehicleError,           ///< Vehicle returned error
        MissingRequestsError,   ///< Vehicle did not request all items during write sequence
        MaxRetryExceeded,       ///< Retry failed
        MissionTypeMismatch,    ///< MAV_MISSION_TYPE does not match _planType
    } ErrorCode_t;

    // These values are public so the unit test can set appropriate signal wait times
    // When passively waiting for a mission process, use a longer timeout.
    static const int _ackTimeoutMilliseconds = 600;
    // When actively retrying to request mission items, use a shorter timeout instead.
    static const int _retryTimeoutMilliseconds = 600;
    static const int _maxRetryCount = 5;
protected:
    typedef enum {
        AckNone,            ///< State machine is idle
        AckMissionCount,    ///< MISSION_COUNT message expected
        AckMissionItem,     ///< MISSION_ITEM expected
        AckMissionRequest,  ///< MISSION_REQUEST is expected, or MISSION_ACK to end sequence
        AckMissionClearAll, ///< MISSION_CLEAR_ALL sent, MISSION_ACK is expected
        AckGuidedItem,      ///< MISSION_ACK expected in response to ArduPilot guided mode single item send
    } AckType_t;

    typedef enum {
        TransactionNone,
        TransactionRead,
        TransactionWrite,
        TransactionRemoveAll
    } TransactionType_t;
    void _startAckTimeout(AckType_t ack);
    bool _checkForExpectedAck(AckType_t receivedAck);

    void _connectToMavlink(void);
    void _disconnectFromMavlink(void);


    void _handleMissionAck(const mavlink_message_t& message);
//    void _finishTransaction(bool success, bool apmGuidedItemWrite = false);
    // request plan from AP
    void _requestList(void);
    void _handleMissionCount(const mavlink_message_t& message);
    void _requestNextMissionItem(int sequence);
    void _handleMissionItem(const mavlink_message_t& message, bool missionItemInt);
    void _readTransactionComplete(void);
    // write plan to AP
    void writeMissionItems(const QList<MissionItem*>& missionItems);
    void _writeMissionCount(void);
    void _handleMissionRequest(const mavlink_message_t& message, bool missionItemInt);
    void _uploadNextMissionItem(int sequence,bool missionItemInt);

    QString _planTypeString(void);
//    void _setTransactionInProgress(TransactionType_t type);
    void _sendError(ErrorCode_t errorCode, const QString& errorMsg);
//    QString _ackTypeToString(AckType_t ackType);
//    void _clearAndDeleteWriteMissionItems(void);
//    void _clearAndDeleteMissionItems(void);
//    void _removeAllWorker(void);
Q_SIGNALS:
    void readyToRequest();
    void missionItemsChanged();
    void writeMissionItemsChanged();
    void requestMissionDone(int valid);
    void uploadMissionDone(int valid);
    void inProgressChanged(bool inProgress);
    void progressPct(float progressPercentPct);
    void error(int errorCode, const QString& errorMsg);
    void removeAllComplete(bool error);
    void sendComplete(bool error);
public Q_SLOTS:
    void _mavlinkMessageReceived(mavlink_message_t message);
    void _ackDownloadTimeout(void);
    void _ackUploadTimeout(void);
    void loadFromVehicle(void);
    void sendToVehicle(void);
    void readWaypointFile(QString file);
    void writeWaypointFile(QString file);
public:
    Vehicle* m_vehicle;
    QVector<MissionItem*> m_missionItems;          ///< Set of mission items on vehicle
    QVector<MissionItem*> m_writeMissionItems;     ///< Set of mission items currently being written to vehicle
protected:
    MAV_MISSION_TYPE    m_planType;

    QTimer*             m_ackTimeoutTimerUpload;
    QTimer*             m_ackTimeoutTimerDownload;
    AckType_t           m_expectedAck;
    int                 m_retryCount;

    TransactionType_t   m_transactionInProgress;
    bool                m_resumeMission;
    QList<int>          m_itemIndicesToRead;     ///< List of mission items which still need to be requested from vehicle
    int                 m_lastMissionRequest;    ///< Index of item last requested by MISSION_REQUEST
    int                 m_missionItemCountToRead;///< Count of all mission items to read

    int                 m_currentMissionIndex;
    int                 m_lastCurrentIndex;

    bool                m_handleMissionCount = false;
    int                 m_requestNextMissionItem = -1;
    int                 m_handleMissionItem = -1;
    bool                m_handleMissionWrite = false;
    int                 m_writeNextMissionItem = -1;
    int                 m_handleWriteMissionItem = 0;
    int                 m_itemsSendCount = 0;
    int                 m_itemsSendAccepted = 0;
private:
    static void appendMissionItem(QQmlListProperty<MissionItem>* list, MissionItem* p) {
        reinterpret_cast<PlanController* >(list->data)->appendMissionItem(p);
    }
    static void clearMissionItems(QQmlListProperty<MissionItem>* list) {
        reinterpret_cast<PlanController* >(list->data)->clearMissionItems();
    }
    static MissionItem* missionItem(QQmlListProperty<MissionItem>* list, int i) {
        return reinterpret_cast<PlanController* >(list->data)->missionItem(i);
    }
    static int missionItemsCount(QQmlListProperty<MissionItem>* list) {
        return reinterpret_cast<PlanController* >(list->data)->missionItemsCount();
    }

    static void appendWriteMissionItem(QQmlListProperty<MissionItem>* list, MissionItem* p) {
        reinterpret_cast<PlanController* >(list->data)->appendWriteMissionItem(p);
    }
    static void clearWriteMissionItems(QQmlListProperty<MissionItem>* list) {
        reinterpret_cast<PlanController* >(list->data)->clearWriteMissionItems();
    }
    static MissionItem* writeMissionItem(QQmlListProperty<MissionItem>* list, int i) {
        return reinterpret_cast<PlanController* >(list->data)->writeMissionItem(i);
    }
    static int writeMissionItemsCount(QQmlListProperty<MissionItem>* list) {
        return reinterpret_cast<PlanController* >(list->data)->writeMissionItemsCount();
    }
};

#endif // PLANCONTROLLER_H
