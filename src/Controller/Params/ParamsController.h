#ifndef PARAMSCONTROLLER_H
#define PARAMSCONTROLLER_H

#include <QObject>
#include <QTimer>
#include <QVariantMap>
#include <QVariant>
#include <mavlink_types.h>
#include <mavlink.h>
//#define DEBUG_FUNC
class Vehicle;
class FirmwarePlugin;
class ParamsController : public QObject
{
    Q_OBJECT
    Q_PROPERTY(bool     parametersReady     READ parametersReady    NOTIFY parametersReadyChanged)      ///< true: Parameters are ready for use
    Q_PROPERTY(bool     missingParameters   READ missingParameters  NOTIFY missingParametersChanged)    ///< true: Parameters are missing from firmware response, false: all parameters received from firmware
    Q_PROPERTY(double   loadProgress        READ loadProgress       NOTIFY loadProgressChanged)
public:
    explicit ParamsController(QObject *parent = nullptr){ Q_UNUSED(parent);}
    ParamsController(Vehicle *vehicle);
    ~ParamsController();
    bool parametersReady    (void) const { return _parametersReady; }
    bool missingParameters  (void) const { return _missingParameters; }
    float loadProgress     (void) const { return _loadProgress; }

    QList<int> componentIds(void);

    /// Re-request the full set of parameters from the autopilot
    void refreshAllParameters(uint8_t componentID = MAV_COMP_ID_ALL);
    QVariant getParam(QString paramName);
    bool containKey(QString paramName);
Q_SIGNALS:
    void parametersReadyChanged(bool parametersReady);
    void missingParametersChanged(bool missingParameters);
    void loadProgressChanged(float value);
    void paramChanged(QString paramName);
public Q_SLOTS:
    void _handleMessageReceived(mavlink_message_t msg);
    void _waitingParamTimeout(void);
    void _updateParamTimeout(void);
    void _initialRequestMissingParams(void);
    void _readParameterRaw(const QString& paramName, int paramIndex);
    void _writeParameterRaw(const QString& paramName, const QVariant& value);
private:
    QVariant _convertParamValue(mavlink_param_value_t paramValue);
    QString _convertParamID(char* param_id);
    void _setLoadProgress(float loadProgress);
    void _handleParamRequest(mavlink_message_t msg);
    void _handleParamRequestList(mavlink_message_t msg);
    void _handleParamRequestRead(mavlink_message_t msg);
public:
    Vehicle*    _vehicle = nullptr;
    const int   _maxRetry = 3;
    int         _lastParamsReceivedCount = -1;
    int         _loadRetry = 0;
    float       _loadProgress = 0;                  ///< Parameter load progess, [0.0,1.0]
    bool        _parametersReady = false;               ///< true: parameter load complete
    bool        _missingParameters = false;             ///< true: parameter missing from initial load
    bool        _startLoadAllParams = false;           ///< true: Initial load of all parameters complete, whether successful or not
    bool        _waitingForDefaultComponent = false;    ///< true: last chance wait for default component params
    bool        _saveRequired = false;                  ///< true: _saveToEEPROM should be called
    bool        _metaDataAddedToFacts = false;          ///< true: FactMetaData has been adde to the default component facts
    bool        _logReplay = false;                     ///< true: running with log replay link
    int         _totalParamCount = 0;   ///< Number of parameters across all components
    int         _paramsReceivedCount = 0;   ///< Number of parameters across all components
    int         _totalLoopParamFail=0;
    QMap<int, int>      _paramCountMap;             ///< Key: Component id, Value: count of parameters in this component
    typedef QPair<QString,mavlink_param_value_t> ParamTypeVal;
    QMap<int, ParamTypeVal> _debugCacheMap;
    QMap<QString, mavlink_param_value_t> _paramMap;
    QTimer _waitingParamTimeoutTimer;
    QTimer _updateParamTimer;
};

#endif // PARAMSCONTROLLER_H
