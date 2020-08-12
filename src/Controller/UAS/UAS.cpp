#include "UAS.h"
#include "UASMessage.h"
UAS::UAS(QObject *parent) : QObject(parent)
{

}
void UAS::handleTextMessage(int, int compId, int severity, QString text)
{
    // So first determine the styling based on the severity.
    QString style;
    switch (severity)
    {
    case MAV_SEVERITY_EMERGENCY:
    case MAV_SEVERITY_ALERT:
    case MAV_SEVERITY_CRITICAL:
    case MAV_SEVERITY_ERROR:
        style = QString("<#E>");
        _errorCount++;
        _errorCountTotal++;
        break;
    case MAV_SEVERITY_NOTICE:
    case MAV_SEVERITY_WARNING:
        style = QString("<#I>");
        _warningCount++;
        break;
    default:
        style = QString("<#N>");
        _normalCount++;
        break;
    }

    // And determine the text for the severitie
    QString severityText("");
    QString formatedColor("white");
    switch (severity)
    {
    case MAV_SEVERITY_EMERGENCY:
        severityText = QString(tr(" EMERGENCY:"));
        formatedColor = "red";
        break;
    case MAV_SEVERITY_ALERT:
        severityText = QString(tr(" ALERT:"));
        formatedColor = "red";
        break;
    case MAV_SEVERITY_CRITICAL:
        severityText = QString(tr(" Critical:"));
        formatedColor = "red";
        break;
    case MAV_SEVERITY_ERROR:
        severityText = QString(tr(" Error:"));
        formatedColor = "red";
        break;
    case MAV_SEVERITY_WARNING:
        severityText = QString(tr(" Warning:"));
        formatedColor = "orange";
        break;
    case MAV_SEVERITY_NOTICE:
        severityText = QString(tr(" Notice:"));
        formatedColor = "orange";
        break;
    case MAV_SEVERITY_INFO:
        severityText = QString(tr(" Info:"));
        break;
    case MAV_SEVERITY_DEBUG:
        severityText = QString(tr(" Debug:"));
        break;
    default:
        severityText = QString(tr(""));
        break;
    }

    // Finally preppend the properly-styled text with a timestamp.
    QString dateString = QDateTime::currentDateTime().toString("hh:mm:ss.zzz");
    UASMessage* message = new UASMessage(compId, severity, text);
    QString compString("");
    if (_multiComp) {
        compString = QString(" COMP:%1").arg(compId);
    }
    message->_setFormatedText(QString("<font style=\"%1\">[%2%3]%4 %5</font><br/>").arg(style).arg(dateString).arg(compString).arg(severityText).arg(text));
    message->_setFormatedColor(formatedColor);
    if (message->severityIsError()) {
        _latestError = severityText + " " + text;
    }
    _messages.append(message);
    int count = _messages.count();

}
