#include "UASMessage.h"

UASMessage::UASMessage(QObject *parent) : QObject(parent)
{

}
UASMessage::UASMessage(int componentid, int severity, QString text)
{
    _compId   = componentid;
    _severity = severity;
    _text     = text;
}

bool UASMessage::severityIsError()
{
    switch (_severity) {
        case MAV_SEVERITY_EMERGENCY:
        case MAV_SEVERITY_ALERT:
        case MAV_SEVERITY_CRITICAL:
        case MAV_SEVERITY_ERROR:
            return true;
        default:
            return false;
    }
}
