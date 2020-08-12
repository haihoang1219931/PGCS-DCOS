#ifndef UAS_H
#define UAS_H

#include <QObject>
#include <QQmlListProperty>
#include <QList>
#include <QDateTime>
#include "UASMessage.h"
class UASMessage;

class UAS : public QObject
{
    Q_OBJECT
    Q_PROPERTY(QQmlListProperty<UASMessage> messages READ messages NOTIFY messagesChanged)
public:
    explicit UAS(QObject *parent = nullptr);
    // Mission Item download from AP
    QQmlListProperty<UASMessage> messages(){
        return QQmlListProperty<UASMessage>(this, this,
                                                &UAS::appendMessage,
                                                &UAS::messagesCount,
                                                &UAS::messages,
                                                &UAS::clearMessages);
    }
    void appendMessage(UASMessage* p) {
        _messages.append(p);
        Q_EMIT messagesChanged();
    }
    int messagesCount() const{return _messages.count();}
    UASMessage *messages(int index) const{ return _messages.at(index);}
    void clearMessages() {
        _messages.clear();
        Q_EMIT messagesChanged();
    }
Q_SIGNALS:
    void messagesChanged();
public Q_SLOTS:
    void handleTextMessage(int uasid, int componentid, int severity, QString text);
public:
    int                     _activeComponent;
    bool                    _multiComp;
    QList<UASMessage*>      _messages;
    int                     _errorCount;
    int                     _errorCountTotal;
    int                     _warningCount;
    int                     _normalCount;
    QString                 _latestError;
    bool                    _showErrorsInToolbar;
private:
    static void appendMessage(QQmlListProperty<UASMessage>* list, UASMessage* p) {
        reinterpret_cast<UAS* >(list->data)->appendMessage(p);
    }
    static void clearMessages(QQmlListProperty<UASMessage>* list) {
        reinterpret_cast<UAS* >(list->data)->clearMessages();
    }
    static UASMessage* messages(QQmlListProperty<UASMessage>* list, int i) {
        return reinterpret_cast<UAS* >(list->data)->messages(i);
    }
    static int messagesCount(QQmlListProperty<UASMessage>* list) {
        return reinterpret_cast<UAS* >(list->data)->messagesCount();
    }
};

#endif // UAS_H
