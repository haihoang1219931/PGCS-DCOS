#ifndef UASMESSAGE_H
#define UASMESSAGE_H

#include <QObject>
#include "../Com/QGCMAVLink.h"
class UASMessage : public QObject
{
    Q_OBJECT
    Q_PROPERTY(QString text READ getText NOTIFY textChanged)
    Q_PROPERTY(QString formatedText READ getFormatedText NOTIFY formatedTextChanged)
    Q_PROPERTY(QString formatedColor READ getFormatedColor NOTIFY formatedColorChanged)
public:
    explicit UASMessage(QObject *parent = nullptr);
    UASMessage(int componentid, int severity, QString text);
    void _setFormatedText(const QString formatedText) { _formatedText = formatedText; }
    void _setFormatedColor(const QString formatedColor) { _formatedColor = formatedColor; }
    /**
     * @brief Get message source component ID
     */
    int getComponentID()        { return _compId; }
    /**
     * @brief Get message severity (from MAV_SEVERITY_XXX enum)
     */
    int getSeverity()           { return _severity; }
    /**
     * @brief Get message text (e.g. "[pm] sending list")
     */
    QString getText()           { return _text; }
    /**
     * @brief Get (html) formatted text (in the form: "[11:44:21.137 - COMP:50] Info: [pm] sending list")
     */
    QString getFormatedText()   { return _formatedText; }
    /**
     * @brief Get (html) formatted color (red,white,orange)
     */
    QString getFormatedColor()   { return _formatedColor; }
    /**
     * @return true: This message is a of a severity which is considered an error
     */
    bool severityIsError();
Q_SIGNALS:
    void textChanged();
    void formatedTextChanged();
    void formatedColorChanged();
private:
    int _compId;
    int _severity;
    QString _text;
    QString _formatedText;
    QString _formatedColor;
};

#endif // UASMESSAGE_H
