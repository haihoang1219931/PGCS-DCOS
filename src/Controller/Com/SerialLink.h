#ifndef SERIALLINK_H
#define SERIALLINK_H

class LinkInterface;
class SerialConfiguration;
class SerialLink;

#include <QObject>
#include <QThread>
#include <QMutex>
#include <QString>
#include "LinkInterface.h"
#ifdef __android__
#include "qserialport.h"
#else
#include <QSerialPort>
#endif
#include <QMetaType>
#include <QLoggingCategory>

// We use QSerialPort::SerialPortError in a signal so we must declare it as a meta type
Q_DECLARE_METATYPE(QSerialPort::SerialPortError)
Q_DECLARE_LOGGING_CATEGORY(SerialLinkLog)
class SerialConfiguration: public QObject
{
    Q_OBJECT

public:

    explicit SerialConfiguration(QObject *parent = nullptr);
    virtual ~SerialConfiguration();
    Q_PROPERTY(int      baud            READ baud               WRITE setBaud               NOTIFY baudChanged)
    Q_PROPERTY(int      dataBits        READ dataBits           WRITE setDataBits           NOTIFY dataBitsChanged)
    Q_PROPERTY(int      flowControl     READ flowControl        WRITE setFlowControl        NOTIFY flowControlChanged)
    Q_PROPERTY(int      stopBits        READ stopBits           WRITE setStopBits           NOTIFY stopBitsChanged)
    Q_PROPERTY(int      parity          READ parity             WRITE setParity             NOTIFY parityChanged)
    Q_PROPERTY(QString  portName        READ portName           WRITE setPortName           NOTIFY portNameChanged)
    Q_PROPERTY(QString  portDisplayName READ portDisplayName                                NOTIFY portDisplayNameChanged)
    Q_PROPERTY(bool     usbDirect       READ usbDirect          WRITE setUsbDirect          NOTIFY usbDirectChanged)        ///< true: direct usb connection to board

    int  baud()         { return _baud; }
    int  dataBits()     { return _dataBits; }
    int  flowControl()  { return _flowControl; }    ///< QSerialPort Enums
    int  stopBits()     { return _stopBits; }
    int  parity()       { return _parity; }         ///< QSerialPort Enums
    bool usbDirect()    { return _usbDirect; }

    const QString portName          () { return _portName; }
    const QString portDisplayName   () { return _portDisplayName; }

    void setBaud            (int baud);
    void setDataBits        (int databits);
    void setFlowControl     (int flowControl);          ///< QSerialPort Enums
    void setStopBits        (int stopBits);
    void setParity          (int parity);               ///< QSerialPort Enums
    void setPortName        (const QString& portName);
    void setUsbDirect       (bool usbDirect);

    static QStringList supportedBaudRates();
    static QString cleanPortDisplayname(const QString name);

    /// From LinkConfiguration
    void        updateSettings  ();
    QString     settingsURL     () { return "SerialSettings.qml"; }
    QString     settingsTitle   () { return tr("Serial Link Settings"); }

Q_SIGNALS:
    void baudChanged            ();
    void dataBitsChanged        ();
    void flowControlChanged     ();
    void stopBitsChanged        ();
    void parityChanged          ();
    void portNameChanged        ();
    void portDisplayNameChanged ();
    void usbDirectChanged       (bool usbDirect);

private:
    static void _initBaudRates();

private:
    int _baud;
    int _dataBits;
    int _flowControl;
    int _stopBits;
    int _parity;
    QString _portName;
    QString _portDisplayName;
    bool _usbDirect;
};
class SerialLink : public LinkInterface
{
    Q_OBJECT
public:
    // LinkInterface
    // Links are only created/destroyed by LinkManager so constructor/destructor is not public
    SerialLink(LinkInterface *parent = nullptr);
    ~SerialLink();
    bool isOpen() override;
    void loadConfig(Config* config) override;
    void sendData(vector<unsigned char> msg) override;
    void writeBytesSafe(const char *bytes, int length) override;
    QString getName() const;
    void    requestReset();
    bool    isConnected() const;
    qint64  getConnectionSpeed() const;
    SerialConfiguration* getSerialConfig() const { return _serialConfig; }
    // These are left unimplemented in order to cause linker errors which indicate incorrect usage of
    // connect/disconnect on link directly. All connect/disconnect calls should be made through LinkManager.

public Q_SLOTS:
    void linkError(QSerialPort::SerialPortError error);

protected:
    QSerialPort* _port;
    quint64 _bytesRead;
    int     _timeout;
    QMutex  _dataMutex;       // Mutex for reading data from _port
    QMutex  _writeMutex;      // Mutex for accessing the _transmitBuffer.
    QByteArray bufferBug;
public Q_SLOTS:
    void closeConnection() override;
    void connect2host() override;
    void readyRead() override;
    void connected() override;
    void disconnected() override;
    void connectionTimeout() override;
private:

    // Internal methods
    void _emitLinkError(const QString& errorMsg);
    bool _hardwareConnect(QSerialPort::SerialPortError& error, QString& errorString);
    bool _isBootloader();
    void _resetConfiguration();
    // Local data
    volatile bool       _stopp;
    volatile bool       _reqReset;
    QMutex              _stoppMutex;      // Mutex for accessing _stopp
    QByteArray          _transmitBuffer;  // An internal buffer for receiving data from member functions and actually transmitting them via the serial port.
    SerialConfiguration* _serialConfig;
Q_SIGNALS:
    void aboutToCloseFlag();
    void communicationUpdate(QString error, QString params);
    void communicationError(QString error, QString params);
};

#endif
