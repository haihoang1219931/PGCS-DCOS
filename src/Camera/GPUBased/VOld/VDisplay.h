#ifndef VDISPLAY_H
#define VDISPLAY_H

#include "VDisplayWorker.h"
#include <QAbstractVideoSurface>
#include <QThread>
#include <QVideoSurfaceFormat>
#include <opencv2/opencv.hpp>
#include "../Video/VFrameGrabber.h"
#include "../Video/VPreprocess.h"
#include "../Video/VTrackWorker.h"
#include "../Video/VODWorker.h"
#include "../Video/VSavingWorker.h"
#include "../Video/VRTSPServer.h"
#include <QVariantList>
#include <QVariant>
#include "OD/yolo_v2_class.hpp"

class VDisplay : public QObject
{
        Q_OBJECT
        Q_PROPERTY(QAbstractVideoSurface *videoSurface READ videoSurface WRITE
                   setVideoSurface)
        Q_PROPERTY(QSize sourceSize READ sourceSize NOTIFY sourceSizeChanged)
        Q_PROPERTY(int frameID READ frameID)
        Q_PROPERTY(bool enOD READ enOD)
        Q_PROPERTY(bool enTrack READ enTrack)
        Q_PROPERTY(bool enSteer READ enSteer)

    public:
        explicit VDisplay(QObject *_parent = 0);
        ~VDisplay();

        QAbstractVideoSurface *videoSurface();
        void setVideoSurface(QAbstractVideoSurface *_videoSurface);
        QSize sourceSize();
        int frameID();
        void init();
        void stop();
        bool enOD()
        {
            return m_enOD;
        }
        bool enTrack()
        {
            return m_enTrack;
        }
        bool enSteer()
        {
            return m_enSteer;
        }

    Q_SIGNALS:
        void determinedTrackObjected(int _id, double _px, double _py, double _oW, double _oH, double _w, double _h);
        void objectLost();
    public Q_SLOTS:
        void onReceivedFrame(int _id, QVideoFrame _frame);
        void onReceivedFrame();

    public:
        Q_INVOKABLE void play();
        Q_INVOKABLE void setVideoSource(QString _ip, int _port);
        Q_INVOKABLE void searchByClass(QVariantList _classList);
        Q_INVOKABLE void setTrackAt(int _id, double _px, double _py, double _w, double _h);
        Q_INVOKABLE void disableObjectDetect();
        Q_INVOKABLE void enableObjectDetect();
        Q_INVOKABLE void setVideoSavingState(bool _state);
        Q_INVOKABLE void resetVideoSource(QString _ip, int _port);
        Q_INVOKABLE void enVisualLock();
        Q_INVOKABLE void disVisualLock();
        Q_INVOKABLE void setDigitalStab(bool _en);
        Q_INVOKABLE void setGimbalRecorder(bool _en);
        Q_INVOKABLE void changeTrackSize(int _val);

    Q_SIGNALS:
        void sourceSizeChanged(int newWidth, int newHeight);

    private:
        const QVideoFrame::PixelFormat VIDEO_OUTPUT_FORMAT =
            QVideoFrame::PixelFormat::Format_RGB32;
        QAbstractVideoSurface *m_videoSurface = nullptr;
        QSize m_sourceSize;
        QThread *m_threadEODisplay;
        VDisplayWorker *m_vDisplayWorker;
        VFrameGrabber *m_vFrameGrabber;
        VPreprocess *m_vPreprocess;
        VODWorker *m_vODWorker;
        VTrackWorker *m_vTrackWorker;
        VSavingWorker *m_vSavingWorker;
        VRTSPServer *m_vRTSPServer;
        int m_id;

        // OD
        Detector *m_detector;
        bool m_enSteer = false;
        bool m_enTrack = false;
        bool m_enOD = false;

        // CLicktrack
        Detector *m_clicktrackDetector;
};

#endif // VDISPLAY_H
