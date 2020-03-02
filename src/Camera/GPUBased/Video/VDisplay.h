#ifndef VDISPLAY_H
#define VDISPLAY_H

#include "VDisplayWorker.h"
#include <QAbstractVideoSurface>
#include <QThread>
#include <QVideoSurfaceFormat>
#include <opencv2/opencv.hpp>
#include "VFrameGrabber.h"
#include "VPreprocess.h"
#include "VTrackWorker.h"
#include "VODWorker.h"
#include "VRTSPServer.h"
#include "VSavingWorker.h"
#include <QVariantList>
#include <QVariant>
#include "Multitracker/PlateOCR.h"


class VDisplay : public QObject
{
        Q_OBJECT
        Q_PROPERTY(QAbstractVideoSurface *videoSurface READ videoSurface WRITE
                   setVideoSurface)
        Q_PROPERTY(QSize sourceSize READ sourceSize NOTIFY sourceSizeChanged)
        Q_PROPERTY(int frameID READ frameID)
    public:
        explicit VDisplay(QObject *_parent = 0);
        ~VDisplay();

        QAbstractVideoSurface *videoSurface();
        void setVideoSurface(QAbstractVideoSurface *_videoSurface);
        QSize sourceSize();
        int frameID();
        void init();

    Q_SIGNALS:
        void determinedTrackObjected(int _id, double _px, double _py, double _w, double _h, double _oW, double _oH);
        void determinedPlateOnTracking(QString _imgPath, QString _plateID);

    public Q_SLOTS:
        void onReceivedFrame(int _id, QVideoFrame _frame);
        void onReceivedFrame();
        void onDeterminedTrackObjected(int _id, double _px, double _py, double _w, double _h, double _oW, double _oH);
        void onDeterminedPlateOnTracking(QString _imgPath, QString _plateID);
        void onPipelineError(int _errorCode);

    public:
        Q_INVOKABLE void play();
        Q_INVOKABLE void setVideoSource(QString _ip, int _port);
        Q_INVOKABLE void searchByClass(QVariantList _classList);
        Q_INVOKABLE void setTrackAt(int _id, double _px, double _py, double _w,
                                    double _h);
        Q_INVOKABLE void disableObjectDetect();
        Q_INVOKABLE void enableObjectDetect();
        Q_INVOKABLE void setVideoSavingState(bool _state);

    Q_SIGNALS:
        void sourceSizeChanged(int newWidth, int newHeight);

    private:
        const QVideoFrame::PixelFormat VIDEO_OUTPUT_FORMAT =
            QVideoFrame::PixelFormat::Format_RGB32;
        QAbstractVideoSurface *m_videoSurface = nullptr;
        QSize m_sourceSize;
        QThread *m_threadDisplay;
        VDisplayWorker *m_vDisplayWorker;
        VFrameGrabber *m_vFrameGrabber;
        VPreprocess *m_vPreprocess;
        VODWorker *m_vODWorker;
        VTrackWorker *m_vTrackWorker;
        VSavingWorker *m_vSavingWorker;
        MultiTrack *m_mulTracker;
        PlateOCR *m_plateOCR;
        VRTSPServer *m_vRTSPServer;        
        int m_id;
};

#endif // VDISPLAY_H
