#ifndef VIDEORENDER_H
#define VIDEORENDER_H

#include <QQuickFramebufferObject>
#include <QOpenGLFramebufferObjectFormat>
#include "I420Render.h"

class VideoRender : public QQuickFramebufferObject
{
    Q_OBJECT
public:
    explicit VideoRender(QQuickItem* parent = nullptr);
    ~VideoRender() override;
    Renderer *createRenderer() const override;
    uchar* image();
public Q_SLOTS:
    void handleNewFrame(const int &_id, unsigned char *_img, const int &_w, const int &_h, float* warpMatrix, unsigned char *_imgOut);
public:
    unsigned char* m_dataRendered = nullptr;
    QString m_videoSource = "";
    unsigned char *m_data = nullptr;
    int m_width = 0;
    int m_height = 0;
    float* m_warpMatrix = nullptr;
    QRect m_drawPosition;
};

#endif // VIDEORENDER_H
