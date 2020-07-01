#include "VideoRender.h"

class VideoFboRender : public QQuickFramebufferObject::Renderer
{
public:
    VideoFboRender(){

    }

    ~VideoFboRender() override {

    }
public:
    void synchronize(QQuickFramebufferObject *item) override {
        m_i420Ptr = dynamic_cast<VideoRender*>(item)->m_data;
        m_videoH = dynamic_cast<VideoRender*>(item)->m_height;
        m_videoW = dynamic_cast<VideoRender*>(item)->m_width;
        m_warpMatrix = dynamic_cast<VideoRender*>(item)->m_warpMatrix;
        m_imageRendered = dynamic_cast<VideoRender*>(item)->m_dataRendered;
    }

    void render() override{
        if(m_i420Ptr == nullptr || m_videoH == 0 || m_videoW == 0) return;
        m_render.render(m_i420Ptr, m_videoW,m_videoH,m_warpMatrix);
//        this->framebufferObject()->toImage().save(QString::fromStdString(std::to_string(count))+".jpg");
//        count++;
//        m_imageRendered = this->framebufferObject()->toImage().bits();
//        memcpy(m_imageRendered,this->framebufferObject()->toImage().bits(),m_videoW*m_videoH*4);
//        printf("m_imageRendered = %p\r\n",m_imageRendered);
    }

    QOpenGLFramebufferObject *createFramebufferObject(const QSize &size) override {
        QOpenGLFramebufferObjectFormat format;
        format.setAttachment(QOpenGLFramebufferObject::CombinedDepthStencil);
        format.setSamples(1);
        return new QOpenGLFramebufferObject(QSize(1920,1080),format);
    }

private:
    uchar* m_imageRendered = nullptr;
    I420Render m_render;
    unsigned char *m_i420Ptr = nullptr;
    int m_videoW = 0;
    int m_videoH = 0;
    float* m_warpMatrix = nullptr;
    int count = 0;
};

VideoRender::VideoRender(QQuickItem *parent) : QQuickFramebufferObject(parent)
{
    setTextureFollowsItemSize(false);
}

VideoRender::~VideoRender()
{

}

QQuickFramebufferObject::Renderer *VideoRender::createRenderer() const
{
    return new VideoFboRender();
}

void VideoRender::handleNewFrame(const int &_id, unsigned char *_img, const int &_w, const int &_h,float* warpMatrix, unsigned char *_imgOut)
{
    this->m_data = _img;
    this->m_width = _w;
    this->m_height = _h;
    this->m_warpMatrix = warpMatrix;
    this->m_dataRendered = _imgOut;
    this->update();
}
uchar* VideoRender::image(){
    return m_dataRendered;
}

