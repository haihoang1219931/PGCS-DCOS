#include "ImageItem.h"

ImageItem::ImageItem(QQuickItem *parent) : QQuickPaintedItem(parent)
{
this->current_image = QImage(":/images/no_image.png");
}

void ImageItem::paint(QPainter *painter)
{
    QRectF bounding_rect = boundingRect();
    if(!current_image.isNull()){
        QPointF center = bounding_rect.center() - this->current_image.rect().center();
        if(center.x() < 0)
            center.setX(0);
        if(center.y() < 0)
            center.setY(0);
        center.setX(center.x()-1);
        center.setY(center.y()-1);
        painter->drawImage(center, this->current_image);
    }
}

QImage ImageItem::image() const
{    return this->current_image;
}

void ImageItem::setImage(const QImage &image)
{
    this->current_image = image;
    update();
}
