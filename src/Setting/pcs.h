#ifndef PCS_H
#define PCS_H

#include "config.h"
class PCSConfig: public Config
{
    Q_OBJECT
public:
    explicit PCSConfig(Config *parent = nullptr);
    Q_INVOKABLE int changeData(QString data,QString value) override;
    Q_INVOKABLE void print() override;
};

#endif // PCS_H
