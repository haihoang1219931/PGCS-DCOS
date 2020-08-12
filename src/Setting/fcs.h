#ifndef FCS_H
#define FCS_H

#include "config.h"
class FCSConfig: public Config
{
    Q_OBJECT
public:
    explicit FCSConfig(Config *parent = nullptr);
    Q_INVOKABLE int changeData(QString data,QString value) override;
    Q_INVOKABLE void print() override;
};

#endif // FCS_H
