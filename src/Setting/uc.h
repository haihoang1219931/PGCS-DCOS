#ifndef UC_H
#define UC_H

#include "config.h"
class UCConfig: public Config
{
    Q_OBJECT
public:
    explicit UCConfig(Config *parent = nullptr);
    Q_INVOKABLE int changeData(QString data,QString value) override;
    Q_INVOKABLE void print() override;

};
#endif // UC_H
