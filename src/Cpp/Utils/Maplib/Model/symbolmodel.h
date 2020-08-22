#ifndef SYMBOLMODEL_H
#define SYMBOLMODEL_H

#include <QAbstractListModel>
#include <QGeoCoordinate>
#include <QString>
#include <QObject>
#include "symbol.h"



class SymbolModel : public QAbstractListModel
{
    Q_OBJECT
    Q_PROPERTY(NOTIFY symbolModelChanging)

    public:
        using QAbstractListModel::QAbstractListModel;
        enum SymbolRoles
        {
            IdRole = Qt::DisplayRole + 1,
            TypeRole,
            Param1Role,
            Param2Role,
            Param3Role,
            Param4Role,
            TextRole,
            CoordinateRole
        };

        enum WPType
        {
            Waypoint=16,
            LoiterTime=19,
            Land=21,
            TakeOff=22,
            VTOLTakeOff=84,
            VTOLLand=85
        };


        Q_INVOKABLE void addSymbol(const int id,const int type,const int param1,const int param2,const int param3,const int param4,const QString text,const QGeoCoordinate coordinate);
        Q_INVOKABLE void editSymbol(const int id,const int type,const int param1,const int param2,const int param3,const int param4,const QString text,const QGeoCoordinate coordinate);

        Q_INVOKABLE void insertSymbol(const int id,const int type,const int param1,const int param2,const int param3,const int param4,const QString text,const QGeoCoordinate coordinate);

        Q_INVOKABLE void deleteSymbol(const int id);
        Q_INVOKABLE void moveSymbol(const int id,const QGeoCoordinate coordinate);
        Q_INVOKABLE void clearSymbol();
        Q_INVOKABLE void refreshModel();

        Q_INVOKABLE void scrollUp(const int id);
        Q_INVOKABLE void scrollDown(const int id);

        Q_INVOKABLE int getTotalDistance();

        void refreshIndexSymbol();

        int rowCount(const QModelIndex &parent = QModelIndex()) const override;

        int columnCount(const QModelIndex& parent = QModelIndex()) const override;

        QVariant data(const QModelIndex &index, int role = IdRole)  const override;

        QHash<int, QByteArray> roleNames() const override{
            QHash<int, QByteArray> roles;
            roles[IdRole]        = "Id_Role";
            roles[TypeRole]      = "Type_Role";
            roles[Param1Role]    = "Param1_Role";
            roles[Param2Role]    = "Param2_Role";
            roles[Param3Role]    = "Param3_Role";
            roles[Param4Role]    = "Param4_Role";
            roles[TextRole]      = "Text_Role";
            roles[CoordinateRole]= "Coordinate_Role";
            return roles;
        }

        Q_INVOKABLE QVariantMap get(int row);


    private:
        QList<symbol> _msymbol;
        bool isWaypoint(symbol obj);

    Q_SIGNALS:
        void symbolModelChanged();


};

#endif // SYMBOLMODEL_H
