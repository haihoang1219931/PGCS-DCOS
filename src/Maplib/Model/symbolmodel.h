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
            AltRole,
            ToWPRole,
            RepeatRole,
            DirCircleRole,
            TimeCircleRole,
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


        Q_INVOKABLE void addSymbol(const int id,const int type,const int alt,const int toWP,const int repeat,const int dirCircle,const int timeCircle, const QGeoCoordinate coordinate);
        Q_INVOKABLE void editSymbol(const int id,const int type,const int alt,const int toWP,const int repeat,const int dirCircle,const int timeCircle, const QGeoCoordinate coordinate);
        Q_INVOKABLE void deleteSymbol(const int id);
        Q_INVOKABLE void moveSymbol(const int id,const QGeoCoordinate coordinate);
        Q_INVOKABLE void clearSymbol();
        Q_INVOKABLE void refreshModel();

        void refreshIndexSymbol();

        int rowCount(const QModelIndex &parent = QModelIndex()) const override;

        int columnCount(const QModelIndex& parent = QModelIndex()) const override;

        QVariant data(const QModelIndex &index, int role = IdRole)  const override;

        QHash<int, QByteArray> roleNames() const override{
            QHash<int, QByteArray> roles;
            roles[IdRole]   = "Id_Role";
            roles[TypeRole] = "Type_Role";
            roles[AltRole]  = "Alt_Role";
            roles[ToWPRole] = "ToWP_Role";
            roles[RepeatRole]    = "Repeat_Role";
            roles[DirCircleRole] = "DirCircle_Role";
            roles[TimeCircleRole]= "TimeCircle_Role";
            roles[CoordinateRole]= "Coordinate_Role";
            return roles;
        }

        Q_INVOKABLE QVariantMap get(int row);


    private:
        QList<symbol> _msymbol;

    Q_SIGNALS:
        void symbolModelChanged();


};

#endif // SYMBOLMODEL_H
