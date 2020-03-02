#include "symbolmodel.h"

#include <QDebug>
void SymbolModel::addSymbol(const int id,const int type,const int alt,const int toWP,const int repeat,const int dirCircle,const int timeCircle, const QGeoCoordinate coordinate)
{
    //beginResetModel();
    int length = _msymbol.length();
    beginInsertRows(QModelIndex(),length,length);

    _msymbol.push_back(symbol(id,type,alt,toWP,repeat,dirCircle,timeCircle,coordinate));

    endInsertRows();
    Q_EMIT symbolModelChanged();
    //endResetModel();
}

void SymbolModel::editSymbol(const int id, const int type, const int alt, const int toWP, const int repeat, const int dirCircle, const int timeCircle, const QGeoCoordinate coordinate)
{
//    if(id>-1)
//    {   beginResetModel();
//        _msymbol.removeAt(id);
//        refreshIndexSymbol();
//        endResetModel();
//        Q_EMIT symbolModelChanged();
//    }
}

void SymbolModel::deleteSymbol(const int id)
{
    if(id>-1)
    {   beginResetModel();
        _msymbol.removeAt(id);
        refreshIndexSymbol();
        endResetModel();
        Q_EMIT symbolModelChanged();
    }
}


void SymbolModel::moveSymbol(const int id,const QGeoCoordinate coordinate)
{
//    beginResetModel();
    if(_msymbol.length()>id)
         _msymbol[id].Coordinate = coordinate;
//    endResetModel();
    Q_EMIT symbolModelChanged();
}

void SymbolModel::clearSymbol()
{
    beginResetModel();
    _msymbol.clear();
    endResetModel();
    Q_EMIT symbolModelChanged();
}

void SymbolModel::refreshModel()
{

    beginResetModel();
    endResetModel();
    Q_EMIT symbolModelChanged();
}


void SymbolModel::refreshIndexSymbol()
{
    for(int i=0;i<_msymbol.count();i++)
    {
        _msymbol[i].Id=i;
    }
}


int SymbolModel::rowCount(const QModelIndex &parent) const
{
    return _msymbol.count();
}

int SymbolModel::columnCount(const QModelIndex &parent) const
{
    return  8;
}

QVariant SymbolModel::data(const QModelIndex &index, int role) const
{
            if (!index.isValid())
                    return QVariant();

            if ( role == IdRole)
                return _msymbol[index.row()].Id;

            if ( role == TypeRole )
                return _msymbol[index.row()].Type;

            if ( role == AltRole )
                return _msymbol[index.row()].Alt;

            if ( role == ToWPRole )
                return _msymbol[index.row()].ToWP;

            if ( role == RepeatRole )
                return _msymbol[index.row()].Repeat;

            if ( role == DirCircleRole )
                return _msymbol[index.row()].DirCircle;

            if ( role == TimeCircleRole )
                return _msymbol[index.row()].TimeCircle;

            if ( role == CoordinateRole )
                return QVariant::fromValue(_msymbol[index.row()].Coordinate);

        return QVariant();
}

QVariantMap SymbolModel::get(int row)
{
                QHash<int,QByteArray> names = roleNames();
                QHashIterator<int, QByteArray> i(names);
                QVariantMap res;
                int j=0;
                while (i.hasNext()) {
                    j++;
                    i.next();
                    QModelIndex idx = index(row, 0);
                    QVariant data = SymbolModel::data(idx,i.key());
                    res[i.value()] = data;
                    //cout << i.key() << ": " << i.value() << endl;

                }
//                 qDebug() <<QString("j:%1").arg(j);
//                 qDebug() <<res;
                return res;
}

