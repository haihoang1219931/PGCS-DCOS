#include "symbolmodel.h"

#include <QDebug>

void SymbolModel::addSymbol(const int id, const int type, const int param1, const int param2, const int param3, const int param4, const QString text, const QGeoCoordinate coordinate)
{
    //beginResetModel();
    int length = _msymbol.length();
    beginInsertRows(QModelIndex(),length,length);

    _msymbol.push_back(symbol(id,type,param1,param2,param3,param4,text,coordinate));

    endInsertRows();
    Q_EMIT symbolModelChanged();
    //endResetModel();
}

void SymbolModel::editSymbol(const int id, const int type, const int param1, const int param2, const int param3, const int param4, const QString text, const QGeoCoordinate coordinate)
{
    if(_msymbol.length()>id && id >= 0)
         _msymbol[id] = symbol(id,type,param1,param2,param3,param4,text,coordinate);
//    endResetModel();
    refreshModel();
    Q_EMIT symbolModelChanged();
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
    {
         _msymbol[id].Coordinate = coordinate;
         double alt =0;
         alt = coordinate.altitude();
        // printf("alt changed: %f",alt);
    }
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

            if ( role == Param1Role )
                return _msymbol[index.row()].Param1;

            if ( role == Param2Role )
                return _msymbol[index.row()].Param2;

            if ( role == Param3Role )
                return _msymbol[index.row()].Param3;

            if ( role == Param4Role )
                return _msymbol[index.row()].Param4;

            if ( role == TextRole )
                return _msymbol[index.row()].Text;

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

