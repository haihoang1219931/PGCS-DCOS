#ifndef SYSTEMSTATUS_H
#define SYSTEMSTATUS_H

#include "Common_type.h"
#include "EyeStatus.h"
#include "IPCStatusResponse.h"
#include "MotionCStatus.h"
#include "Object.h"
#include "Telemetry.h"
#include "utils.h"

namespace Eye {
class SystemStatus : public Object {
public:
  SystemStatus() { m_idx = -1; }
  ~SystemStatus() {}

  index_type getIndex() { return m_idx; }
  Telemetry getTelemetry() { return m_tele; }
  IPCStatusResponse getIPCStatus() { return m_ipcStatus; }
  MotionCStatus getMotionCStatus() { return m_motionCStatus; }
  void setIndex(index_type _idx) { m_idx = _idx; }
  void setTelemetry(Telemetry _tele) { m_tele = _tele; }
  void setIPCStatus(IPCStatusResponse _ipcStatus) { m_ipcStatus = _ipcStatus; }
  void setMotionCStatus(MotionCStatus _motionCStatus) {
    m_motionCStatus = _motionCStatus;
  }

  std::vector<byte> toByte() {
    std::vector<byte> res, b_tmp;
    res = Utils::toByte<index_type>(m_idx);
    b_tmp = m_tele.toByte();
    res.insert(res.end(), b_tmp.begin(), b_tmp.end());
    b_tmp = m_ipcStatus.toByte();
    res.insert(res.end(), b_tmp.begin(), b_tmp.end());
    b_tmp = m_motionCStatus.toByte();
    res.insert(res.end(), b_tmp.begin(), b_tmp.end());
    return res;
  }

  SystemStatus *parse(std::vector<byte> _data, index_type _index = 0) {
    byte *data = _data.data();
    m_idx = Utils::toValue<index_type>(data, _index);
    m_tele.parse(data, _index + sizeof(index_type));
    m_ipcStatus.parse(data, _index + sizeof(index_type) + m_tele.size());
    m_motionCStatus.parse(data, _index + sizeof(index_type) + m_tele.size() +
                                    m_ipcStatus.size());
    return this;
  }

  SystemStatus *parse(byte *_data, index_type _index = 0) {
    byte *data = _data;
    m_idx = Utils::toValue<index_type>(data, _index);
    m_tele.parse(data, _index + sizeof(index_type));
    m_ipcStatus.parse(data, _index + sizeof(index_type) + m_tele.size());
    m_motionCStatus.parse(data, _index + sizeof(index_type) + m_tele.size() +
                                    m_ipcStatus.size());
    return this;
  }

  SystemStatus &operator=(const SystemStatus &_systemStatus) {
    m_idx = _systemStatus.m_idx;
    m_tele = _systemStatus.m_tele;
    m_ipcStatus = _systemStatus.m_ipcStatus;
    m_motionCStatus = _systemStatus.m_motionCStatus;
  }

  length_type size() {
    return (sizeof(index_type) + m_tele.size() + m_ipcStatus.size() +
            m_motionCStatus.size());
  }

private:
  index_type m_idx;
  Telemetry m_tele;
  IPCStatusResponse m_ipcStatus;
  MotionCStatus m_motionCStatus;
};
} // namespace Eye

#endif // SYSTEMSTATUS_H
