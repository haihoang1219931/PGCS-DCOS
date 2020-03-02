// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Copyright Drew Noakes 2013-2016

#include "JoystickController.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <sstream>
#include <unistd.h>

Joystick::Joystick()
{
    openPath("/dev/input/js0");
}

Joystick::Joystick(int joystickNumber)
{
    std::stringstream sstm;
    sstm << "/dev/input/js" << joystickNumber;
    m_devicePath = sstm.str();
    openPath(sstm.str());
}

Joystick::Joystick(std::string devicePath)
{
    m_devicePath = devicePath;
    openPath(devicePath);
}

Joystick::Joystick(std::string devicePath, bool blocking)
{
    m_devicePath = devicePath;
    openPath(devicePath, blocking);
}

void Joystick::openPath(std::string devicePath, bool blocking)
{
    m_devicePath = devicePath;
  // Open the device using either blocking or non-blocking
  _fd = open(devicePath.c_str(), blocking ? O_RDONLY : O_RDONLY | O_NONBLOCK);
  uint32_t version;
  uint8_t axes;
  uint8_t buttons;
  char name[256];
  if(_fd < 0){
      m_axes = 0;
      m_buttons = 0;
  }else{
      ioctl(_fd, JSIOCGNAME(256), name);
      ioctl(_fd, JSIOCGVERSION, &version);
      ioctl(_fd, JSIOCGAXES, &axes);
      ioctl(_fd, JSIOCGBUTTONS, &buttons);
      m_name = std::string(name);
      m_version = std::to_string(version);
      m_axes = axes;
      m_buttons = buttons;
  }
//  qDebug("Open the device using either blocking or non-blocking");
}
void Joystick::closePath(){
    close(_fd);
}
bool Joystick::sample(JoystickEvent* event)
{
  int bytes = read(_fd, event, sizeof(*event));

  if (bytes == -1)
    return false;

  // NOTE if this condition is not met, we're probably out of sync and this
  // Joystick instance is likely unusable
  return bytes == sizeof(*event);
}

bool Joystick::isFound()
{
  return _fd >= 0;
}
bool Joystick::isExist(){
    return FileController::isExists(m_devicePath);
}
Joystick::~Joystick()
{
    printf("Close joystick\r\n");
    close(_fd);
}

std::ostream& operator<<(std::ostream& os, const JoystickEvent& e)
{
  os << "type=" << static_cast<int>(e.type)
     << " number=" << static_cast<int>(e.number)
     << " value=" << static_cast<int>(e.value);
  return os;
}


