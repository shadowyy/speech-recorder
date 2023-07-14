#include <portaudio.h>

#include <string>
#include <vector>

#include <iostream>
#include "devices.h"

namespace speechrecorder {

std::vector<Device> GetDevices() {
  std::vector<Device> result;
  Pa_Initialize();

  int count = Pa_GetDeviceCount();

  // This log helps updating the device list, for some reason
  std::cerr << "[speech-recorder] Found " << count << " devices." << std::endl;

  for (int i = 0; i < count; i++) {
    const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
    bool include = info->maxInputChannels > 0;

#ifdef WIN32
    if (strcmp(Pa_GetHostApiInfo(info->hostApi)->name, "MME") != 0) {
      include = false;
    }
#endif

    if (include) {
      result.emplace_back(i, info->name, Pa_GetHostApiInfo(info->hostApi)->name,
                          info->maxInputChannels, info->maxOutputChannels,
                          info->defaultSampleRate,
                          i == Pa_GetDefaultInputDevice(),
                          i == Pa_GetDefaultOutputDevice());
    }
  }
  Pa_Terminate();

  return result;
}

}  // namespace speechrecorder
