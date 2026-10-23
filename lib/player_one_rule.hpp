#pragma once
// aleks2 (rc17): which of the two npads that feed the game's pad 0 (Handheld and No1, read together by SDL's Switch
// joystick) is player 1 -- the one vibration and six-axis go to. Pure, so a host test can check it
// (platforms/switch/tools/test-player-one-rule.sh); device_switch.cpp feeds it from libnx every frame.

namespace aurora::device::player_one_rule {

enum class Npad { None, No1, Handheld };

struct Seen {
  bool no1Connected = false;
  bool handheldConnected = false;
  bool no1Input = false;       // a button pressed or a stick off-centre since the last look
  bool handheldInput = false;
};

// One connected: that one. Both: the one with input now; with input on both or neither, the previous choice,
// and No1 at first (an external controller was connected for a reason). Neither: None.
inline Npad choose(const Seen& s, const Npad previous) {
  if (s.no1Connected && !s.handheldConnected) {
    return Npad::No1;
  }
  if (s.handheldConnected && !s.no1Connected) {
    return Npad::Handheld;
  }
  if (!s.no1Connected && !s.handheldConnected) {
    return Npad::None;
  }
  if (s.no1Input != s.handheldInput) {
    return s.no1Input ? Npad::No1 : Npad::Handheld;
  }
  return previous == Npad::None ? Npad::No1 : previous;
}

}  // namespace aurora::device::player_one_rule
