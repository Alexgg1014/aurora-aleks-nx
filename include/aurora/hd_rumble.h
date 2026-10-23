#pragma once
// aleks2 (V1-rc16): HD Rumble on the Switch's own controllers (handheld, a Joy-Con pair, one Joy-Con, a Pro
// Controller). The game describes a vibration as an envelope of keyframes -- amplitude and frequency for the low and
// the high band, and a left/right balance -- and a thread plays it at ~5 ms resolution through
// hidSendVibrationValues. Two layers: a one-shot (a hit) and a loop (a quake, a gallop); when both play, each band
// takes the stronger of the two. Elsewhere, and for a pad that is not the console's own (a GameCube adapter keeps its
// motor), every call is a no-op and aurora_hd_rumble_available() is false.
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AuroraHdKey {
  uint16_t ms;       // how long until the next key (the last key holds this long, then the layer ends or loops)
  float amp_low;     // 0..1
  float freq_low;    // Hz, ~40..640 (the low band)
  float amp_high;    // 0..1
  float freq_high;   // Hz, ~80..1280 (the high band)
  float pan;         // -1 left motor only .. 0 both .. +1 right motor only
} AuroraHdKey;

enum { AURORA_HD_LAYER_SHOT = 0, AURORA_HD_LAYER_LOOP = 1 };

// Whether pad `chan` vibrates through the console's own controllers (and so takes HD envelopes).
bool aurora_hd_rumble_available(uint32_t chan);
// Replaces the layer's envelope (count 0 clears it). Game thread.
void aurora_hd_rumble_play(int layer, const AuroraHdKey* keys, int count, bool loop);
// Both layers off and the motors still. Any thread.
void aurora_hd_rumble_stop(void);
// 0..1: scales every amplitude sent to the console's controllers, HD and the classic on/off rumble alike.
void aurora_rumble_set_gain(float gain);
// aleks2 (rc17): once per frame, game thread. Follows which of Handheld and No1 is player 1 (the one in use), so
// vibration and six-axis go to that controller; a change moves the motors at once.
void aurora_player_one_tick(void);

#ifdef __cplusplus
}
#endif
