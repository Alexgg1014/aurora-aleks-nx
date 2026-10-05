#include <aurora/aux_layout_switch.hpp>
#include <cstdio>
namespace lay = aurora::auxwin::layout;
static int fails = 0;
static void chk(bool ok, const char* what) {
  std::printf("%s  %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) ++fails;
}
int main() {
  const int W = 1280, H = 720;

  // --- Normal returns a ZEROED layout on purpose -------------------------
  lay::set_mode(lay::kNormal);
  lay::Layout n{};
  lay::compute(W, H, 4.f/3.f, &n);
  chk(n.mode == lay::kNormal && !n.game.valid() && !n.companion.valid(),
      "Normal: zeroed layout (never routes through compute)");

  // --- Flip, 4:3, companion at 100% --------------------------------------
  lay::set_mode(lay::kFlip270);
  lay::set_params(100, 0);
  lay::Layout f{};
  lay::compute(W, H, 4.f/3.f, &f);
  std::printf("     game=%d,%d %dx%d   companion=%d,%d %dx%d  (canvas %dx%d rot %d)\n",
              f.game.x,f.game.y,f.game.w,f.game.h,
              f.companion.x,f.companion.y,f.companion.w,f.companion.h,
              f.logicalW,f.logicalH,f.rotationDegrees);
  chk(f.logicalW==720 && f.logicalH==1280 && f.rotationDegrees==270, "Flip: 720x1280 canvas, 270 deg");
  chk(f.game.w==720 && f.game.h==540, "Flip 4:3: game is 720x540 (540p native)");
  chk(f.companion.w==720 && f.companion.h==740, "Flip 4:3: companion is 720x740");
  chk(f.game.y + f.game.h == f.companion.y, "Flip: stack is contiguous at gap 0");
  chk(f.game.h + f.companion.h == 1280, "Flip: stack exactly fills the canvas");

  // --- 16:9 ---------------------------------------------------------------
  lay::Layout w{};
  lay::compute(W, H, 16.f/9.f, &w);
  std::printf("     16:9 game=%dx%d companion=%dx%d\n", w.game.w,w.game.h,w.companion.w,w.companion.h);
  chk(w.game.h==405 && w.companion.h==875, "Flip 16:9: game 720x405, companion 720x875");

  // --- THE corner check: the trap that broke every tap in the ALEKS port ---
  // Logical (0,0) must present at physical (0, outH).
  lay::Rect origin{0,0,1,1};
  lay::Rect p = lay::to_physical(origin, W, H, 270);
  std::printf("     logical(0,0) -> physical(%d,%d)\n", p.x, p.y + p.h);
  chk(p.x==0 && (p.y + p.h)==H, "rot270: logical (0,0) presents at physical (0,outH)");

  // Width and height must swap.
  lay::Rect r{0, 100, 720, 540};
  lay::Rect pr = lay::to_physical(r, W, H, 270);
  std::printf("     logical(0,100,720x540) -> physical(%d,%d,%dx%d)\n", pr.x,pr.y,pr.w,pr.h);
  chk(pr.w==540 && pr.h==720, "rot270: width/height swap");
  chk(pr.x==100, "rot270: physical x comes from logical y");
  chk(pr.y==H-0-720, "rot270: physical y = outH - lx - lw");

  // The full stack must stay inside the panel after mapping.
  lay::Rect pg = lay::to_physical(f.game, W, H, 270);
  lay::Rect pc = lay::to_physical(f.companion, W, H, 270);
  chk(pg.x>=0 && pg.y>=0 && pg.x+pg.w<=W && pg.y+pg.h<=H, "rot270: game rect inside the panel");
  chk(pc.x>=0 && pc.y>=0 && pc.x+pc.w<=W && pc.y+pc.h<=H, "rot270: companion rect inside the panel");
  chk(pg.x+pg.w<=pc.x || pc.x+pc.w<=pg.x, "rot270: game and companion do not overlap");

  // --- clamping discipline: nothing may overflow --------------------------
  lay::set_params(100, 5000);          // absurd gap
  lay::Layout g{}; lay::compute(W,H,4.f/3.f,&g);
  chk(g.game.y>=0 && g.companion.y+g.companion.h<=1280, "absurd gap is clamped, not overflowed");
  lay::set_params(50, 0);
  lay::Layout h{}; lay::compute(W,H,4.f/3.f,&h);
  std::printf("     pct=50 companion=%dx%d\n", h.companion.w,h.companion.h);
  chk(h.companion.w==360 && h.companion.h==370, "pct 50 halves the companion");

  // --- degenerate input must fall back, not divide by zero ----------------
  lay::set_mode(lay::kFlip270);
  lay::Layout z{}; lay::compute(0,0,0.f,&z);
  chk(z.mode==lay::kNormal, "degenerate size/aspect falls back to Normal");

  std::printf("\n%s (%d failing)\n", fails==0 ? "ALL PASS" : "FAILURES", fails);
  return fails!=0;
}
