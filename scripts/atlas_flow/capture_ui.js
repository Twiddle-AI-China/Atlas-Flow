/**
 * Capture a screenshot of the Atlas Flow UI, for the README and the paper.
 *
 * The screenshots in docs/assets/ were made with this. It drives a real window
 * against a running local server -- theme, a drawn path, a looping progression,
 * and enough dwell for the scopes to fill with real audio -- then writes a PNG
 * at twice the logical size, so the result is retina-sharp.
 *
 * Audio is muted three ways: the Chromium `mute-audio` switch, the webContents
 * flag before the first load, and again before the shutter. This is not
 * belt-and-braces for its own sake -- an earlier unmuted capture run played the
 * live voice through the speakers, in a meeting. Do not remove any of them, and
 * do not add an autoplay-policy exemption to "fix" a silent capture.
 *
 * Usage:
 *   PYTHONPATH=src python -m midibrave.atlas_flow_local --port 18899 &
 *   PORT=18899 THEME=mono OUT=docs/assets/atlas-flow-ui-monochrome.png \
 *     ./electron/node_modules/.bin/electron scripts/atlas_flow/capture_ui.js
 *
 * THEME is one of mono | ember | reef | dusk | paper, matching the settings
 * panel. Downscale the result to the 1600 px the README uses with:
 *   sips -Z 1600 -s format png <out> --out <out>
 */
const {app, BrowserWindow} = require('electron');
const {writeFileSync} = require('node:fs');

app.commandLine.appendSwitch('mute-audio');

const PORT = process.env.PORT || '18796';
const THEME = process.env.THEME || 'mono';
const OUT = process.env.OUT || 'atlas-flow-ui.png';
const W = Number(process.env.WIDTH || 1469);
const H = Number(process.env.HEIGHT || 869);
// The scopes draw against wall-clock time, so they need real seconds of audio
// before they show anything worth photographing. Shorten it and you get a
// mostly-empty spectrogram that looks like a bug.
const DWELL_MS = Number(process.env.DWELL_MS || 16000);

const sleep = ms => new Promise(r => setTimeout(r, ms));

app.whenReady().then(async () => {
  const win = new BrowserWindow({
    width: W, height: H, show: true, frame: false,
    webPreferences: {backgroundThrottling: false},
  });
  win.webContents.setAudioMuted(true);

  const url = `http://127.0.0.1:${PORT}/`;
  await win.loadURL(url);
  // The theme is read from localStorage once, at module scope, so it has to be
  // seeded and the page reloaded rather than set through the settings panel.
  await win.webContents.executeJavaScript(`
    localStorage.setItem('atlas.theme', ${JSON.stringify(THEME)});
    localStorage.setItem('atlas.lang', 'en');
    true;
  `);
  await win.loadURL(url);
  win.webContents.setAudioMuted(true);
  await sleep(3500);

  await win.webContents.executeJavaScript(`
    (async () => {
      const sleep = ms => new Promise(r => setTimeout(r, ms));
      const map = document.querySelector('#map');
      const box = map.getBoundingClientRect();
      const at = (u, v) => ({
        clientX: box.left + u * box.width,
        clientY: box.top + v * box.height,
      });
      const fire = (type, p) => map.dispatchEvent(new PointerEvent(type, {
        ...p, pointerId: 1, pointerType: 'mouse',
        bubbles: true, cancelable: true, isPrimary: true,
      }));

      document.querySelector('#draw').click();
      await sleep(120);

      // A sweeping arc through the populated middle of the atlas. The map
      // thins the stroke itself, so emitting densely here is harmless.
      const pts = [];
      for (let i = 0; i <= 64; i++) {
        const t = i / 64;
        pts.push(at(0.16 + 0.62 * t,
                    0.74 - 0.52 * Math.sin(Math.PI * t * 0.92) - 0.06 * t));
      }
      fire('pointerdown', pts[0]);
      for (const p of pts.slice(1)) { fire('pointermove', p); await sleep(16); }
      fire('pointerup', pts.at(-1));
      await sleep(400);

      const prog = document.querySelector('#progression');
      if (prog && prog.options.length) {
        prog.selectedIndex = Math.min(3, prog.options.length - 1);
        prog.dispatchEvent(new Event('change', {bubbles: true}));
      }
      await sleep(300);
      document.querySelector('#loop').click();
    })();
  `);

  await sleep(DWELL_MS);
  win.webContents.setAudioMuted(true);

  const image = await win.webContents.capturePage();
  const png = image.resize({width: W * 2, height: H * 2}).toPNG();
  writeFileSync(OUT, png);
  console.log(`wrote ${OUT} (${png.length} bytes, ${W * 2}x${H * 2})`);
  app.exit(0);
}).catch(error => {
  console.error('capture failed:', error);
  app.exit(1);
});
