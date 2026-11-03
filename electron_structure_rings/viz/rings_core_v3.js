// rings_core_v3.js — математика всех динамик (v3, 3 ноя 2026)
// Кольцо радиуса R в плоскости z = s·H; s=+1 — N, s=−1 — S. Базис кольца e_r, e_τ (линия), e_z.
// φ = ω t — фаза прецессии. POLE_OFF — сдвиг фазы особых точек S относительно N (π — противофаза, 0 — в фазе).
var R = 4.0, H = 3.0, B0 = 1.3, TWO_PI = 2 * Math.PI, POLE_OFF = Math.PI;
var PLANES = { rz: ['r', 'z'], rt: ['r', 't'], tz: ['t', 'z'] };

function mod2pi(x) { return ((x % TWO_PI) + TWO_PI) % TWO_PI; }
function frame(th) {
  return { r: [Math.cos(th), Math.sin(th), 0], t: [-Math.sin(th), Math.cos(th), 0], z: [0, 0, 1] };
}
function lin(a, ka, b, kb) { return [a[0] * ka + b[0] * kb, a[1] * ka + b[1] * kb, a[2] * ka + b[2] * kb]; }
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function norm(a) { return Math.sqrt(dot(a, a)); }
function poleAngle(s, phi) { return s > 0 ? phi : phi + POLE_OFF; }
function cutAngle(s) { return s > 0 ? 0 : POLE_OFF; }

// director: true — b определён с точностью до знака (рисуется без остриёв, разрез ненаблюдаем)
// status: 'ok' — согласуется с теорией; 'cond' — при допущении; 'bad' — противоречие в текущем виде
var KINDS = {
  gT3:  { mobius: false, director: false, title: 'glide, строгая T(3)', sub: 'b постоянен и неподвижен', period: '—', stat: 'нет прецессии', status: 'ok' },
  gCos: { mobius: false, director: false, title: 'glide, среда Коссера', sub: 'b переносится репером с ω', period: '2π', stat: 'бозон', status: 'cond' },
  mA:   { mobius: true,  director: false, title: 'Мёбиус A', sub: 'b стоит, бежит шов (нужна дисклинация в шве)', period: '4π', stat: 'фермион', status: 'bad' },
  mB:   { mobius: true,  director: true,  title: 'Мёбиус B', sub: 'директор ±b вращается с ω/2', period: '4π', stat: 'фермион', status: 'cond' },
  mC:   { mobius: true,  director: false, title: 'Мёбиус C', sub: 'b с ω/2 + бегущий шов', period: '2π', stat: 'бозон (контрпример)', status: 'ok' }
};

function burgers(kind, s, th, phi, plane) {
  if (kind === 'gT3') return [s * B0, 0, 0];
  if (kind === 'gCos') return [s * B0 * Math.cos(phi), s * B0 * Math.sin(phi), 0];
  var f = frame(th), u = f[PLANES[plane][0]], v = f[PLANES[plane][1]], psi;
  if (kind === 'mA') psi = mod2pi(th - poleAngle(s, phi)) / 2 + phi / 2;
  else if (kind === 'mB') psi = -mod2pi(th - cutAngle(s)) / 2 + phi / 2;
  else psi = mod2pi(th - poleAngle(s, phi)) / 2 + phi;
  return lin(u, s * B0 * Math.cos(psi), v, s * B0 * Math.sin(psi));
}

// Бегущая особая точка (null — у динамики её нет)
function specialPoint(kind, s, phi) {
  if (kind === 'gT3' || kind === 'gCos') return null;
  if (kind === 'mB') return cutAngle(s) + mod2pi(phi);
  return poleAngle(s, phi);
}
// Шов со скачком b (только для векторных A и C; у директора B скачка нет)
function seamAngle(kind, s, phi) { return (kind === 'mA' || kind === 'mC') ? poleAngle(s, phi) : null; }
function seamJump(kind, s, phi, plane) {
  var a = seamAngle(kind, s, phi), e = 1e-7;
  if (a === null) return [0, 0, 0];
  return lin(burgers(kind, s, a - e, phi, plane), 1, burgers(kind, s, a + e, phi, plane), -1);
}

// Суммарный b по контуру, охватывающему оба кольца, в азимутальной точке th (для директора — min по знаку)
function pairSum(kind, th, phi, plane) {
  var bN = burgers(kind, 1, th, phi, plane), bS = burgers(kind, -1, th, phi, plane);
  var a = norm(lin(bN, 1, bS, 1));
  if (KINDS[kind].director) a = Math.min(a, norm(lin(bN, 1, bS, -1)));
  return a;
}
function pairSumMax(kind, phi, plane) {
  var m = 0;
  for (var i = 0; i < 72; i++) m = Math.max(m, pairSum(kind, 0.013 + TWO_PI * i / 72, phi, plane));
  return m;
}

function charPoints(kind, s, phi, plane) {
  var n = 720, out = { t: [], r: [], z: [] }, seam = seamAngle(kind, s, phi);
  ['t', 'r', 'z'].forEach(function (key) {
    var c = [];
    for (var i = 0; i < n; i++) {
      var th = TWO_PI * i / n, b = burgers(kind, s, th, phi, plane);
      c.push(Math.abs(dot(b, frame(th)[key])) / B0);
    }
    for (var i = 0; i < n; i++) {
      var p = c[(i - 1 + n) % n], q = c[(i + 1) % n];
      if (c[i] > 0.985 && c[i] >= p && c[i] > q) {
        var th = TWO_PI * i / n;
        if (seam !== null && Math.abs(mod2pi(th - seam + Math.PI) - Math.PI) < 0.03) continue;
        out[key].push(th);
      }
    }
  });
  return out;
}

if (typeof module !== 'undefined') module.exports = {
  burgers: burgers, specialPoint: specialPoint, seamJump: seamJump, charPoints: charPoints, pairSumMax: pairSumMax,
  KINDS: KINDS, PLANES: PLANES, B0: B0, frame: frame, dot: dot, lin: lin, norm: norm,
  setPoleOff: function (x) { POLE_OFF = x; }
};
