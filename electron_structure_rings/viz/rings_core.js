// rings_core.js — математика всех динамик (общая для веб-модели и тестов)
// Кольцо радиуса R в плоскости z = s·H; s=+1 — N, s=−1 — S. Базис кольца e_r, e_t (линия), e_z.
// φ = ω t — фаза прецессии; особые точки N/S в противофазе.
var R = 4.0, H = 3.0, B0 = 1.3, TWO_PI = 2 * Math.PI;
var PLANES = { rz: ['r', 'z'], rt: ['r', 't'], tz: ['t', 'z'] };
var PLANE_NAME = { rz: '(e_r,e_z): вокруг линии', rt: '(e_r,e_t): в плоскости кольца', tz: '(e_t,e_z): винт↔призм.' };

function mod2pi(x) { return ((x % TWO_PI) + TWO_PI) % TWO_PI; }
function frame(th) {
  return { r: [Math.cos(th), Math.sin(th), 0], t: [-Math.sin(th), Math.cos(th), 0], z: [0, 0, 1] };
}
function lin(a, ka, b, kb) { return [a[0] * ka + b[0] * kb, a[1] * ka + b[1] * kb, a[2] * ka + b[2] * kb]; }
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function poleAngle(s, phi) { return s > 0 ? phi : phi + Math.PI; }
function cutAngle(s) { return s > 0 ? 0 : Math.PI; }

// Описание вариантов
var KINDS = {
  gT3: { mobius: false, title: 'glide, строгая T(3)', sub: 'b постоянен и неподвижен', period: '—', stat: 'нет прецессии' },
  gCos: { mobius: false, title: 'glide, среда Коссера', sub: 'b переносится репером, ω', period: '2π', stat: 'бозон' },
  mA: { mobius: true, title: 'мёбиус A', sub: 'b стоит, бежит шов Δb; перемычка N–S', period: '4π', stat: 'фермион' },
  mB: { mobius: true, title: 'мёбиус B', sub: 'b вращается с ω/2, разрез неподвижен (±b)', period: '4π', stat: 'фермион' },
  mC: { mobius: true, title: 'мёбиус C', sub: 'b с ω/2 + бегущий шов', period: '2π', stat: 'бозон (контрпример)' }
};

// Вектор Бюргерса в точке кольца с азимутом th
function burgers(kind, s, th, phi, plane) {
  if (kind === 'gT3') return [s * B0, 0, 0];
  if (kind === 'gCos') return [s * B0 * Math.cos(phi), s * B0 * Math.sin(phi), 0];
  var f = frame(th), u = f[PLANES[plane][0]], v = f[PLANES[plane][1]], psi;
  if (kind === 'mA') psi = mod2pi(th - poleAngle(s, phi)) / 2 + phi / 2;        // χ от бегущего шва
  else if (kind === 'mB') psi = -mod2pi(th - cutAngle(s)) / 2 + phi / 2;        // χ от неподвижного разреза
  else psi = mod2pi(th - poleAngle(s, phi)) / 2 + phi;                           // mC
  return lin(u, s * B0 * Math.cos(psi), v, s * B0 * Math.sin(psi));
}

// Положение особой точки (то, что «бежит» по кольцу)
function specialPoint(kind, s, phi) {
  if (kind === 'mB') return cutAngle(s) + mod2pi(phi);   // там, где b ∥ ±u
  return poleAngle(s, phi);
}
// Скачок b на шве/разрезе (для A и C — на бегущем шве, для B — на разрезе)
function seamAngle(kind, s, phi) { return kind === 'mB' ? cutAngle(s) : poleAngle(s, phi); }
function seamJump(kind, s, phi, plane) {
  var a = seamAngle(kind, s, phi), e = 1e-7;
  return lin(burgers(kind, s, a - e, phi, plane), 1, burgers(kind, s, a + e, phi, plane), -1);
}

// Характерные точки: где b ∥ e_t (винт), e_r (glide-край), e_z (призм. край)
function charPoints(kind, s, phi, plane) {
  var n = 720, out = { t: [], r: [], z: [] }, seam = KINDS[kind].mobius ? seamAngle(kind, s, phi) : null;
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
  burgers: burgers, specialPoint: specialPoint, seamJump: seamJump, charPoints: charPoints,
  KINDS: KINDS, PLANES: PLANES, B0: B0, frame: frame, dot: dot, lin: lin, poleAngle: poleAngle, cutAngle: cutAngle
};
