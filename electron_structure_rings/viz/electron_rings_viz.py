"""
electron_rings_viz.py — визуализация модели электрона как пары прецессирующих колец.
Папка проекта: ~/articles/electron_structure_rings/viz/

Геометрия (основа — draw_scheme из Structure_of_electron.ipynb):
  * два соосных кольца радиуса R, плоскости z=+H (N, s=+1) и z=-H (S, s=-1);
  * особые точки сдвинуты по фазе на 180°: θ_N = φ, θ_S = φ + π;
  * прецессия φ = ω t, обе пары вращаются в ОДНУ сторону.
Локальный базис кольца: e_r (радиально), e_t (касательная = линия дефекта), e_z (нормаль).

МЁБИУС-кольцо (непрерывная прецессия, 1 ноя 2026):
  χ — азимут от УСЛОВНОГО разреза (неподвижен; N: θ=0, S: θ=π);
      ψ = −χ/2 + φ/2,     b = s·b0·(cos ψ · u + sin ψ · v),
  • каждый вектор b вращается в своей плоскости с частотой ω/2 и своей фазой −χ/2;
  • особая (бегущая) точка — где b ∥ s·u (ψ = 0): движется по кольцу с частотой ω,
    вдвое быстрее вращения b; N и S в противофазе;
  • φ → φ+2π: b → −b всюду (знак ∝ cos(φ/2), спин ½), φ → φ+4π: исходное.
  • скачок b на разрезе (−b ↔ b) — свойство неориентируемого контура; положение разреза
    условно, если b определён с точностью до знака (двулистное накрытие SO(3) → спинор).
  Плоскости (u, v): 'rz' (e_r,e_z), ось e_t — вокруг линии; 'rt' (e_r,e_t), ось e_z;
  'tz' (e_t,e_z), ось e_r.

GLIDE-пара:
  'glide_T3'  — строгая T(3): b в плоскости петли, постоянен и неподвижен — вращаться нечему.
  'glide_Cos' — среда Коссера: b переносится вращающимся репером, b = s·b0·(cos φ, sin φ, 0);
                b1 + b2 = 0 в любой момент; 4 характерные точки бегут по кольцу; период 2π (бозон).

Это ИЛЛЮСТРАЦИЯ модели, а не решение уравнений поля.
"""
import numpy as np

R, H, B0 = 4.0, 3.0, 1.3
TILT = np.deg2rad(22)
COL = {+1: 'tab:blue', -1: 'tab:red'}
PLANES = {'rz': ('r', 'z'), 'rt': ('r', 't'), 'tz': ('t', 'z')}
PLANE_TITLE = {'rz': 'плоскость (e_r, e_z): поворот вокруг линии',
               'rt': 'плоскость (e_r, e_t): в плоскости кольца',
               'tz': 'плоскость (e_t, e_z): винт ↔ призм.'}
# стиль характерной точки по направлению b: e_t — винт, e_r — glide-край, e_z — призматический край
MARK = {'t': ('s', 'gold', 'square'), 'r': ('D', 'white', 'diamond-open'), 'z': ('^', 'orange', 'diamond')}


def pole_angle(s, phi):
    return phi if s > 0 else phi + np.pi


def frame(theta):
    th = np.asarray(theta, float)
    return {'r': np.stack([np.cos(th), np.sin(th), 0*th], -1),
            't': np.stack([-np.sin(th), np.cos(th), 0*th], -1),
            'z': np.stack([0*th, 0*th, 1+0*th], -1)}


CUT = {+1: 0.0, -1: np.pi}     # условный разрез (Вольтерра) мёбиус-кольца, неподвижен


def mobius_b(chi, s, phi, plane='rz'):
    """Непрерывная прецессия: chi — азимут от РАЗРЕЗА (0 ≤ chi < 2π).
    ψ = −chi/2 + φ/2: в каждой точке b вращается с частотой ω/2 со своей фазой −chi/2;
    точки фиксированного типа (b ∥ u, b ∥ v) бегут по кольцу с частотой ω;
    φ → φ+2π: b → −b всюду (фермион), φ → φ+4π: исходное состояние."""
    chi = np.asarray(chi, float)
    th = CUT[s] + chi
    f = frame(th); u, v = PLANES[plane]
    psi = -chi/2 + phi/2
    return th, s*B0*(np.cos(psi)[..., None]*f[u] + np.sin(psi)[..., None]*f[v])


def moving_point(s, phi):
    """Особая (бегущая) точка: где b ∥ s·u, т.е. ψ = 0 (mod 2π) — движется с частотой ω."""
    return CUT[s] + np.mod(phi, 2*np.pi)


def glide_b(theta, s, phi, cosserat):
    th = np.asarray(theta, float)
    a = phi if cosserat else 0.0
    return np.broadcast_to(s*B0*np.array([np.cos(a), np.sin(a), 0.0]), th.shape + (3,)).copy()


def kind_parse(kind):
    if kind.startswith('mobius'):
        return 'mobius', kind.split('_')[1]
    return kind, None


def arrows(kind, s, phi, n):
    k, plane = kind_parse(kind)
    chi = 1e-3 + np.linspace(0, 2*np.pi, n, endpoint=False)
    if k == 'mobius':
        return mobius_b(chi, s, phi, plane)   # стрелки привязаны к неподвижным точкам
    return chi, glide_b(chi, s, phi, k == 'glide_Cos')


def char_points(kind, s, phi):
    """dict: направление ('t','r','z') -> список θ, где b параллелен этому базисному вектору."""
    k, plane = kind_parse(kind)
    if k != 'mobius':
        a = phi if k == 'glide_Cos' else 0.0
        return {'t': [a + np.pi/2, a - np.pi/2], 'r': [a, a + np.pi]}
    u, v = PLANES[plane]
    out = {u: [], v: []}
    for m in range(-6, 8):
        for key, target in [(u, m*np.pi), (v, np.pi/2 + m*np.pi)]:
            chi = phi - 2*target          # из ψ = −chi/2 + φ/2 = target
            if 1e-3 < chi < 2*np.pi - 1e-3:
                out[key].append(CUT[s] + chi)
    return out


def project(P):
    P = np.asarray(P, float)
    x = P[..., 0]
    y = P[..., 2]*np.cos(TILT) + P[..., 1]*np.sin(TILT)
    depth = P[..., 1]*np.cos(TILT) - P[..., 2]*np.sin(TILT)
    return x, y, depth


def ring_xyz(theta, z0, x0=0.0):
    th = np.asarray(theta, float)
    return np.stack([x0 + R*np.cos(th), R*np.sin(th), np.full_like(th, z0)], -1)


def draw_pair(ax, kind='mobius_rz', phi=0.0, n_arrows=16, x0=0.0, title=None):
    k, plane = kind_parse(kind)
    for s, z0 in [(+1, H), (-1, -H)]:
        col = COL[s]
        t = np.linspace(0, 2*np.pi, 400)
        x, y, d = project(ring_xyz(t, z0, x0))
        ax.plot(np.where(d > 0, x, np.nan), np.where(d > 0, y, np.nan), '--', c=col, lw=1, alpha=.5)
        ax.plot(np.where(d <= 0, x, np.nan), np.where(d <= 0, y, np.nan), '-', c=col, lw=2)
        th, b = arrows(kind, s, phi, n_arrows)
        base = ring_xyz(th, z0, x0)
        bx, by, _ = project(base); ex, ey, _ = project(base + b)
        ax.quiver(bx, by, ex-bx, ey-by, angles='xy', scale_units='xy', scale=1,
                  color='darkgreen', width=.0045, headwidth=4, zorder=3)
        for key, ths in char_points(kind, s, phi).items():
            for th_c in ths:
                px, py, _ = project(ring_xyz(np.array([th_c]), z0, x0))
                ax.plot(px, py, MARK[key][0], ms=7, mfc=MARK[key][1], mec='k', zorder=4)
        if k == 'mobius':
            ps = ring_xyz(np.array([moving_point(s, phi)]), z0, x0)[0]
            sx, sy, _ = project(ps)
            ax.plot(sx, sy, 'o', ms=12, mfc=col, mec='k', zorder=5)
            pc = ring_xyz(np.array([CUT[s]]), z0, x0)[0]
            cx, cy, _ = project(pc)
            ax.plot(cx, cy, 'x', ms=11, mew=2.5, c='k', zorder=6)
            ax.text(cx, cy - .55, 'разрез', fontsize=7, ha='center')
        else:
            ps = ring_xyz(np.array([pole_angle(s, phi)]), z0, x0)[0]
            sx, sy, _ = project(ps)
            ax.plot(sx, sy, 'o', ms=12, mfc=col, mec='k', alpha=.3, zorder=5)
            ax.text(sx, sy + .35, '?', fontsize=12, ha='center', weight='bold')
        lx, ly, _ = project(np.array([x0 + R + .3, 0, z0]))
        ax.text(lx, ly, 'N' if s > 0 else 'S', color=col, fontsize=13, weight='bold')
    if k == 'mobius':
        pN = project(ring_xyz(np.array([moving_point(+1, phi)]), H, x0)[0])
        pS = project(ring_xyz(np.array([moving_point(-1, phi)]), -H, x0)[0])
    else:
        pN = project(ring_xyz(np.array([phi]), H, x0)[0])
        pS = project(ring_xyz(np.array([phi + np.pi]), -H, x0)[0])
    ax.plot([pN[0], pS[0]], [pN[1], pS[1]], ':', c='gray', lw=1.5)
    a0, a1 = project(np.array([x0, 0, -H-1.6])), project(np.array([x0, 0, H+1.6]))
    ax.annotate('', (a1[0], a1[1]), (a0[0], a0[1]), arrowprops=dict(arrowstyle='-|>', color='purple', lw=2))
    ax.set_xlim(x0-6.8, x0+7.2); ax.set_ylim(-6.5, 7.0)
    ax.set_aspect('equal'); ax.axis('off')
    if title:
        ax.set_title(title, fontsize=9.5)


def _legend(fig):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker='o', ls='', mfc='tab:blue', mec='k', ms=10, label='особая точка N'),
         Line2D([], [], marker='o', ls='', mfc='tab:red', mec='k', ms=10, label='особая точка S (180°)'),
         Line2D([], [], c='darkgreen', label='вектор Бюргерса b'),
         Line2D([], [], marker='s', ls='', mfc='gold', mec='k', label='b ∥ e_t: винтовая'),
         Line2D([], [], marker='D', ls='', mfc='white', mec='k', label='b ∥ e_r: glide-край'),
         Line2D([], [], marker='^', ls='', mfc='orange', mec='k', label='b ∥ e_z: призм. край'),
         Line2D([], [], c='purple', label='ось спина')]
    fig.legend(handles=h, loc='lower center', ncol=4, fontsize=9, frameon=False)


def flat_mobius_three_planes(path, phi0=np.pi/5):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 3, figsize=(16, 16.5))
    for j, pl in enumerate(['rz', 'rt', 'tz']):
        for i, (dphi, lab) in enumerate([(0, 'φ₀'), (2*np.pi, 'φ₀+360°: b → −b'), (4*np.pi, 'φ₀+720°: исходное')]):
            draw_pair(axs[i, j], 'mobius_' + pl, phi0 + dphi,
                      title=f'{PLANE_TITLE[pl]}\n{lab}')
    fig.suptitle('Мёбиус-кольца: три возможные плоскости поворота b (ψ = −χ/2 + φ/2): b в каждой точке вращается с ω/2, '
                 'особые точки бегут с ω в противофазе', fontsize=13)
    _legend(fig)
    fig.savefig(path, dpi=100, bbox_inches='tight'); plt.close(fig)


def flat_glide(path, phi0=np.pi/5):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(17, 7))
    draw_pair(axs[0], 'glide_T3', phi0, title='Glide-пара, строгая T(3):\nb постоянен и неподвижен — вращаться нечему')
    draw_pair(axs[1], 'glide_Cos', phi0, title=f'Glide-пара в среде Коссера, φ = {np.degrees(phi0):.0f}°:\n'
                                               'b переносится вращающимся репером, b₁+b₂ = 0')
    draw_pair(axs[2], 'glide_Cos', phi0 + 2*np.pi, title='то же, φ + 360°: исходное состояние\n(период 2π — бозон)')
    _legend(fig)
    fig.savefig(path, dpi=100, bbox_inches='tight'); plt.close(fig)


ANIM_KINDS = [('glide_Cos', 'glide (Коссер)'), ('mobius_rz', 'мёбиус (e_r,e_z)'),
              ('mobius_rt', 'мёбиус (e_r,e_t)'), ('mobius_tz', 'мёбиус (e_t,e_z)')]


def animation(path, frames=72, fps=14):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    fig, axs = plt.subplots(2, 2, figsize=(12, 12.5))
    _legend(fig)

    def upd(i):
        phi = 4*np.pi*i/frames
        turn = int(phi // (2*np.pi)) + 1
        for ax, (k, ttl) in zip(axs.flat, ANIM_KINDS):
            ax.clear()
            draw_pair(ax, k, phi, n_arrows=14, title=f'{ttl}\nφ = {np.degrees(phi):.0f}°, оборот {turn} из 2')
    FuncAnimation(fig, upd, frames=frames).save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def model3d_html(path, frames=36, n_arrows=16, gap=8.0):
    """3D (WebGL/plotly): 4 пары в углах квадрата с подписями, Play и ползунок φ ∈ [0, 4π)."""
    import plotly.graph_objects as go
    corners = [(-gap, gap), (gap, gap), (-gap, -gap), (gap, -gap)]
    labels = ['glide-пара (Коссер)<br>период 2π, бозон',
              'мёбиус (e_r, e_z)<br>поворот вокруг линии',
              'мёбиус (e_r, e_t)<br>в плоскости кольца',
              'мёбиус (e_t, e_z)<br>винт ↔ призм.']

    def sh(P, o):
        P = np.array(P, float); P[..., 0] += o[0]; P[..., 1] += o[1]; return P

    def traces(phi):
        tr = []
        for (kind, _), o, lab in zip(ANIM_KINDS, corners, labels):
            k, _p = kind_parse(kind)
            for s, z0 in [(+1, H), (-1, -H)]:
                col = 'royalblue' if s > 0 else 'crimson'
                t = np.linspace(0, 2*np.pi, 160); P = sh(ring_xyz(t, z0), o)
                tr.append(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode='lines', line=dict(color=col, width=6)))
                th, b = arrows(kind, s, phi, n_arrows)
                base = sh(ring_xyz(th, z0), o)
                xs, ys, zs = [], [], []
                for q in range(n_arrows):
                    for p in (base[q], base[q] + b[q]):
                        xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
                    xs.append(None); ys.append(None); zs.append(None)
                tr.append(go.Scatter3d(x=xs, y=ys, z=zs, mode='lines', line=dict(color='green', width=5)))
                tip = base + b
                tr.append(go.Scatter3d(x=tip[:, 0], y=tip[:, 1], z=tip[:, 2], mode='markers',
                                       marker=dict(size=3, color='darkgreen')))
                cp = char_points(kind, s, phi)
                for key in ['t', 'r', 'z']:
                    q = sh(ring_xyz(np.array(cp.get(key, []), float), z0), o)
                    tr.append(go.Scatter3d(x=q[:, 0], y=q[:, 1], z=q[:, 2], mode='markers',
                                           marker=dict(size=6, color=MARK[key][1] if key != 'r' else 'black',
                                                       symbol=MARK[key][2])))
                ps = sh(ring_xyz(np.array([moving_point(s, phi) if k == 'mobius' else pole_angle(s, phi)]), z0), o)
                if k == 'mobius':
                    pc = sh(ring_xyz(np.array([CUT[s]]), z0), o)
                    tr.append(go.Scatter3d(x=pc[:, 0], y=pc[:, 1], z=pc[:, 2], mode='markers+text', text=['разрез'],
                                           marker=dict(size=7, color='black', symbol='x')))
                tr.append(go.Scatter3d(x=ps[:, 0], y=ps[:, 1], z=ps[:, 2], mode='markers',
                                       marker=dict(size=10, color=col, opacity=1.0 if k == 'mobius' else 0.3,
                                                   line=dict(color='black', width=2))))
                nl = sh(np.array([R + 0.8, 0, z0]), o)
                tr.append(go.Scatter3d(x=[nl[0]], y=[nl[1]], z=[nl[2]], mode='text', text=['N' if s > 0 else 'S'],
                                       textfont=dict(size=14, color=col)))
            if k == 'mobius':
                pn = sh(ring_xyz(np.array([moving_point(+1, phi)]), H), o); pS = sh(ring_xyz(np.array([moving_point(-1, phi)]), -H), o)
            else:
                pn = sh(ring_xyz(np.array([phi]), H), o); pS = sh(ring_xyz(np.array([phi + np.pi]), -H), o)
            tr.append(go.Scatter3d(x=[pn[0, 0], pS[0, 0]], y=[pn[0, 1], pS[0, 1]], z=[pn[0, 2], pS[0, 2]], mode='lines',
                                   line=dict(color='gray', width=3, dash='dot')))
            tr.append(go.Scatter3d(x=[o[0], o[0]], y=[o[1], o[1]], z=[-H-1.6, H+1.6], mode='lines',
                                   line=dict(color='purple', width=7)))
            tr.append(go.Scatter3d(x=[o[0]], y=[o[1]], z=[H + 3.0], mode='text', text=[f'<b>{lab}</b>'],
                                   textfont=dict(size=13, color='black')))
        return tr

    phis = np.linspace(0, 4*np.pi, frames, endpoint=False)
    fig = go.Figure(data=traces(0.0),
                    frames=[go.Frame(data=traces(p), name=f'{np.degrees(p):.0f}') for p in phis])
    steps = [dict(method='animate', label=f'{np.degrees(p):.0f}°',
                  args=[[f'{np.degrees(p):.0f}'], dict(mode='immediate', frame=dict(duration=0, redraw=True))])
             for p in phis]
    fig.update_layout(
        title='Четыре конфигурации · зелёное: b · ● особые точки N/S (180°) · ■ винт · ◇ glide-край · ◆ призм. край',
        showlegend=False, margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(aspectmode='data', xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
                   camera=dict(eye=dict(x=0.9, y=-1.5, z=0.9))),
        updatemenus=[dict(type='buttons', x=0.05, y=0.05, buttons=[
            dict(label='▶ Play', method='animate',
                 args=[None, dict(frame=dict(duration=120, redraw=True), fromcurrent=True, mode='immediate')]),
            dict(label='❚❚', method='animate',
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])])],
        sliders=[dict(steps=steps, x=0.15, len=0.8, currentvalue=dict(prefix='φ = '))])
    fig.write_html(path, include_plotlyjs=True, auto_play=False)


def total_burgers(kind, phi, n=2000):
    """Контур, охватывающий ОБА кольца (сцеплен с обеими линиями): b_tot = b_N + b_S.
    Для мёбиуса b зависит от точки пересечения контура с линией — берём ту же азимутальную точку."""
    k, plane = kind_parse(kind)
    th = np.array([0.7])
    if k == 'mobius':
        bN = mobius_b(np.mod(th - CUT[+1], 2*np.pi), +1, phi, plane)[1][0]
        bS = mobius_b(np.mod(th - CUT[-1], 2*np.pi), -1, phi, plane)[1][0]
    else:
        bN = glide_b(th, +1, phi, k == 'glide_Cos')[0]; bS = glide_b(th, -1, phi, k == 'glide_Cos')[0]
    return bN + bS


def self_check():
    chi = np.linspace(0.01, 2*np.pi - 0.01, 50)
    res = {}
    for pl in PLANES:
        ok2 = ok4 = okj = True
        for s in (+1, -1):
            _, b0 = mobius_b(chi, s, 0.3, pl)
            _, b2 = mobius_b(chi, s, 0.3 + 2*np.pi, pl)
            _, b4 = mobius_b(chi, s, 0.3 + 4*np.pi, pl)
            ok2 &= np.allclose(b2, -b0); ok4 &= np.allclose(b4, b0)
            _, e = mobius_b(np.array([2*np.pi - 1e-7]), s, 0.3, pl)
            _, g = mobius_b(np.array([1e-7]), s, 0.3, pl)
            okj &= abs(np.linalg.norm(e - g) - 2*B0) < 1e-5
        res[f'мёбиус {pl}: φ+2π → −b'] = ok2
        res[f'мёбиус {pl}: φ+4π → b'] = ok4
        res[f'мёбиус {pl}: |Δb| = 2b0'] = okj
    th, b = mobius_b(chi, 1, 0.3, 'rz')
    res['мёбиус rz: b ⊥ e_t (вокруг линии)'] = np.allclose(np.sum(b*frame(th)['t'], -1), 0)
    # непрерывная прецессия в фиксированной точке: угол ψ растёт с ω/2, без скачков
    ph = np.linspace(0, 4*np.pi, 801)
    _, bb = mobius_b(np.full_like(ph, 1.0), 1, ph, 'rz')
    f1 = frame(np.array([CUT[1] + 1.0]))
    ang = np.unwrap(np.arctan2(bb @ f1['z'][0], bb @ f1['r'][0]))
    res['мёбиус: b в фикс. точке вращается с ω/2 непрерывно'] = np.allclose(np.diff(ang), ph[1]/2, atol=1e-9)
    mp = [moving_point(1, p) for p in np.linspace(0, 1.9*np.pi, 20)]
    res['мёбиус: особая точка бежит с частотой ω'] = np.allclose(np.diff(mp), 0.1*np.pi)
    res['мёбиус: особые точки N/S в противофазе'] = np.isclose(moving_point(-1, .3) - moving_point(1, .3), np.pi)
    res['glide Коссер: b₁+b₂ = 0 при всех φ'] = all(np.allclose(total_burgers('glide_Cos', p), 0)
                                                    for p in np.linspace(0, 2*np.pi, 13))
    _, g0 = arrows('glide_Cos', 1, 0.3, 8); _, g2 = arrows('glide_Cos', 1, 0.3 + 2*np.pi, 8)
    res['glide Коссер: период 2π'] = np.allclose(g0, g2)
    res['полюса в противофазе'] = np.isclose(pole_angle(-1, 0.3) - pole_angle(1, 0.3), np.pi)
    for key, v in res.items():
        print(('OK   ' if v else 'FAIL ') + key)
    return all(res.values())


if __name__ == '__main__':
    import os
    out = os.path.dirname(os.path.abspath(__file__))
    self_check()
    flat_mobius_three_planes(os.path.join(out, 'flat_mobius_3planes.png'))
    flat_glide(os.path.join(out, 'flat_glide.png'))
    animation(os.path.join(out, 'precession.gif'))
    model3d_html(os.path.join(out, 'model3d.html'))
