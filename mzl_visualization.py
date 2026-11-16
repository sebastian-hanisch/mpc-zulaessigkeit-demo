"""Plotly-Abbildungen (alle Achsen gesperrt, damit Touch-Scrollen nicht am Diagramm hängt; DEMO-PLAYBOOK §3).

Ebenen-Bilder (Zustandsraum des Wagens) zeichnen die exakten Polygone aus `mzl_poly` als Gleitkomma-Linienzüge; gleicher Maßstab über `scaleanchor` mit
automatischem Bereich und zwei unsichtbaren Eckpunkten des Kastens (DEMO-PLAYBOOK §3)."""

import numpy as np
import plotly.graph_objects as go

import mzl_constants as C
import mzl_evaluation as ev
import mzl_poly as P

FONT = dict(size=12)


def _fixed(fig, height=380, legend=True, x_title=None, y_title=None):
    fig.update_xaxes(fixedrange=True, title=x_title, zeroline=False)
    fig.update_yaxes(fixedrange=True, title=y_title, zeroline=False)
    fig.update_layout(height=height, margin=dict(l=50, r=15, t=30, b=45 if not legend else 85), font=FONT,
                      legend=dict(orientation="h", yanchor="top", y=-0.18, x=0) if legend else None, showlegend=legend, dragmode=False)
    return fig


def _poly_xy(cons):
    """Linienzug der Ecken (geschlossen). Entartete Mengen (Punkt, Strecke) geben ihre ein oder zwei Ecken."""
    vs = P.float_vertices(cons)
    if len(vs) >= 3:
        vs = vs + [vs[0]]
    return [v[0] for v in vs], [v[1] for v in vs]


def poly_trace(cons, name, fill, line, width=2, legend=True, dash=None):
    x, y = _poly_xy(cons)
    return go.Scatter(x=x, y=y, mode="lines", name=name, fill="toself" if len(x) > 3 else None, fillcolor=fill, line=dict(color=line, width=width, dash=dash),
                      hoverinfo="name", showlegend=legend)


def _plane(fig, s1, s2, height=470):
    """Gleicher Maßstab: scaleanchor mit automatischem Bereich; der Kasten wird über zwei unsichtbare Eckpunkte erzwungen."""
    fig.add_trace(go.Scatter(x=[-s1 * 1.04, s1 * 1.04], y=[-s2 * 1.04, s2 * 1.04], mode="markers", marker=dict(opacity=0), hoverinfo="skip", showlegend=False))
    _fixed(fig, height, True, "Position x₁", "Geschwindigkeit x₂")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


# --- Skalar --------------------------------------------------------------------------------------------------------------------------------------

def build_scalar_line(a) -> go.Figure:
    """Die Gebiete F_N (bzw. K_N) als Intervalle auf der Zahlengeraden, eine Zeile je N; C als grünes Band, x₀ als Linie."""
    nm = "K" if a.terminal else "F"
    fig = go.Figure()
    n_rows = C.SK_N_MAX
    c = float(a.c)
    fig.add_vrect(x0=-c, x1=c, fillcolor="rgba(47,107,101,0.22)", line_width=0, layer="below")
    for n in range(n_rows + 1):
        f = float(a.f_list[n])
        strong = n == a.N
        fig.add_trace(go.Scatter(x=[-f, f, f, -f, -f], y=[n - 0.32, n - 0.32, n + 0.32, n + 0.32, n - 0.32], mode="lines", fill="toself",
                                 fillcolor="rgba(76,120,168,0.55)" if strong else "rgba(76,120,168,0.22)", line=dict(color=C.COLOR_FLINE, width=2 if strong else 1),
                                 name=f"{nm}_{n}", hovertemplate=f"{nm}_{n} = [−{ev.fr(a.f_list[n])}; {ev.fr(a.f_list[n])}]<extra></extra>", showlegend=False))
    x0 = float(a.x0)
    fig.add_vline(x=x0, line=dict(color=C.COLOR_FALL if a.klass == "aussen" else "#222222", width=2, dash="dash"))
    fig.add_annotation(x=x0, y=-0.9, text=f"x₀ = {ev.fr(a.x0)}", showarrow=False, bgcolor="rgba(255,255,255,0.85)", font=dict(color="#222222"))
    fig.add_annotation(x=0, y=n_rows + 0.95, text=f"C = [−{ev.fr(a.c)}; {ev.fr(a.c)}]", showarrow=False, font=dict(color=C.COLOR_C))
    _fixed(fig, 360, False, "Zustand x", "Horizont N")
    fig.update_xaxes(range=[-C.SK_S * 1.1, C.SK_S * 1.1])
    fig.update_yaxes(range=[n_rows + 1.4, -1.4], tickmode="array", tickvals=list(range(n_rows + 1)), ticktext=[f"N = {n}" for n in range(n_rows + 1)])
    return fig


def build_scalar_traj(a) -> go.Figure:
    fig = go.Figure()
    k_max = 12
    for sign in (1, -1):
        fig.add_hline(y=sign * C.SK_S, line=dict(color=C.COLOR_EDGE, width=1, dash="dot"))
        fig.add_hline(y=sign * float(a.c), line=dict(color=C.COLOR_C, width=2, dash="dash"))
        fig.add_hline(y=sign * float(a.f_N), line=dict(color=C.COLOR_FLINE, width=1.5, dash="dashdot"))
    if a.run is not None:
        traj = a.run.traj[:, 0]
        ks = list(range(len(traj)))
        fig.add_trace(go.Scatter(x=ks, y=traj, mode="lines+markers", name="Regler", line=dict(color="#222222"), marker=dict(size=7)))
        if a.run.status == "verloren":
            fig.add_trace(go.Scatter(x=[ks[-1]], y=[traj[-1]], mode="markers", name="unlösbar ab hier", marker=dict(color=C.COLOR_FALL, size=12, symbol="x")))
    else:
        fig.add_trace(go.Scatter(x=[0], y=[float(a.x0)], mode="markers", name="x₀", marker=dict(color=C.COLOR_FALL, size=10, symbol="x")))
    nm = "K" if a.terminal else "F"
    fig.add_annotation(x=k_max, y=float(a.f_N), text=f"±f_N ({nm}_{a.N})", showarrow=False, xanchor="right", yshift=10, font=dict(color=C.COLOR_FLINE))
    fig.add_annotation(x=k_max, y=float(a.c), text="±C", showarrow=False, xanchor="right", yshift=-10, font=dict(color=C.COLOR_C))
    _fixed(fig, 360, True, "Takt k", "Zustand x")
    fig.update_xaxes(range=[-0.3, k_max + 0.3], dtick=1)
    fig.update_yaxes(range=[-C.SK_S * 1.1, C.SK_S * 1.1])
    return fig


# --- Wagen: Gebiet gegen Horizont ---------------------------------------------------------------------------------------------------------------

def build_regions(a) -> go.Figure:
    """Zustandsebene: Kasten, F_N, C; der gewählte Start und die Bahn des Reglers; die vier Zustände aus Beispiel 12.1."""
    reg = a.reg
    fig = go.Figure()
    fig.add_trace(poly_trace(reg.X, "Kasten X = F₀", "rgba(150,150,150,0.10)", C.COLOR_EDGE, width=1, dash="dot"))
    fig.add_trace(poly_trace(reg.F(a.N), f"F_{a.N}: lösbar mit N = {a.N}", C.COLOR_F, C.COLOR_FLINE))
    fig.add_trace(poly_trace(reg.Cset, "C: für immer haltbar", "rgba(47,107,101,0.40)", C.COLOR_C))
    for key, p in C.START_POINTS.items():
        x, y = float(p["x"][0]), float(p["x"][1])
        if abs(x) <= float(a.s1) and abs(y) <= float(a.s2):
            fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers", name=p["label"], marker=dict(size=10 if key == a.start_key else 7, symbol="diamond" if key == a.start_key else "circle-open",
                                                                                              color="#222222" if key == a.start_key else "#555555", line=dict(width=1.5, color="#222222")),
                                     hovertemplate=p["label"] + "<extra></extra>", showlegend=key == a.start_key))
    if a.run is not None:
        t = a.run.traj
        fig.add_trace(go.Scatter(x=t[:, 0], y=t[:, 1], mode="lines+markers", name="Bahn des Reglers", line=dict(color="#222222", width=2), marker=dict(size=6)))
        if a.run.status == "verloren":
            fig.add_trace(go.Scatter(x=[t[-1, 0]], y=[t[-1, 1]], mode="markers", name="unlösbar ab hier", marker=dict(color=C.COLOR_FALL, size=13, symbol="x")))
    return _plane(fig, float(a.s1), float(a.s2))


def build_area_chart(a) -> go.Figure:
    """Fläche von F_N gegen N (exakt), die Fläche von C als Linie, der gewählte Horizont markiert."""
    reg = a.reg
    ns = list(range(len(reg.seq)))
    ar = [float(x) for x in a.areas]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=ar, mode="lines+markers", name="Fläche F_N", line=dict(color=C.COLOR_FLINE), marker=dict(size=8)))
    fig.add_hline(y=ar[-1], line=dict(color=C.COLOR_C, width=2, dash="dash"), annotation_text="Fläche von C" if reg.n_bar is not None else "letztes gerechnetes Glied",
                  annotation_position="bottom right")
    xs = min(a.N, ns[-1])
    fig.add_trace(go.Scatter(x=[xs], y=[ar[xs]], mode="markers", name=f"gewählt: N = {a.N}", marker=dict(size=14, color=C.COLOR_FALL, symbol="circle-open", line=dict(width=3))))
    _fixed(fig, 330, True, "Horizont N", "Fläche")
    fig.update_xaxes(dtick=1)
    return fig


# --- Wagen: wer fällt? -------------------------------------------------------------------------------------------------------------------------

def build_falls(res, reg, s1, s2) -> go.Figure:
    st = np.array(res.status)
    fall = st == "verloren"
    cats = [
        ("außerhalb C, fällt", (~res.in_c) & fall, C.COLOR_FALL, "circle"),
        ("außerhalb C, bleibt in den gerechneten Takten lösbar", (~res.in_c) & ~fall, "#888888", "circle-open"),
        ("in C, Austritt aus C (fällt)", res.in_c & res.exits_c & fall, C.COLOR_EXIT, "diamond"),
        ("in C, Austritt aus C (Randlauf, nicht als Fall gezählt)", res.in_c & res.exits_c & ~fall, "#b8860b", "diamond-open"),
        ("in C, bleibt in C", res.in_c & ~res.exits_c, C.COLOR_OK, "circle"),
    ]
    fig = go.Figure()
    fig.add_trace(poly_trace(reg.F(res.N), f"F_{res.N}", "rgba(76,120,168,0.10)", C.COLOR_FLINE, width=1))
    fig.add_trace(poly_trace(reg.Cset, "C", "rgba(47,107,101,0.10)", C.COLOR_C, width=2))
    for name, mask, color, symbol in cats:
        if mask.any():
            fig.add_trace(go.Scatter(x=res.starts[mask, 0], y=res.starts[mask, 1], mode="markers", name=f"{name} ({int(mask.sum())})",
                                     marker=dict(size=6, color=color, symbol=symbol, line=dict(width=1, color=color))))
    return _plane(fig, s1, s2, 500)


def build_fall_profile(profile: dict, cost_key: str, N_sel: int) -> go.Figure:
    """Fälle gegen den Horizont N für die gewählten Kosten (vorgerechnete Reihe, Zufalls-Seed und Starts wie in der Tabelle)."""
    rows = profile["costs"][cost_key]
    ns = [r["N"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ns, y=[r["faelle_aussen"] for r in rows], name="Fälle, Start außerhalb C", marker_color=C.COLOR_FALL))
    fig.add_trace(go.Bar(x=ns, y=[r["faelle_innen"] for r in rows], name="Fälle, Start in C (Austritt)", marker_color=C.COLOR_EXIT))
    fig.add_trace(go.Bar(x=ns, y=[r["randlauf"] for r in rows], name="Randlauf (nicht als Fall)", marker_color="#bbbbbb", visible="legendonly"))
    fig.add_vline(x=N_sel, line=dict(color="#222222", width=1.5, dash="dot"))
    _fixed(fig, 340, True, "Horizont N", f"Starts (von {rows[0]['starts']})")
    fig.update_layout(barmode="stack")
    fig.update_xaxes(dtick=1)
    return fig


# --- Wagen: Terminalmenge ------------------------------------------------------------------------------------------------------------------------

def build_terminal(a) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(poly_trace(a.reg.X, "Kasten X", "rgba(150,150,150,0.08)", C.COLOR_EDGE, width=1, dash="dot"))
    fig.add_trace(poly_trace(a.reg.Cset, "C", "rgba(47,107,101,0.20)", C.COLOR_C))
    fig.add_trace(poly_trace(a.T.K(a.N), f"K_{a.N}: lösbar mit Terminalmenge", C.COLOR_K, C.COLOR_KLINE))
    fig.add_trace(poly_trace(a.T.Xf, f"Terminalmenge {C.TERM_LABELS[a.term_key]}", "rgba(120,60,0,0.45)", "#7a3b00", width=2))
    p = C.START_POINTS[a.start_key]
    x, y = float(p["x"][0]), float(p["x"][1])
    fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers", name=p["label"], marker=dict(size=10, symbol="diamond", color="#222222")))
    return _plane(fig, float(a.s1), float(a.s2))


def build_share_chart(analyses: dict, N_sel: int, reg) -> go.Figure:
    """Anteil von K_N an C gegen N für die Terminalmengen; gestrichelt: F_N ohne Terminalmenge."""
    fig = go.Figure()
    n_max = len(next(iter(analyses.values())).shares) - 1
    ns = list(range(n_max + 1))
    f_share = [float(P.area_of(reg.F(n)) / P.area_of(reg.Cset)) * 100 for n in ns]
    fig.add_trace(go.Scatter(x=ns, y=f_share, mode="lines", name="F_N ohne Terminalmenge (Anteil an C)", line=dict(color="#555555", dash="dash")))
    colors = {"lqr1": C.COLOR_KLINE, "lqr001": "#9467bd", "null": "#8c564b"}
    for key, an in analyses.items():
        fig.add_trace(go.Scatter(x=ns, y=[float(v) * 100 for v in an.shares], mode="lines+markers", name=f"K_N, {C.TERM_LABELS[key]}", line=dict(color=colors[key]), marker=dict(size=6)))
    fig.add_vline(x=N_sel, line=dict(color="#222222", width=1.5, dash="dot"))
    fig.add_hline(y=100, line=dict(color=C.COLOR_C, width=1, dash="dash"))
    _fixed(fig, 340, True, "Horizont N", "Anteil an der Fläche von C (%)")
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(range=[-3, 110])
    return fig


# --- Stellgrenze ---------------------------------------------------------------------------------------------------------------------------------

def build_umax(table: list, s1: float, s2: float, umax_sel: float) -> go.Figure:
    """Die Mengen C für die fünf Stellgrenzen (bei der gewählten Positionsgrenze), ineinander gelegt."""
    import mzl_regions as R

    fig = go.Figure()
    palette = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]
    for col, um in zip(palette, C.UMAX_OPTIONS):
        reg = R.regions(um, s1, s2)
        sel = um == umax_sel
        fig.add_trace(poly_trace(reg.Cset, f"C bei |u| ≤ {ev.fr(P.frac(um))} (N̄ = {reg.n_bar})", "rgba(0,0,0,0)" if not sel else "rgba(47,107,101,0.18)", col, width=3 if sel else 1.5))
    return _plane(fig, s1, s2, 470)
