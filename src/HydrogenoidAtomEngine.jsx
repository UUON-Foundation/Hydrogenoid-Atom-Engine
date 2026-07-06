/*
 * UUON-HAE-001  |  Hydrogenoid Atom Engine  v1.0
 * -----------------------------------------------
 * UUON Foundation Inc. — engine metadata suggested, adjust IDs/AUTHORS as needed.
 *
 * Computes and renders the true probability density |psi_nlm|^2 of a
 * hydrogen-like (one-electron) ion via 3D Cartesian rejection sampling.
 * No hardcoded shell geometry — radial part uses the generalized Laguerre
 * polynomial, angular part uses the associated Legendre polynomial, both
 * built from real recurrence relations.
 *
 * Two physically distinct rendering modes are exposed on purpose:
 *   REAL     -> chemist orbitals (real linear combos of +m/-m). Lobed.
 *               Eigenstates of Energy and L^2, NOT of Lz individually.
 *   COMPLEX  -> true |n,l,m> eigenstates. Density has no phi-dependence:
 *               axially symmetric rings. Eigenstates of Energy, L^2, AND Lz.
 * Conflating these two is the single most common error in orbital viz.
 *
 * Design tokens: void black #04080f, teal #00E5CC, gold #C9A84C, violet #7B5EA7.
 */

import React, { useEffect, useRef, useState, useCallback } from "react";
import * as THREE from "three";

// ---------------------------------------------------------------------------
// Design tokens
// ---------------------------------------------------------------------------
const VOID = "#04080f";
const TEAL = "#00E5CC";
const GOLD = "#C9A84C";
const VIOLET = "#7B5EA7";

// ---------------------------------------------------------------------------
// Pure math: factorial, double factorial, generalized Laguerre, associated Legendre
// ---------------------------------------------------------------------------
function factorial(n) {
  let r = 1;
  for (let i = 2; i <= n; i++) r *= i;
  return r;
}

function doubleFactorial(n) {
  let r = 1;
  for (let i = n; i > 1; i -= 2) r *= i;
  return r;
}

// Generalized Laguerre polynomial L_n^k(x) via three-term recurrence.
function genLaguerre(n, k, x) {
  if (n === 0) return 1;
  if (n === 1) return 1 + k - x;
  let Lm2 = 1;
  let Lm1 = 1 + k - x;
  let L = 0;
  for (let i = 2; i <= n; i++) {
    L = ((2 * i - 1 + k - x) * Lm1 - (i - 1 + k) * Lm2) / i;
    Lm2 = Lm1;
    Lm1 = L;
  }
  return L;
}

// Associated Legendre P_l^m(x), m >= 0, via standard stable recurrence.
function assocLegendre(l, m, x) {
  let pmm = 1.0;
  if (m > 0) {
    const somx2 = Math.sqrt(Math.max(0, (1 - x) * (1 + x)));
    let fact = 1.0;
    for (let i = 1; i <= m; i++) {
      pmm *= -fact * somx2;
      fact += 2.0;
    }
  }
  if (l === m) return pmm;
  let pmmp1 = x * (2 * m + 1) * pmm;
  if (l === m + 1) return pmmp1;
  let pll = 0;
  for (let ll = m + 2; ll <= l; ll++) {
    pll = ((2 * ll - 1) * x * pmmp1 - (ll + m - 1) * pmm) / (ll - m);
    pmm = pmmp1;
    pmmp1 = pll;
  }
  return pll;
}

// Radial wavefunction R_nl(r), Z in atomic units, a0 = 1.
function radialR(n, l, Z, r) {
  const rho = (2 * Z * r) / n;
  const norm = Math.sqrt(
    Math.pow((2 * Z) / n, 3) * (factorial(n - l - 1) / (2 * n * factorial(n + l)))
  );
  const lag = genLaguerre(n - l - 1, 2 * l + 1, rho);
  return norm * Math.pow(rho, l) * Math.exp(-rho / 2) * lag;
}

// Normalized associated Legendre prefactor K_lm shared by real & complex forms.
function Klm(l, m) {
  const am = Math.abs(m);
  return Math.sqrt(((2 * l + 1) / (4 * Math.PI)) * (factorial(l - am) / factorial(l + am)));
}

// Real spherical harmonic value (chemist orbital angular part).
function realY(l, m, theta, phi) {
  const am = Math.abs(m);
  const P = assocLegendre(l, am, Math.cos(theta));
  const k = Klm(l, m);
  if (m === 0) return k * P;
  if (m > 0) return Math.SQRT2 * k * P * Math.cos(m * phi);
  return Math.SQRT2 * k * P * Math.sin(am * phi);
}

// Complex eigenstate angular magnitude^2 (phi-independent by construction).
function complexYmag2(l, m, theta) {
  const am = Math.abs(m);
  const P = assocLegendre(l, am, Math.cos(theta));
  const k = Klm(l, m);
  return k * k * P * P;
}

function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}

// Angular density factor |Y|^2, shared by point-cloud sampling and isosurface extraction.
function angularDensityFactor(l, m, theta, phi, realMode) {
  if (realMode) {
    const v = realY(l, m, theta, phi);
    return v * v;
  }
  return complexYmag2(l, m, theta);
}

// Coarse Monte Carlo probe for the peak density, used to scale both the point-cloud
// acceptance threshold and the isosurface level to the same physical units.
function estimateMaxDensity(n, l, m, Z, realMode, R, probes = 6000) {
  let maxD = 1e-12;
  for (let i = 0; i < probes; i++) {
    const x = (Math.random() * 2 - 1) * R;
    const y = (Math.random() * 2 - 1) * R;
    const z = (Math.random() * 2 - 1) * R;
    const r = Math.sqrt(x * x + y * y + z * z);
    if (r < 1e-9) continue;
    const theta = Math.acos(clamp(z / r, -1, 1));
    const phi = Math.atan2(y, x);
    const Rnl = radialR(n, l, Z, r);
    const d = Rnl * Rnl * angularDensityFactor(l, m, theta, phi, realMode);
    if (d > maxD) maxD = d;
  }
  return maxD * 1.35;
}

// R_nl(r)^2 sampled on a fixed radial grid. Because the wavefunction is separable
// (R(r) does not depend on theta/phi), this single 1D profile is reused for every
// direction below — the isosurface reduces to a 1D root-find per direction, not a
// generic 3D marching-cubes problem.
function buildRadialProfile(n, l, Z, R, steps) {
  const arr = new Float32Array(steps + 1);
  for (let i = 0; i <= steps; i++) {
    const r = (R * i) / steps;
    const Rnl = radialR(n, l, Z, r);
    arr[i] = Rnl * Rnl;
  }
  return arr;
}

// Scan the radial profile from the outside in and return the radius of the
// outermost crossing of `target`, or null if this direction never reaches it
// (an angular node — correctly produces a gap in the mesh, not an error).
function findOuterCrossing(radialProfile, R, steps, target) {
  for (let i = steps; i > 0; i--) {
    if (radialProfile[i] < target && radialProfile[i - 1] >= target) {
      const r0 = (R * (i - 1)) / steps;
      const r1 = (R * i) / steps;
      const d0 = radialProfile[i - 1];
      const d1 = radialProfile[i];
      const frac = (d0 - target) / (d0 - d1);
      return r0 + (r1 - r0) * frac;
    }
  }
  return null;
}

// Outermost isosurface of |psi|^2, rendered as a lat/long wireframe. Only the
// single outermost shell is extracted — orbitals with radial nodes (3s, 4d, ...)
// have inner shells this does not draw. That's a scoped limitation, not a bug.
function generateIsosurfaceOuter({ n, l, m, Z, realMode, isoFraction, thetaSteps = 40, phiSteps = 80, radialSteps = 260 }) {
  const R = (1.8 * n * n) / Z + 4;
  const maxD = estimateMaxDensity(n, l, m, Z, realMode, R);
  const threshold = maxD * isoFraction;
  const radialProfile = buildRadialProfile(n, l, Z, R, radialSteps);

  const grid = [];
  for (let i = 0; i <= thetaSteps; i++) {
    const theta = (Math.PI * i) / thetaSteps;
    const row = [];
    for (let j = 0; j < phiSteps; j++) {
      const phi = (2 * Math.PI * j) / phiSteps;
      const aF = angularDensityFactor(l, m, theta, phi, realMode);
      if (aF <= 1e-12) {
        row.push(null);
        continue;
      }
      const target = threshold / aF;
      const r = findOuterCrossing(radialProfile, R, radialSteps, target);
      row.push(r);
    }
    grid.push(row);
  }

  const teal = new THREE.Color(TEAL);
  const violet = new THREE.Color(VIOLET);
  const segPos = [];
  const segCol = [];
  const dirVec = (theta, phi) => [
    Math.sin(theta) * Math.cos(phi),
    Math.sin(theta) * Math.sin(phi),
    Math.cos(theta),
  ];
  const colorFor = (theta, phi, r) => {
    if (realMode) {
      const psi = radialR(n, l, Z, r) * realY(l, m, theta, phi);
      return psi >= 0 ? teal : violet;
    }
    const hueAngle = (((m * phi) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
    return new THREE.Color().setHSL(hueAngle / (2 * Math.PI), 0.75, 0.55);
  };
  const pushSeg = (p0, p1, c) => {
    segPos.push(p0[0], p0[1], p0[2], p1[0], p1[1], p1[2]);
    segCol.push(c.r, c.g, c.b, c.r, c.g, c.b);
  };

  for (let i = 0; i <= thetaSteps; i++) {
    const theta = (Math.PI * i) / thetaSteps;
    for (let j = 0; j < phiSteps; j++) {
      const phi = (2 * Math.PI * j) / phiSteps;
      const r = grid[i][j];
      if (r == null) continue;
      const [dx, dy, dz] = dirVec(theta, phi);
      const p = [dx * r, dy * r, dz * r];
      const c = colorFor(theta, phi, r);

      const jN = (j + 1) % phiSteps;
      const rN = grid[i][jN];
      if (rN != null) {
        const phiN = (2 * Math.PI * jN) / phiSteps;
        const [dxN, dyN, dzN] = dirVec(theta, phiN);
        pushSeg(p, [dxN * rN, dyN * rN, dzN * rN], c);
      }
      if (i < thetaSteps) {
        const thetaS = (Math.PI * (i + 1)) / thetaSteps;
        const rS = grid[i + 1][j];
        if (rS != null) {
          const [dxS, dyS, dzS] = dirVec(thetaS, phi);
          pushSeg(p, [dxS * rS, dyS * rS, dzS * rS], c);
        }
      }
    }
  }

  return { positions: new Float32Array(segPos), colors: new Float32Array(segCol) };
}

// ---------------------------------------------------------------------------
// Point cloud generation: Cartesian rejection sampling of |psi|^2
// ---------------------------------------------------------------------------
function generateCloud({ n, l, m, Z, realMode, pointCount }) {
  // Radial extent scales ~ n^2 a0 / Z ; generous margin for outer lobes.
  const R = (1.8 * n * n) / Z + 4;

  const densityAt = (x, y, z) => {
    const r = Math.sqrt(x * x + y * y + z * z);
    if (r < 1e-9) return { d: 0, sign: 1, phi: 0 };
    const theta = Math.acos(clamp(z / r, -1, 1));
    const phi = Math.atan2(y, x);
    const Rnl = radialR(n, l, Z, r);
    if (realMode) {
      const psi = Rnl * realY(l, m, theta, phi);
      return { d: psi * psi, sign: psi >= 0 ? 1 : -1, phi };
    } else {
      const d = Rnl * Rnl * complexYmag2(l, m, theta);
      return { d, sign: 1, phi };
    }
  };

  // Estimate max density via coarse probing.
  let maxD = 1e-12;
  const probes = 6000;
  for (let i = 0; i < probes; i++) {
    const x = (Math.random() * 2 - 1) * R;
    const y = (Math.random() * 2 - 1) * R;
    const z = (Math.random() * 2 - 1) * R;
    const { d } = densityAt(x, y, z);
    if (d > maxD) maxD = d;
  }
  maxD *= 1.35;

  const positions = new Float32Array(pointCount * 3);
  const colors = new Float32Array(pointCount * 3);
  const maxCandidates = pointCount * 80;
  let accepted = 0;
  let candidates = 0;

  const teal = new THREE.Color(TEAL);
  const violet = new THREE.Color(VIOLET);
  const gold = new THREE.Color(GOLD);
  const tmp = new THREE.Color();

  while (accepted < pointCount && candidates < maxCandidates) {
    candidates++;
    const x = (Math.random() * 2 - 1) * R;
    const y = (Math.random() * 2 - 1) * R;
    const z = (Math.random() * 2 - 1) * R;
    const { d, sign, phi } = densityAt(x, y, z);
    if (d <= 0) continue;
    if (Math.random() * maxD < d) {
      const idx = accepted * 3;
      positions[idx] = x;
      positions[idx + 1] = y;
      positions[idx + 2] = z;

      const brightness = clamp(Math.sqrt(d / maxD), 0.15, 1);

      if (realMode) {
        tmp.copy(sign > 0 ? teal : violet).multiplyScalar(brightness);
      } else if (m === 0) {
        tmp.copy(gold).multiplyScalar(brightness);
      } else {
        // Hue-cycle by azimuthal phase m*phi -> shows Lz eigenstate structure.
        const hueAngle = (((m * phi) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
        tmp.setHSL(hueAngle / (2 * Math.PI), 0.75, 0.3 + 0.4 * brightness);
      }
      colors[idx] = tmp.r;
      colors[idx + 1] = tmp.g;
      colors[idx + 2] = tmp.b;

      accepted++;
    }
  }

  return { positions, colors, actualCount: accepted, boundingR: R };
}

function easeInOutCubic(t) {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

// Soft circular sprite so points render as glowing dots, not flat square quads
// (THREE.PointsMaterial with no map renders literal squares — that's the actual
// cause of the "cube" look, not anything to do with the physics).
function makeCircleSprite() {
  const size = 64;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  const grad = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(0.5, "rgba(255,255,255,0.55)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
  return new THREE.CanvasTexture(canvas);
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export default function HydrogenoidAtomEngine() {
  const mountRef = useRef(null);
  const rendererRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const pointsRef = useRef(null);
  const wireframeRef = useRef(null);
  const rafRef = useRef(null);
  const wireDebounceRef = useRef(null);

  // Imperative mirror of all params, read every rAF tick / regen call.
  const paramsRef = useRef({
    n: 3,
    l: 1,
    m: 0,
    Z: 1,
    realMode: true,
    pointCount: 12000,
    autoRotate: true,
    wireMode: "points", // 'points' | 'wire' | 'both'
    isoFraction: 0.18,
  });

  // Camera orbit state (manual controls — no OrbitControls dependency).
  // theta/phi/radius are the *displayed* values; velocities give drag inertia and
  // eased (non-constant) idle rotation instead of a rigid fixed-rate spin.
  // targetRadius is eased toward each frame instead of being snapped to directly.
  const camState = useRef({
    theta: 0.9,
    phi: 1.1,
    radius: 40,
    targetRadius: 40,
    thetaVel: 0,
    phiVel: 0,
    idleNoise: 0,
    dragging: false,
    lastX: 0,
    lastY: 0,
  });

  // Point-cloud motion state: base (target) positions/colors, the previous cloud
  // (for morphing between parameter changes instead of hard-swapping), and a
  // per-point mass-spring system (offset/velocity/next-kick-time) that drives
  // erratic, non-periodic vibration instead of a synchronized sine wave.
  const cloudMotionRef = useRef({
    basePositions: null,
    baseColors: null,
    prevPositions: null,
    offset: null,
    vel: null,
    nextKick: null,
    boundingR: 40,
    count: 0,
    morphing: false,
    morphStart: 0,
    morphDuration: 700,
  });
  const frameTimeRef = useRef(performance.now());

  // React display mirrors (drive UI only).
  const [n, setNDisplay] = useState(3);
  const [l, setLDisplay] = useState(1);
  const [m, setMDisplay] = useState(0);
  const [Z, setZDisplay] = useState(1);
  const [realMode, setRealModeDisplay] = useState(true);
  const [pointCount, setPointCountDisplay] = useState(12000);
  const [autoRotate, setAutoRotateDisplay] = useState(true);
  const [wireMode, setWireModeDisplay] = useState("points");
  const [isoFraction, setIsoFractionDisplay] = useState(0.18);
  const [actualCount, setActualCount] = useState(0);
  const [showInfo, setShowInfo] = useState(false);
  const [computing, setComputing] = useState(false);

  const energyEV = (-13.6 * Z * Z) / (n * n);

  // ---- write-through setters: update ref + display state together ----
  const applyClampedParams = useCallback((next) => {
    const p = { ...paramsRef.current, ...next };
    p.n = clamp(Math.round(p.n), 1, 8);
    p.l = clamp(Math.round(p.l), 0, p.n - 1);
    p.m = clamp(Math.round(p.m), -p.l, p.l);
    p.Z = clamp(Math.round(p.Z), 1, 10);
    p.pointCount = clamp(Math.round(p.pointCount), 2000, 20000);
    p.isoFraction = clamp(p.isoFraction, 0.03, 0.6);
    if (!["points", "wire", "both"].includes(p.wireMode)) p.wireMode = "points";
    paramsRef.current = p;
    setNDisplay(p.n);
    setLDisplay(p.l);
    setMDisplay(p.m);
    setZDisplay(p.Z);
    setRealModeDisplay(p.realMode);
    setPointCountDisplay(p.pointCount);
    setAutoRotateDisplay(p.autoRotate);
    setWireModeDisplay(p.wireMode);
    setIsoFractionDisplay(p.isoFraction);
  }, []);

  const setN = (v) => applyClampedParams({ n: v });
  const setL = (v) => applyClampedParams({ l: v });
  const setM = (v) => applyClampedParams({ m: v });
  const setZ = (v) => applyClampedParams({ Z: v });
  const setPointCount = (v) => applyClampedParams({ pointCount: v });
  const setIsoFraction = (v) => applyClampedParams({ isoFraction: v });
  const setWireMode = (mode) => applyClampedParams({ wireMode: mode });
  const toggleRealMode = () => applyClampedParams({ realMode: !paramsRef.current.realMode });
  const toggleAutoRotate = () => applyClampedParams({ autoRotate: !paramsRef.current.autoRotate });

  const setAutoOrbitalPreset = (nn, ll, mm) => applyClampedParams({ n: nn, l: ll, m: mm });

  // ---- regeneration: recompute point cloud, write into Three.js geometry ----
  const regenerate = useCallback(() => {
    if (!pointsRef.current) return;
    setComputing(true);
    // Slight defer so the "computing" state can paint before the (sync) heavy work.
    requestAnimationFrame(() => {
      const p = paramsRef.current;
      const { positions, colors, actualCount: ac, boundingR } = generateCloud(p);
      const cm = cloudMotionRef.current;

      // If the point count matches the previous cloud, morph from old -> new
      // positions instead of teleporting. A mismatched count (pointCount slider
      // moved) can't morph 1:1, so it falls back to an instant set — still
      // correct, just not animated for that one transition.
      const canMorph = cm.basePositions && cm.basePositions.length === positions.length;
      cm.prevPositions = canMorph ? cm.basePositions : null;
      cm.basePositions = positions;
      cm.baseColors = colors;
      cm.boundingR = boundingR;
      cm.count = ac;
      cm.morphing = canMorph;
      cm.morphStart = performance.now();

      // Reset the spring-mass system for the new cloud. Kicks are staggered
      // with a random initial delay so every point doesn't jump on the same frame.
      cm.offset = new Float32Array(ac * 3);
      cm.vel = new Float32Array(ac * 3);
      cm.nextKick = new Float32Array(ac);
      const nowT = performance.now();
      for (let i = 0; i < ac; i++) cm.nextKick[i] = nowT + Math.random() * 1400;

      const geo = pointsRef.current.geometry;
      // Position buffer is written every frame in the render loop (base + jitter/morph),
      // so it only needs to exist with the right length here.
      if (!geo.getAttribute("position") || geo.getAttribute("position").array.length !== positions.length) {
        geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(positions.length), 3));
      }
      geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      geo.computeBoundingSphere();
      setActualCount(ac);
      // Ease the camera to the new scale instead of snapping to it.
      camState.current.targetRadius = Math.max(20, boundingR * 1.8);
      setComputing(false);
    });
  }, []);

  // Wireframe is debounced, not recomputed per-frame or even on every keystroke —
  // it's a settled isosurface, not something that needs to track a live drag.
  const regenerateWireframe = useCallback(() => {
    if (wireDebounceRef.current) clearTimeout(wireDebounceRef.current);
    wireDebounceRef.current = setTimeout(() => {
      if (!wireframeRef.current) return;
      const p = paramsRef.current;
      if (p.wireMode === "points") return; // nothing to compute, wireframe hidden
      const { positions, colors } = generateIsosurfaceOuter({
        n: p.n,
        l: p.l,
        m: p.m,
        Z: p.Z,
        realMode: p.realMode,
        isoFraction: p.isoFraction,
      });
      const geo = wireframeRef.current.geometry;
      geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      geo.computeBoundingSphere();
    }, 180);
  }, []);
  useEffect(() => {
    const mount = mountRef.current;
    const width = mount.clientWidth;
    const height = mount.clientHeight;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(VOID);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(55, width / height, 0.1, 2000);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Nucleus marker.
    const nucleusGeo = new THREE.SphereGeometry(0.35, 16, 16);
    const nucleusMat = new THREE.MeshBasicMaterial({ color: GOLD });
    const nucleus = new THREE.Mesh(nucleusGeo, nucleusMat);
    scene.add(nucleus);

    const geometry = new THREE.BufferGeometry();
    const spriteTex = makeCircleSprite();
    const material = new THREE.PointsMaterial({
      size: 0.2,
      map: spriteTex,
      vertexColors: true,
      transparent: true,
      opacity: 0.9,
      depthWrite: false,
      sizeAttenuation: true,
      blending: THREE.AdditiveBlending,
    });
    const points = new THREE.Points(geometry, material);
    scene.add(points);
    pointsRef.current = points;

    const wireGeo = new THREE.BufferGeometry();
    const wireMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.6 });
    const wireframe = new THREE.LineSegments(wireGeo, wireMat);
    wireframe.visible = false;
    scene.add(wireframe);
    wireframeRef.current = wireframe;

    // Manual orbit controls: drag to rotate, wheel to zoom.
    const onPointerDown = (e) => {
      camState.current.dragging = true;
      camState.current.lastX = e.clientX;
      camState.current.lastY = e.clientY;
    };
    const onPointerUp = () => {
      camState.current.dragging = false;
    };
    const onPointerMove = (e) => {
      if (!camState.current.dragging) return;
      const dx = e.clientX - camState.current.lastX;
      const dy = e.clientY - camState.current.lastY;
      camState.current.lastX = e.clientX;
      camState.current.lastY = e.clientY;
      const thetaDelta = -dx * 0.005;
      const phiDelta = -dy * 0.005;
      camState.current.theta += thetaDelta;
      camState.current.phi = clamp(camState.current.phi + phiDelta, 0.1, Math.PI - 0.1);
      // Record velocity so releasing the drag carries momentum instead of
      // stopping dead — this is what "inertia" actually means mechanically.
      camState.current.thetaVel = thetaDelta;
      camState.current.phiVel = phiDelta;
    };
    const onWheel = (e) => {
      e.preventDefault();
      camState.current.targetRadius = clamp(camState.current.targetRadius * (1 + e.deltaY * 0.001), 8, 400);
    };
    renderer.domElement.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("pointerup", onPointerUp);
    window.addEventListener("pointermove", onPointerMove);
    renderer.domElement.addEventListener("wheel", onWheel, { passive: false });

    const onResize = () => {
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    const resizeObserver = new ResizeObserver(onResize);
    resizeObserver.observe(mount);

    // rAF loop — imperative, reads camState + paramsRef.current + cloudMotionRef every tick.
    const tick = () => {
      const now = performance.now();
      const dt = clamp((now - frameTimeRef.current) / 1000, 0, 0.05); // capped so a stalled tab doesn't blow up the spring
      frameTimeRef.current = now;
      const cs = camState.current;

      const wm = paramsRef.current.wireMode;
      points.visible = wm !== "wire";
      wireframe.visible = wm !== "points";

      // --- Camera: velocity spring toward an idle target that itself drifts via a
      // mean-reverting random walk (Ornstein-Uhlenbeck-style), not a fixed sine wave.
      // A deterministic sine is still perfectly periodic — this isn't, which is the
      // actual difference between "smooth" and "erratic" mechanically.
      if (!cs.dragging) {
        cs.idleNoise += (Math.random() - 0.5) * 0.0007;
        cs.idleNoise *= 0.95; // mean-reverting so it wanders instead of drifting away
        const idleVel = paramsRef.current.autoRotate ? 0.0018 + cs.idleNoise : 0;
        cs.thetaVel += (idleVel - cs.thetaVel) * 0.04;
        cs.phiVel *= 0.92;
        cs.theta += cs.thetaVel;
        cs.phi = clamp(cs.phi + cs.phiVel, 0.1, Math.PI - 0.1);
      }
      cs.radius += (cs.targetRadius - cs.radius) * 0.08; // eased zoom, never snaps

      const x = cs.radius * Math.sin(cs.phi) * Math.cos(cs.theta);
      const y = cs.radius * Math.cos(cs.phi);
      const z = cs.radius * Math.sin(cs.phi) * Math.sin(cs.theta);
      camera.position.set(x, y, z);
      camera.lookAt(0, 0, 0);

      // --- Point cloud: morph old -> new on parameter change, plus a per-point
      // mass-spring system. Each point sits at rest until a randomly-scheduled
      // impulse kicks it in a random direction; it then rings back via an
      // underdamped spring. Staggered, non-periodic, erratic by construction —
      // not a synchronized global oscillation.
      const cm = cloudMotionRef.current;
      if (cm.basePositions && pointsRef.current) {
        const posAttr = pointsRef.current.geometry.getAttribute("position");
        if (posAttr && posAttr.array.length === cm.basePositions.length) {
          const arr = posAttr.array;
          let eased = 1;
          if (cm.morphing) {
            const t = clamp((now - cm.morphStart) / cm.morphDuration, 0, 1);
            eased = easeInOutCubic(t);
            if (t >= 1) cm.morphing = false;
          }
          // Kick strength and spring constants scale with the cloud's own radius
          // so the effect reads as proportionate at any n/Z, not a fixed absolute wobble.
          const kickMag = cm.boundingR * 0.05;
          const stiffness = 55; // higher = snaps back faster
          const damping = 5.5; // < 2*sqrt(stiffness) (~14.8) so it's underdamped and rings
          const { offset, vel, nextKick } = cm;

          for (let i = 0; i < cm.count; i++) {
            const idx = i * 3;

            if (now >= nextKick[i]) {
              const u = Math.random() * 2 - 1;
              const th = Math.random() * Math.PI * 2;
              const s = Math.sqrt(Math.max(0, 1 - u * u));
              const mag = kickMag * (0.5 + Math.random() * 0.9);
              vel[idx] += s * Math.cos(th) * mag;
              vel[idx + 1] += s * Math.sin(th) * mag;
              vel[idx + 2] += u * mag;
              nextKick[i] = now + 220 + Math.random() * 900; // staggered, erratic timing
            }

            // Semi-implicit Euler integration of a damped spring pulling offset -> 0.
            for (let a = 0; a < 3; a++) {
              const j = idx + a;
              const accel = -stiffness * offset[j] - damping * vel[j];
              vel[j] += accel * dt;
              offset[j] += vel[j] * dt;
            }

            const bx = cm.basePositions[idx];
            const by = cm.basePositions[idx + 1];
            const bz = cm.basePositions[idx + 2];
            let px = bx, py = by, pz = bz;
            if (cm.morphing && cm.prevPositions) {
              px = cm.prevPositions[idx] + (bx - cm.prevPositions[idx]) * eased;
              py = cm.prevPositions[idx + 1] + (by - cm.prevPositions[idx + 1]) * eased;
              pz = cm.prevPositions[idx + 2] + (bz - cm.prevPositions[idx + 2]) * eased;
            }
            arr[idx] = px + offset[idx];
            arr[idx + 1] = py + offset[idx + 1];
            arr[idx + 2] = pz + offset[idx + 2];
          }
          posAttr.needsUpdate = true;
        }
      }

      renderer.render(scene, camera);
      rafRef.current = requestAnimationFrame(tick);
    };
    tick();

    // Initial cloud.
    regenerate();
    regenerateWireframe();

    return () => {
      cancelAnimationFrame(rafRef.current);
      if (wireDebounceRef.current) clearTimeout(wireDebounceRef.current);
      resizeObserver.disconnect();
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("pointerup", onPointerUp);
      window.removeEventListener("pointermove", onPointerMove);
      renderer.domElement.removeEventListener("wheel", onWheel);
      renderer.dispose();
      geometry.dispose();
      material.dispose();
      spriteTex.dispose();
      wireGeo.dispose();
      wireMat.dispose();
      nucleusGeo.dispose();
      nucleusMat.dispose();
      if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---- recompute cloud + wireframe whenever physical params change ----
  useEffect(() => {
    regenerate();
    regenerateWireframe();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [n, l, m, Z, realMode, pointCount, wireMode, isoFraction, regenerate, regenerateWireframe]);

  const lMax = n - 1;

  return (
    <div
      style={{
        width: "100%",
        height: "100vh",
        background: VOID,
        position: "relative",
        fontFamily: "'Rajdhani', sans-serif",
        color: "#e8f4f2",
        overflow: "hidden",
      }}
    >
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&display=swap');
        .hae-mono { font-family: 'Share Tech Mono', monospace; }
        .hae-slider { -webkit-appearance: none; width: 100%; height: 3px; background: #1a2530; outline: none; border-radius: 2px; }
        .hae-slider::-webkit-slider-thumb { -webkit-appearance: none; width: 13px; height: 13px; border-radius: 50%; background: ${TEAL}; cursor: pointer; box-shadow: 0 0 6px ${TEAL}; }
        .hae-slider::-moz-range-thumb { width: 13px; height: 13px; border-radius: 50%; background: ${TEAL}; cursor: pointer; border: none; box-shadow: 0 0 6px ${TEAL}; }
        .hae-btn { background: transparent; border: 1px solid #2a3a45; color: #cfe; padding: 6px 12px; font-family: 'Share Tech Mono', monospace; font-size: 12px; cursor: pointer; letter-spacing: 0.05em; transition: all 0.15s; }
        .hae-btn:hover { border-color: ${TEAL}; color: ${TEAL}; }
        .hae-btn.active { border-color: ${GOLD}; color: ${GOLD}; background: rgba(201,168,76,0.08); }
        .hae-panel { background: rgba(4,8,15,0.85); border: 1px solid #1a2530; backdrop-filter: blur(6px); }
      `}</style>

      <div ref={mountRef} style={{ width: "100%", height: "100%" }} />

      {/* Title */}
      <div style={{ position: "absolute", top: 18, left: 20, pointerEvents: "none" }}>
        <div style={{ fontSize: 20, fontWeight: 700, letterSpacing: "0.08em", color: TEAL }}>
          HYDROGENOID ATOM ENGINE
        </div>
        <div className="hae-mono" style={{ fontSize: 11, color: "#7a8fa0", marginTop: 2 }}>
          UUON-HAE-001 · |ψ|² rejection-sampled point cloud
        </div>
      </div>

      {/* State readout */}
      <div className="hae-mono" style={{ position: "absolute", top: 18, right: 20, textAlign: "right", fontSize: 12, color: "#9fb8c4", lineHeight: 1.6 }}>
        <div>n={n}  l={l}  m={m}  Z={Z}</div>
        <div style={{ color: GOLD }}>E = {energyEV.toFixed(2)} eV</div>
        <div style={{ color: "#5a7080" }}>{actualCount.toLocaleString()} pts{computing ? " · resampling…" : ""}</div>
      </div>

      {/* Control panel */}
      <div className="hae-panel" style={{ position: "absolute", bottom: 0, left: 0, right: 0, padding: "14px 20px 18px", display: "flex", flexWrap: "wrap", gap: 22, alignItems: "flex-end" }}>
        <Slider label="n (principal)" value={n} min={1} max={8} onChange={setN} />
        <Slider label={`l (0…${lMax})`} value={l} min={0} max={lMax} onChange={setL} />
        <Slider label={`m (−${l}…${l})`} value={m} min={-l} max={l} onChange={setM} />
        <Slider label="Z (charge)" value={Z} min={1} max={10} onChange={setZ} />
        <Slider label="points" value={pointCount} min={2000} max={20000} step={1000} onChange={setPointCount} />
        {wireMode !== "points" && (
          <Slider label="iso level" value={isoFraction} min={0.03} max={0.6} step={0.01} onChange={setIsoFraction} />
        )}

        <div style={{ display: "flex", gap: 8 }}>
          <button className={`hae-btn ${wireMode === "points" ? "active" : ""}`} onClick={() => setWireMode("points")}>
            POINTS
          </button>
          <button className={`hae-btn ${wireMode === "wire" ? "active" : ""}`} onClick={() => setWireMode("wire")}>
            WIRE
          </button>
          <button className={`hae-btn ${wireMode === "both" ? "active" : ""}`} onClick={() => setWireMode("both")}>
            BOTH
          </button>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button className={`hae-btn ${realMode ? "active" : ""}`} onClick={() => { if (!realMode) toggleRealMode(); }}>
            REAL (lobes)
          </button>
          <button className={`hae-btn ${!realMode ? "active" : ""}`} onClick={() => { if (realMode) toggleRealMode(); }}>
            COMPLEX (Lz eigenstate)
          </button>
          <button className={`hae-btn ${autoRotate ? "active" : ""}`} onClick={toggleAutoRotate}>
            {autoRotate ? "ROTATING" : "ROTATE: OFF"}
          </button>
          <button className="hae-btn" onClick={() => setShowInfo((s) => !s)}>
            {showInfo ? "HIDE MATH" : "SHOW MATH"}
          </button>
        </div>

        <div style={{ display: "flex", gap: 6, marginLeft: "auto" }}>
          <span className="hae-mono" style={{ fontSize: 10, color: "#5a7080", alignSelf: "center", marginRight: 4 }}>presets:</span>
          <button className="hae-btn" onClick={() => setAutoOrbitalPreset(1, 0, 0)}>1s</button>
          <button className="hae-btn" onClick={() => setAutoOrbitalPreset(2, 1, 0)}>2p</button>
          <button className="hae-btn" onClick={() => setAutoOrbitalPreset(3, 2, 1)}>3d</button>
          <button className="hae-btn" onClick={() => setAutoOrbitalPreset(4, 3, 2)}>4f</button>
        </div>
      </div>

      {showInfo && (
        <div className="hae-panel hae-mono" style={{ position: "absolute", top: 90, left: 20, width: 400, padding: 16, fontSize: 11.5, lineHeight: 1.7, color: "#b8ccd6" }}>
          <div style={{ color: GOLD, fontWeight: 700, marginBottom: 8, fontFamily: "'Rajdhani', sans-serif", fontSize: 14 }}>
            ψ_nlm(r,θ,φ) = R_nl(r) · Y_lm(θ,φ)
          </div>
          <div>R_nl(r): normalized radial part, built from the generalized Laguerre polynomial L_(n−l−1)^(2l+1)(ρ), ρ = 2Zr/n.</div>
          <div style={{ marginTop: 6 }}>Y_lm(θ,φ): angular part, built from the associated Legendre polynomial P_l^|m|(cosθ).</div>
          <div style={{ marginTop: 10, color: TEAL }}>REAL mode</div>
          <div>Real linear combinations of +m/−m (the familiar p/d/f lobes). These are eigenstates of energy and L², but NOT of Lz — colored teal/violet by the sign of ψ, which marks the nodal surfaces.</div>
          <div style={{ marginTop: 10, color: VIOLET }}>COMPLEX mode</div>
          <div>True |n,l,m⟩ eigenstates of energy, L², AND Lz. Density |ψ|² has no φ-dependence — it is exactly axially symmetric (rings), never lobed. Color here encodes the phase angle m·φ, not sign.</div>
          <div style={{ marginTop: 10, color: "#5a7080" }}>
            Points are Cartesian-rejection-sampled directly from |ψ|² — no coordinate Jacobian is introduced, so point density in the render is proportional to true probability density in real 3D space. Z&gt;1 models a hydrogenic ion (e.g. He⁺), not a screened multi-electron atom.
          </div>
          <div style={{ marginTop: 10, color: GOLD }}>WIRE mode</div>
          <div>
            Renders the <em>outermost</em> isosurface of |ψ|² only, exploiting separability (R(r) doesn't depend on θ,φ) to reduce the search to a 1D radial profile per direction — not generic marching cubes. Orbitals with radial nodes (3s, 4d, …) have inner shells this does not draw yet.
          </div>
        </div>
      )}
    </div>
  );
}

function Slider({ label, value, min, max, step = 1, onChange }) {
  return (
    <div style={{ minWidth: 110 }}>
      <div className="hae-mono" style={{ fontSize: 10, color: "#7a8fa0", marginBottom: 4, letterSpacing: "0.04em" }}>
        {label.toUpperCase()} <span style={{ color: "#00E5CC" }}>{value}</span>
      </div>
      <input
        type="range"
        className="hae-slider"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}
