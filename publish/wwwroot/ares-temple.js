(function () {
  var templeInstance = null;

  window.AresTemple = {
    init: function (stageId, canvasId, initialPosture) {
      if (templeInstance) {
        templeInstance.destroy();
        templeInstance = null;
      }
      templeInstance = createTemple(stageId, canvasId, initialPosture || "safe");
    },
    setPosture: function (state) {
      if (templeInstance) {
        templeInstance.setPosture(state);
      }
    },
    drawTrend: function (canvasId, trendData) {
      drawTrendChart(canvasId, trendData);
    },
    initTilt: function () {
      initTabletTilt();
    },
    destroy: function () {
      if (templeInstance) {
        templeInstance.destroy();
        templeInstance = null;
      }
    }
  };

  function initTabletTilt() {
    var reduce = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) return;
    document.querySelectorAll(".tablet").forEach(function (t) {
      if (t._tiltBound) return;
      t._tiltBound = true;
      t.addEventListener("pointermove", function (e) {
        var r = t.getBoundingClientRect();
        var x = (e.clientX - r.left) / r.width - 0.5;
        var y = (e.clientY - r.top) / r.height - 0.5;
        t.style.setProperty("--ry", (x * 12).toFixed(2) + "deg");
        t.style.setProperty("--rx", (-y * 12).toFixed(2) + "deg");
      });
      t.addEventListener("pointerleave", function () {
        t.style.setProperty("--ry", "0deg");
        t.style.setProperty("--rx", "0deg");
      });
    });
  }

  function drawTrendChart(canvasId, trendData) {
    var c = document.getElementById(canvasId);
    if (!c) return;

    var C = {
      bronze: "#c08a4a",
      laurel: "#86b394",
      crimson: "#c4343c",
      dim: "#8d8678",
      grid: "rgba(233,228,216,0.08)"
    };

    var N = (trendData && trendData.tested && trendData.tested.length) ? trendData.tested.length : 14;
    var tested = (trendData && trendData.tested && trendData.tested.length) ? trendData.tested : new Array(N).fill(0);
    var blocked = (trendData && trendData.blocked) ? trendData.blocked : new Array(N).fill(0);
    var successful = (trendData && trendData.successful) ? trendData.successful : new Array(N).fill(0);
    var incidents = (trendData && trendData.incidents) ? trendData.incidents : new Array(N).fill(0);
    var labels = (trendData && trendData.labels) ? trendData.labels : [];
    var visible = new Set((trendData && trendData.visible) || ["tested", "blocked", "successful", "incidents"]);
    var activeIndex = -1;

    function render() {
      var dpr = Math.min(window.devicePixelRatio || 1, 2);
      var w = c.clientWidth, h = c.clientHeight;
      if (!w || !h) return;
      c.width = w * dpr;
      c.height = h * dpr;
      var g = c.getContext("2d");
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, w, h);

      var L = 32, R = 16, T = 16, B = 28, pw = w - L - R, ph = h - T - B;
      var allVals = [];
      if (visible.has("tested")) allVals = allVals.concat(tested);
      if (visible.has("blocked")) allVals = allVals.concat(blocked);
      if (visible.has("successful")) allVals = allVals.concat(successful);
      if (visible.has("incidents")) allVals = allVals.concat(incidents);
      if (!allVals.length) allVals = [0];
      var maxVal = Math.max.apply(null, allVals);
      var max = Math.max(3, maxVal);

      function X(i) { return L + pw * i / Math.max(1, N - 1); }
      function Y(v) { return T + ph * (1 - v / max); }

      g.font = "11px 'IBM Plex Mono', monospace";
      g.textBaseline = "middle";

      var steps = Math.min(4, max);
      for (var s = 0; s <= steps; s++) {
        var v = Math.round(s * (max / steps));
        var yPos = Y(v);
        g.strokeStyle = C.grid;
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(L, yPos + 0.5);
        g.lineTo(w - R, yPos + 0.5);
        g.stroke();

        g.fillStyle = C.dim;
        g.textAlign = "right";
        g.fillText(String(v), L - 8, yPos);
      }

      g.textAlign = "center";
      g.textBaseline = "alphabetic";
      if (labels.length > 0) {
        var indices = [0, Math.floor(N * 0.33), Math.floor(N * 0.66), N - 1];
        indices.forEach(function (idx) {
          if (idx >= 0 && idx < labels.length) {
            g.textAlign = idx === 0 ? "left" : (idx === N - 1 ? "right" : "center");
            g.fillStyle = C.dim;
            g.fillText(labels[idx], X(idx) + (idx === 0 ? -4 : idx === N - 1 ? 4 : 0), h - 6);
          }
        });
      }

      function line(data, color, dash, lw) {
        g.beginPath();
        data.forEach(function (v, i) {
          if (i) g.lineTo(X(i), Y(v));
          else g.moveTo(X(i), Y(v));
        });
        g.strokeStyle = color;
        g.lineWidth = lw || 2;
        g.lineJoin = "round";
        g.setLineDash(dash || []);
        g.stroke();
        g.setLineDash([]);
      }

      if (visible.has("incidents")) line(incidents, C.dim, [3, 4], 1.5);
      if (visible.has("successful")) line(successful, C.crimson, [6, 4], 2);
      if (visible.has("blocked")) line(blocked, C.laurel, [8, 4], 2);

      if (visible.has("tested")) {
        var grad = g.createLinearGradient(0, T, 0, T + ph);
        grad.addColorStop(0, "rgba(192,138,74,0.38)");
        grad.addColorStop(0.7, "rgba(192,138,74,0.08)");
        grad.addColorStop(1, "rgba(192,138,74,0)");
        g.beginPath();
        tested.forEach(function (v, i) {
          if (i) g.lineTo(X(i), Y(v));
          else g.moveTo(X(i), Y(v));
        });
        g.lineTo(X(N - 1), Y(0));
        g.lineTo(X(0), Y(0));
        g.closePath();
        g.fillStyle = grad;
        g.fill();
        line(tested, C.bronze, [], 2.5);
      }

      if (visible.has("tested") && tested.length > 0) {
        var ex = X(N - 1), ey = Y(tested[N - 1]);
        g.beginPath();
        g.arc(ex, ey, 8, 0, 6.2832);
        g.fillStyle = "rgba(192,138,74,0.25)";
        g.fill();
        g.beginPath();
        g.arc(ex, ey, 4, 0, 6.2832);
        g.fillStyle = C.bronze;
        g.fill();
      }

      if (activeIndex >= 0 && activeIndex < N) {
        g.strokeStyle = "rgba(233,228,216,0.34)";
        g.setLineDash([3, 4]);
        g.beginPath();
        g.moveTo(X(activeIndex), T);
        g.lineTo(X(activeIndex), T + ph);
        g.stroke();
        g.setLineDash([]);
        var readout = document.getElementById("trend-readout");
        if (readout) readout.textContent = (labels[activeIndex] || "Selected day") + " · " + tested[activeIndex] + " tested · " + blocked[activeIndex] + " blocked · " + successful[activeIndex] + " succeeded · " + incidents[activeIndex] + " incidents";
      }
    }

    render();
    c.onpointermove = function (event) {
      var rect = c.getBoundingClientRect();
      var x = event.clientX - rect.left;
      var L = 32, R = 16;
      var ratio = Math.max(0, Math.min(1, (x - L) / Math.max(1, rect.width - L - R)));
      activeIndex = Math.round(ratio * Math.max(0, N - 1));
      render();
    };
    c.onpointerleave = function () {
      activeIndex = -1;
      render();
      var readout = document.getElementById("trend-readout");
      if (readout) readout.textContent = "Hover over the chart to inspect a day.";
    };
    if (window.ResizeObserver && !c._chartObserver) {
      c._chartObserver = new ResizeObserver(render);
      c._chartObserver.observe(c);
    }
  }

  function createTemple(stageId, canvasId, initialPosture) {
    var stage = document.getElementById(stageId);
    var canvas = document.getElementById(canvasId);
    if (!stage || !canvas) return null;

    var reduce = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;

    var STATES = {
      safe: { rim: 0x86b394, ring: 0x8c1d22, glow: 0.35, css: "#86b394" },
      elevated: { rim: 0xd9a441, ring: 0xa8641e, glow: 0.7, css: "#d9a441" },
      breached: { rim: 0xc4343c, ring: 0xc4343c, glow: 1.3, css: "#c4343c" }
    };
    var FLAVOR = {
      safe: "The flame burns green. Enforcement is holding.",
      elevated: "The flame turns amber. Review recent findings.",
      breached: "The flame runs red. Attacks are getting through."
    };

    var current = initialPosture || "safe";
    stage.style.setProperty("--state", (STATES[current] || STATES.safe).css);
    var flavorEl = document.getElementById("flavor");
    if (flavorEl) flavorEl.textContent = FLAVOR[current] || FLAVOR.safe;

    if (!window.THREE) {
      stage.classList.add("nogl");
      return {
        setPosture: function (st) {
          current = st;
          stage.style.setProperty("--state", (STATES[current] || STATES.safe).css);
          if (flavorEl) flavorEl.textContent = FLAVOR[current] || FLAVOR.safe;
        },
        destroy: function () {}
      };
    }

    var T3 = THREE;
    var renderer;
    try {
      renderer = new T3.WebGLRenderer({
        canvas: canvas,
        antialias: true,
        alpha: true,
        powerPreference: "high-performance"
      });
    } catch (e) {
      stage.classList.add("nogl");
      return {
        setPosture: function (st) {
          current = st;
          stage.style.setProperty("--state", (STATES[current] || STATES.safe).css);
          if (flavorEl) flavorEl.textContent = FLAVOR[current] || FLAVOR.safe;
        },
        destroy: function () {}
      };
    }

    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

    var FLOOR = -2.4;
    var scene = new T3.Scene();
    scene.fog = new T3.FogExp2(0x101f1a, 0.035);
    var cam = new T3.PerspectiveCamera(38, 1.6, 0.1, 160);

    function C3(a, b, c) { return [new T3.Color(a), new T3.Color(b), new T3.Color(c)]; }

    // Chromatic palettes across Safe -> Elevated -> Breached
    var P = {
      skyTop: C3(0x0c1a22, 0x1d1118, 0x180505),
      skyHor: C3(0x356e5c, 0xb86f26, 0xc72714),
      fog: C3(0x101f1a, 0x24150f, 0x1f0a08),
      floor: C3(0x141f1b, 0x1c1411, 0x160909),
      flameCore: C3(0xf6fffa, 0xfff8d8, 0xffe8d8),
      flameTip: C3(0x34d399, 0xf59e0b, 0xef4444),
      flameGlow: C3(0x10b981, 0xd97706, 0xdc2626),
      emberLight: C3(0x6ee7b7, 0xfcd34d, 0xfca5a5),
      coals: C3(0x059669, 0xd97706, 0xb91c1c)
    };

    function mixN(arr, lv, out) {
      if (lv <= 1) out.copy(arr[0]).lerp(arr[1], lv);
      else out.copy(arr[1]).lerp(arr[2], Math.min(1, lv - 1));
      return out;
    }
    function mixV(a, b, c, lv) {
      return lv <= 1 ? a + (b - a) * lv : b + (c - b) * Math.min(1, lv - 1);
    }
    function clamp01(v) { return Math.max(0, Math.min(1, v)); }

    /* ── Atmospheric Sky Dome ── */
    var skyU = { top: { value: new T3.Color() }, hor: { value: new T3.Color() } };
    var sky = new T3.Mesh(
      new T3.SphereGeometry(110, 32, 16),
      new T3.ShaderMaterial({
        side: T3.BackSide,
        depthWrite: false,
        fog: false,
        uniforms: skyU,
        vertexShader: "varying vec3 vP; void main(){ vP = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }",
        fragmentShader: "uniform vec3 top; uniform vec3 hor; varying vec3 vP; void main(){ float h = normalize(vP).y; float k = smoothstep(-0.04, 0.62, h); gl_FragColor = vec4(mix(hor, top, k), 1.0); }"
      })
    );
    sky.renderOrder = -10;
    scene.add(sky);

    /* ── Star Field ── */
    var SN = 280, spos = new Float32Array(SN * 3);
    for (var si = 0; si < SN; si++) {
      var sa = Math.random() * Math.PI * 2, sh = 0.12 + Math.random() * 0.86, sr = 96;
      spos[si * 3] = Math.cos(sa) * Math.sqrt(1 - sh * sh) * sr;
      spos[si * 3 + 1] = sh * sr;
      spos[si * 3 + 2] = Math.sin(sa) * Math.sqrt(1 - sh * sh) * sr;
    }
    var sgeo = new T3.BufferGeometry();
    sgeo.setAttribute("position", new T3.BufferAttribute(spos, 3));
    var starMat = new T3.PointsMaterial({ color: 0xf6f4ed, size: 0.65, transparent: true, opacity: 0.9, fog: false, depthWrite: false });
    scene.add(new T3.Points(sgeo, starMat));

    /* ── Dynamic Lighting ── */
    var amb = new T3.AmbientLight(0x38333e, 1.2);
    scene.add(amb);
    var keySun = new T3.DirectionalLight(0xffeed6, 0.9);
    keySun.position.set(5, 8, 8);
    scene.add(keySun);

    var rimLight = new T3.PointLight(0x86b394, 2.5, 24);
    rimLight.position.set(-6, 2.5, -4);
    scene.add(rimLight);

    var horizonGlow = new T3.PointLight(0xff8833, 0.2, 32);
    horizonGlow.position.set(7, 1.2, -8);
    scene.add(horizonGlow);

    /* ── Floor & Classical Pavement ── */
    var floorMat = new T3.MeshStandardMaterial({
      color: 0x141f1b,
      roughness: 0.42,
      metalness: 0.32
    });
    var floor = new T3.Mesh(new T3.PlaneGeometry(96, 96), floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = FLOOR;
    scene.add(floor);

    var grid = new T3.GridHelper(96, 96, 0x6b4d27, 0x1e1611);
    grid.position.y = FLOOR + 0.005;
    scene.add(grid);

    /* ── Realistic Pentelic Marble Texture ── */
    function makeMarbleTex() {
      var c = document.createElement("canvas");
      c.width = c.height = 512;
      var g = c.getContext("2d");
      g.fillStyle = "#ded6cb";
      g.fillRect(0, 0, 512, 512);

      for (var i = 0; i < 90; i++) {
        g.strokeStyle = "rgba(" + (100 + (Math.random() * 40 | 0)) + ",90,80," + (0.02 + Math.random() * 0.06) + ")";
        g.lineWidth = 1 + Math.random() * 4;
        g.beginPath();
        var x = Math.random() * 512, y = Math.random() * 512;
        g.moveTo(x, y);
        for (var j = 0; j < 5; j++) {
          x += (Math.random() - 0.5) * 160;
          y += (Math.random() - 0.5) * 160;
          g.lineTo(x, y);
        }
        g.stroke();
      }

      for (var k = 0; k < 35; k++) {
        g.strokeStyle = "rgba(192, 138, 74, " + (0.04 + Math.random() * 0.1) + ")";
        g.lineWidth = 0.6 + Math.random() * 1.6;
        g.beginPath();
        var vx = Math.random() * 512, vy = Math.random() * 512;
        g.moveTo(vx, vy);
        for (var m = 0; m < 6; m++) {
          vx += (Math.random() - 0.45) * 120;
          vy += (Math.random() - 0.45) * 120;
          g.lineTo(vx, vy);
        }
        g.stroke();
      }

      var t = new T3.CanvasTexture(c);
      t.wrapS = t.wrapT = T3.RepeatWrapping;
      return t;
    }

    var mTex = makeMarbleTex();
    var marble = new T3.MeshStandardMaterial({ color: 0xe2d9ce, map: mTex, roughness: 0.55, metalness: 0.1 });
    var marbleDark = new T3.MeshStandardMaterial({ color: 0xb7ada0, map: mTex, roughness: 0.68, metalness: 0.08 });
    var bronzePatina = new T3.MeshStandardMaterial({ color: 0x8a6239, roughness: 0.38, metalness: 0.85 });
    var darkIron = new T3.MeshStandardMaterial({ color: 0x221d1b, roughness: 0.6, metalness: 0.7 });
    var goldInlay = new T3.MeshStandardMaterial({ color: 0xd98a3b, roughness: 0.3, metalness: 0.8, emissive: 0x945012, emissiveIntensity: 0.3 });

    /* ── AUTHENTIC CLASSICAL GREEK DORIC COLUMNS (Stacked Tambours/Drums) ── */
    // Real ancient Greek columns are built from modular cylindrical drums (tambours)
    // with 20 vertical flutes and horizontal joint seams.
    var columnAssemblies = [];
    var architraveBeams = [];

    // Fluted Column Drum Generator
    function createFlutedDrumGeo(radiusTop, radiusBottom, height, flutesCount) {
      var segments = flutesCount * 2; // 24 segments for 12 deep flutes
      var geo = new T3.CylinderGeometry(radiusTop, radiusBottom, height, segments, 1, false);
      var pos = geo.attributes.position;
      // Inset alternate vertices to create authentic carved Doric fluting ridges (arrises)
      for (var i = 0; i < pos.count; i++) {
        var x = pos.getX(i);
        var z = pos.getZ(i);
        var angle = Math.atan2(z, x);
        var dist = Math.sqrt(x * x + z * z);
        // Fluting depth modulation
        var fluteMod = 1.0 - Math.pow(Math.sin(angle * flutesCount * 0.5), 2) * 0.055;
        pos.setX(i, x * fluteMod);
        pos.setZ(i, z * fluteMod);
      }
      geo.computeVertexNormals();
      return geo;
    }

    var DRUMS_PER_COLUMN = 5;
    var DRUM_HEIGHT = 0.95;
    var columnPositionsZ = [-1.5, -4.5, -7.5, -10.5, -13.5, -16.5, -19.5];

    columnPositionsZ.forEach(function (z, colIndex) {
      [-1, 1].forEach(function (side) {
        var colRoot = new T3.Group();
        colRoot.position.set(side * 5.2, FLOOR, z);
        scene.add(colRoot);

        // 1. Classical Plinth Base (Stylobate block)
        var plinth = new T3.Mesh(new T3.BoxGeometry(1.2, 0.28, 1.2), marbleDark);
        plinth.position.y = 0.14;
        colRoot.add(plinth);

        var subBase = new T3.Mesh(new T3.CylinderGeometry(0.55, 0.58, 0.18, 24), marble);
        subBase.position.y = 0.37;
        colRoot.add(subBase);

        // 2. Stacked Modular Drums with Entasis (gentle taper)
        var drums = [];
        var currY = 0.46 + DRUM_HEIGHT * 0.5;

        for (var d = 0; d < DRUMS_PER_COLUMN; d++) {
          var tFactor = d / (DRUMS_PER_COLUMN - 1);
          // Ancient Greek entasis: bottom drum is wider, gently tapering up
          var rBottom = 0.48 - tFactor * 0.07;
          var rTop = 0.48 - (tFactor + 0.2) * 0.07;

          var drumGeo = createFlutedDrumGeo(rTop, rBottom, DRUM_HEIGHT, 12);
          var drumMesh = new T3.Mesh(drumGeo, (d % 2 === 0) ? marble : marbleDark);
          drumMesh.position.y = currY;

          // Seam ring between drums
          var seamGeo = new T3.CylinderGeometry(rBottom + 0.015, rBottom + 0.015, 0.025, 24);
          var seamMesh = new T3.Mesh(seamGeo, darkIron);
          seamMesh.position.y = currY - DRUM_HEIGHT * 0.5;
          colRoot.add(seamMesh);

          colRoot.add(drumMesh);

          // Store drum data for authentic seismic shear-slip displacement
          drums.push({
            mesh: drumMesh,
            origX: 0,
            origZ: 0,
            origRotX: 0,
            origRotZ: 0,
            origY: currY,
            drumIndex: d,
            // Higher drums experience greater slip during seismic breach
            shearBiasX: (Math.sin(colIndex * 1.5 + d) * 0.12 + (side * 0.08)) * (d * 0.35),
            shearBiasZ: Math.cos(colIndex * 2.1 + d) * 0.09 * (d * 0.3)
          });

          currY += DRUM_HEIGHT;
        }

        // 3. Doric Capital (Echinus + Abacus)
        var capGroup = new T3.Group();
        capGroup.position.y = currY - DRUM_HEIGHT * 0.5;

        // Echinus (circular flared cushion)
        var echinus = new T3.Mesh(new T3.CylinderGeometry(0.62, 0.38, 0.26, 24), marble);
        echinus.position.y = 0.13;
        capGroup.add(echinus);

        // Abacus (square heavy stone block)
        var abacus = new T3.Mesh(new T3.BoxGeometry(1.28, 0.22, 1.28), marbleDark);
        abacus.position.y = 0.37;
        capGroup.add(abacus);

        colRoot.add(capGroup);

        columnAssemblies.push({
          root: colRoot,
          side: side,
          colIndex: colIndex,
          drums: drums,
          capital: capGroup,
          origY: FLOOR
        });
      });
    });

    // 4. Architrave Entablature Beams bridging the columns
    [-1, 1].forEach(function (side) {
      var beamGroup = new T3.Group();
      beamGroup.position.set(side * 5.2, FLOOR + 5.95, -10.5);

      // Architrave lower beam
      var b1 = new T3.Mesh(new T3.BoxGeometry(0.92, 0.46, 26), marbleDark);
      beamGroup.add(b1);

      // Frieze moulding
      var b2 = new T3.Mesh(new T3.BoxGeometry(1.05, 0.16, 26), marble);
      b2.position.y = 0.31;
      beamGroup.add(b2);

      scene.add(beamGroup);
      architraveBeams.push({
        group: beamGroup,
        side: side,
        origY: FLOOR + 5.95
      });
    });

    /* ── AUTHENTIC SACRIFICIAL ALTAR & CEREMONIAL BRONZE BRAZIER (Delphic Tripod) ── */
    var hero = new T3.Group();
    scene.add(hero);

    var altar = new T3.Group();
    altar.position.y = FLOOR;
    hero.add(altar);

    // 1. Classical Octagonal Altar Base of Pentelic Marble
    function addBox(w, h, d, y, mat, parent) {
      var m = new T3.Mesh(new T3.BoxGeometry(w, h, d), mat);
      m.position.y = y;
      (parent || altar).add(m);
      return m;
    }

    addBox(2.2, 0.18, 2.2, 0.09, marbleDark);
    addBox(1.95, 0.24, 1.95, 0.3, marble);
    addBox(2.0, 0.04, 2.0, 0.44, goldInlay); // Gold Greek Key accent
    addBox(1.75, 1.6, 1.75, 1.26, marble);   // Main Altar Shaft
    addBox(1.8, 0.04, 1.8, 2.08, goldInlay);
    addBox(2.05, 0.18, 2.05, 2.19, marbleDark);
    addBox(1.6, 0.14, 1.6, 2.35, marble);

    // 2. Cast Bronze Sacrificial Tripod Legs (Lion-paw bosses)
    var tripodGroup = new T3.Group();
    tripodGroup.position.y = 2.42;
    altar.add(tripodGroup);

    for (var legI = 0; legI < 3; legI++) {
      var legAngle = (legI / 3) * Math.PI * 2;
      var leg = new T3.Group();
      leg.rotation.y = legAngle;

      // Curved bronze strut
      var legGeo = new T3.CylinderGeometry(0.045, 0.07, 0.82, 12);
      var legMesh = new T3.Mesh(legGeo, bronzePatina);
      legMesh.position.set(0.55, 0.38, 0);
      legMesh.rotation.z = -0.32;
      leg.add(legMesh);

      // Lion paw foot
      var paw = new T3.Mesh(new T3.SphereGeometry(0.08, 10, 8), bronzePatina);
      paw.scale.set(1.4, 0.8, 1);
      paw.position.set(0.68, 0.05, 0);
      leg.add(paw);

      // Top lion head boss
      var boss = new T3.Mesh(new T3.SphereGeometry(0.09, 10, 8), bronzePatina);
      boss.position.set(0.42, 0.72, 0);
      leg.add(boss);

      tripodGroup.add(leg);
    }

    // 3. Heavy Hammered Bronze Basin / Cauldron
    var cauldronPoints = [
      [0, 0], [0.35, 0.02], [0.72, 0.1], [1.05, 0.26], [1.32, 0.48],
      [1.38, 0.54], [1.35, 0.58], [1.18, 0.46], [0.8, 0.28], [0.38, 0.16], [0, 0.14]
    ].map(function (p) { return new T3.Vector2(p[0], p[1]); });

    var cauldron = new T3.Mesh(new T3.LatheGeometry(cauldronPoints, 48), bronzePatina);
    cauldron.position.y = 0.65;
    tripodGroup.add(cauldron);

    // Twin Bronze Ring Handles
    [-1, 1].forEach(function (s) {
      var ring = new T3.Mesh(new T3.TorusGeometry(0.18, 0.035, 12, 24), bronzePatina);
      ring.position.set(s * 1.36, 1.08, 0);
      ring.rotation.y = Math.PI * 0.5;
      ring.rotation.x = 0.4;
      tripodGroup.add(ring);
    });

    // 4. Glowing Carbonized Charcoal / Coal Bed (Crackling Embers)
    var coalGroup = new T3.Group();
    coalGroup.position.set(0, 3.2, 0);
    altar.add(coalGroup);

    var coalMat = new T3.MeshStandardMaterial({
      color: 0x14100e,
      roughness: 0.95,
      emissive: 0x059669,
      emissiveIntensity: 0.85
    });

    // Heap of realistic charcoal rocks inside the basin
    for (var ci = 0; ci < 24; ci++) {
      var cRadius = Math.sqrt(Math.random()) * 0.88;
      var cAngle = Math.random() * Math.PI * 2;
      var cSz = 0.14 + Math.random() * 0.18;
      var coal = new T3.Mesh(new T3.DodecahedronGeometry(cSz, 0), coalMat);
      coal.position.set(
        Math.cos(cAngle) * cRadius,
        0.05 + (1 - (cRadius / 0.88)) * 0.18 + (Math.random() * 0.06),
        Math.sin(cAngle) * cRadius
      );
      coal.rotation.set(Math.random() * 3, Math.random() * 3, Math.random() * 3);
      coalGroup.add(coal);
    }

    var FLAME_Y = FLOOR + 3.42;

    /* ── HYBRID PHOTOREALISTIC FLAME (3D Volumetric Mesh Core + Fluid Particles) ── */
    // 1. Procedural 3D Flame Mesh Core (Ensures true 3D perspective depth, not flat stickers)
    var flameCoreGeo = new T3.ConeGeometry(0.55, 2.2, 24, 18, true);
    // Deform into natural teardrop
    var fcPos = flameCoreGeo.attributes.position;
    for (var fci = 0; fci < fcPos.count; fci++) {
      var fy = fcPos.getY(fci) + 1.1; // 0 to 2.2
      var tNorm = fy / 2.2;
      // Bulge at lower third, sharp pinch at tip
      var bulge = Math.sin(tNorm * Math.PI * 0.85) * 1.35;
      fcPos.setX(fci, fcPos.getX(fci) * bulge);
      fcPos.setZ(fci, fcPos.getZ(fci) * bulge);
    }
    flameCoreGeo.computeVertexNormals();

    var flameCoreMat = new T3.MeshBasicMaterial({
      color: 0xf6fffa,
      transparent: true,
      opacity: 0.72,
      blending: T3.AdditiveBlending,
      side: T3.DoubleSide,
      depthWrite: false
    });

    var flameCoreMesh = new T3.Mesh(flameCoreGeo, flameCoreMat);
    flameCoreMesh.position.set(0, FLAME_Y + 0.95, 0);
    hero.add(flameCoreMesh);

    // Secondary Outer Flame Mantle Mesh
    var flameOuterMat = new T3.MeshBasicMaterial({
      color: 0x34d399,
      transparent: true,
      opacity: 0.45,
      blending: T3.AdditiveBlending,
      side: T3.DoubleSide,
      depthWrite: false
    });
    var flameOuterMesh = new T3.Mesh(flameCoreGeo.clone(), flameOuterMat);
    flameOuterMesh.scale.set(1.35, 1.15, 1.35);
    flameOuterMesh.position.set(0, FLAME_Y + 0.9, 0);
    hero.add(flameOuterMesh);

    // 2. High-Resolution Teardrop Plasma Sprite Texture for Licking Tongues
    function createTeardropTex() {
      var c = document.createElement("canvas");
      c.width = 128;
      c.height = 256;
      var g = c.getContext("2d");

      // Plasma tongue gradient
      var grad = g.createRadialGradient(64, 190, 2, 64, 150, 115);
      grad.addColorStop(0, "rgba(255, 255, 255, 1.0)");
      grad.addColorStop(0.25, "rgba(255, 255, 255, 0.9)");
      grad.addColorStop(0.55, "rgba(255, 255, 255, 0.35)");
      grad.addColorStop(0.85, "rgba(255, 255, 255, 0.08)");
      grad.addColorStop(1, "rgba(255, 255, 255, 0)");

      g.save();
      g.scale(0.82, 1.7);
      g.fillStyle = grad;
      g.beginPath();
      g.arc(78, 105, 68, 0, Math.PI * 2);
      g.fill();
      g.restore();

      var t = new T3.CanvasTexture(c);
      t.premultiplyAlpha = true;
      return t;
    }

    function createPuffTex() {
      var c = document.createElement("canvas");
      c.width = 128;
      c.height = 128;
      var g = c.getContext("2d");
      var gr = g.createRadialGradient(64, 64, 0, 64, 64, 64);
      gr.addColorStop(0, "rgba(255, 255, 255, 0.75)");
      gr.addColorStop(0.35, "rgba(255, 255, 255, 0.35)");
      gr.addColorStop(0.7, "rgba(255, 255, 255, 0.08)");
      gr.addColorStop(1, "rgba(255, 255, 255, 0)");
      g.fillStyle = gr;
      g.fillRect(0, 0, 128, 128);
      return new T3.CanvasTexture(c);
    }

    var plasmaTex = createTeardropTex();
    var puffTex = createPuffTex();

    // 3. Fluid Convective Licking Flame Sprites (Mantle swirl)
    var MANTLE_N = 50, mantleFlames = [];
    for (var mfi = 0; mfi < MANTLE_N; mfi++) {
      var mfs = new T3.Sprite(new T3.SpriteMaterial({
        map: plasmaTex,
        blending: T3.AdditiveBlending,
        depthWrite: false,
        transparent: true,
        fog: false
      }));
      hero.add(mfs);
      mantleFlames.push({
        s: mfs,
        age: Math.random(),
        spd: 0.55 + Math.random() * 0.6,
        ph: Math.random() * 6.28,
        r: 0.12 + Math.sqrt(Math.random()) * 0.38,
        a: Math.random() * 6.28
      });
    }

    // 4. Detaching Sparks & Ejected Rising Embers
    var SPARK_N = 45, sparks = [];
    for (var spi = 0; spi < SPARK_N; spi++) {
      var sps = new T3.Sprite(new T3.SpriteMaterial({
        map: puffTex,
        blending: T3.AdditiveBlending,
        depthWrite: false,
        transparent: true,
        fog: false
      }));
      hero.add(sps);
      sparks.push({
        s: sps,
        age: Math.random(),
        spd: 0.75 + Math.random() * 0.9,
        r: Math.random() * 0.45,
        a: Math.random() * 6.28,
        rotSpd: (Math.random() - 0.5) * 4.0
      });
    }

    // Dual-Frequency Dynamic Point Lights
    var mainFlameLight = new T3.PointLight(0x34d399, 2.6, 18);
    mainFlameLight.position.set(0, FLAME_Y + 0.9, 0.4);
    hero.add(mainFlameLight);

    var coalGlowLight = new T3.PointLight(0x059669, 1.8, 5);
    coalGlowLight.position.set(0, FLAME_Y + 0.1, 0);
    hero.add(coalGlowLight);

    // Large Ethereal Atmospheric Bloom Halo
    var halo = new T3.Sprite(new T3.SpriteMaterial({
      map: puffTex,
      blending: T3.AdditiveBlending,
      depthWrite: false,
      transparent: true,
      opacity: 0.28,
      fog: false
    }));
    halo.scale.set(9.5, 11.5, 1);
    halo.position.set(0, FLAME_Y + 1.2, -1.0);
    hero.add(halo);

    // Floor Specular Reflection Disc
    var floorGlowMat = new T3.MeshBasicMaterial({
      map: puffTex,
      color: 0x34d399,
      transparent: true,
      opacity: 0.38,
      blending: T3.AdditiveBlending,
      depthWrite: false
    });
    var floorGlow = new T3.Mesh(new T3.PlaneGeometry(6.5, 6.5), floorGlowMat);
    floorGlow.rotation.x = -Math.PI / 2;
    floorGlow.position.set(0, FLOOR + 0.02, 0);
    hero.add(floorGlow);

    /* ── Distant Atmospheric Temple Horizon Braziers ── */
    var EMIT = [
      [-4.0, -4.5], [4.2, -5.5], [-7.5, -8.0], [7.6, -9.0],
      [-4.2, -12.5], [4.4, -13.5], [-7.8, -16.0], [7.9, -17.5]
    ];
    var horizonEmitters = EMIT.map(function (p, i) {
      var sp = [];
      for (var k = 0; k < 6; k++) {
        var s = new T3.Sprite(new T3.SpriteMaterial({
          map: puffTex,
          blending: T3.AdditiveBlending,
          depthWrite: false,
          transparent: true,
          fog: true
        }));
        s.visible = false;
        scene.add(s);
        sp.push({
          s: s,
          age: Math.random(),
          spd: 0.4 + Math.random() * 0.45,
          ox: (Math.random() - 0.5) * 0.7,
          oz: (Math.random() - 0.5) * 0.5
        });
      }
      return { x: p[0], z: p[1], th: (i / EMIT.length) * 0.65, level: 0, sp: sp };
    });

    /* ── Floating Starlight Embers / Fireflies ── */
    var EN = 320, epos = new Float32Array(EN * 3), espd = new Float32Array(EN), ephase = new Float32Array(EN);
    for (var e = 0; e < EN; e++) {
      epos[e * 3] = (Math.random() - 0.5) * 16;
      epos[e * 3 + 1] = FLOOR + Math.random() * 8.8;
      epos[e * 3 + 2] = -12 + Math.random() * 18;
      espd[e] = 0.32 + Math.random() * 0.7;
      ephase[e] = Math.random() * Math.PI * 2;
    }
    var egeo = new T3.BufferGeometry();
    egeo.setAttribute("position", new T3.BufferAttribute(epos, 3));
    var emberMat = new T3.PointsMaterial({
      color: 0x6ee7b7,
      size: 0.055,
      transparent: true,
      opacity: 0.85,
      depthWrite: false,
      blending: T3.AdditiveBlending
    });
    scene.add(new T3.Points(egeo, emberMat));

    /* ── Responsive Viewport Sizing ── */
    var wide = true;
    function resize() {
      var w = stage.clientWidth, h = stage.clientHeight;
      if (!w || !h) return;
      renderer.setSize(w, h, false);
      cam.aspect = w / h;
      cam.updateProjectionMatrix();
      wide = cam.aspect > 1.25;
      if (wide) {
        cam.position.set(0, 0.95, 10);
        hero.position.set(Math.min(1.35 * cam.aspect, 4.4), 0, 0);
        hero.scale.setScalar(1);
      } else {
        cam.position.set(0, 0.95, 11.5);
        hero.position.set(0, -1.05, 0);
        hero.scale.setScalar(0.66);
      }
    }
    resize();

    var stageObserver = null;
    if (window.ResizeObserver) {
      stageObserver = new ResizeObserver(resize);
      stageObserver.observe(stage);
    } else {
      window.addEventListener("resize", resize);
    }

    var mx = 0, my = 0, tx = 0, ty = 0;
    function onPointerMove(e) {
      var r = stage.getBoundingClientRect();
      tx = ((e.clientX - r.left) / r.width) * 2 - 1;
      ty = ((e.clientY - r.top) / r.height) * 2 - 1;
    }
    function onPointerLeave() {
      tx = 0;
      ty = 0;
    }
    if (!reduce) {
      stage.addEventListener("pointermove", onPointerMove);
      stage.addEventListener("pointerleave", onPointerLeave);
    }

    var NUM = { safe: 0, elevated: 1, breached: 2 };
    var lv = NUM[current] != null ? NUM[current] : 0;
    var time = 0, clock = new T3.Clock();

    var coreColor = new T3.Color();
    var tipColor = new T3.Color();
    var glowColor = new T3.Color();
    var coalsColor = new T3.Color();
    var tmpC = new T3.Color();

    var animId = null;

    /* ── Render Loop ── */
    function frame() {
      var dt = Math.min(clock.getDelta(), 0.05);
      var targetLv = NUM[current] != null ? NUM[current] : 0;

      // Smooth damped exponential follower (zero abrupt pops)
      if (reduce) {
        dt = 0;
        lv = targetLv;
      } else {
        lv += (targetLv - lv) * (1 - Math.exp(-dt * 2.0));
        mx += (tx - mx) * 0.06;
        my += (ty - my) * 0.06;
      }
      time += dt;

      // Color interpolation along continuous palette
      mixN(P.skyTop, lv, skyU.top.value);
      mixN(P.skyHor, lv, skyU.hor.value);
      mixN(P.fog, lv, scene.fog.color);
      mixN(P.floor, lv, floorMat.color);
      mixN(P.flameCore, lv, coreColor);
      mixN(P.flameTip, lv, tipColor);
      mixN(P.flameGlow, lv, glowColor);
      mixN(P.coals, lv, coalsColor);

      // Star dimming as atmospheric haze increases
      starMat.opacity = mixV(0.9, 0.45, 0.15, lv);
      amb.intensity = mixV(1.2, 0.95, 0.72, lv);

      var flameHeightScale = mixV(1.0, 1.3, 1.72, lv);
      var turbulence = mixV(0.55, 1.05, 2.1, lv);
      var bgIntensity = lv <= 1 ? lv * 0.45 : 0.45 + (lv - 1) * 0.55;

      /* Camera & Altar idle drift */
      var camIdle = Math.sin(time * 0.35) * 0.05;
      altar.rotation.y = 0.72 + Math.sin(time * 0.3) * 0.25 + mx * 0.42;
      hero.rotation.x = -my * 0.032;
      cam.position.x = mx * 0.32;
      cam.position.y = 0.95 + camIdle - my * 0.16;
      cam.lookAt(hero.position.x * 0.26, 0.25, 0);

      /* Natural multi-frequency flame flicker */
      var flicker = 0.95 + Math.sin(time * 14.5) * 0.04 + Math.sin(time * 26.2) * 0.025 + Math.sin(time * 47.1) * 0.015;

      mainFlameLight.color.copy(glowColor);
      mainFlameLight.intensity = mixV(2.4, 3.4, 5.0, lv) * flicker;

      coalGlowLight.color.copy(coalsColor);
      coalGlowLight.intensity = mixV(1.6, 2.2, 3.2, lv) * flicker;

      rimLight.color.copy(glowColor);
      rimLight.intensity = mixV(2.5, 1.9, 1.5, lv);

      coalMat.emissive.copy(coalsColor);
      coalMat.emissiveIntensity = mixV(0.75, 1.1, 1.85, lv) * flicker;

      halo.material.color.copy(tipColor);
      halo.material.opacity = mixV(0.24, 0.35, 0.48, lv) * flicker;

      floorGlowMat.color.copy(tipColor);
      floorGlowMat.opacity = mixV(0.32, 0.42, 0.55, lv) * flicker;

      /* ── 3D Volumetric Flame Mesh Undulation ── */
      flameCoreMat.color.copy(coreColor);
      flameCoreMesh.scale.set(
        (0.95 + Math.sin(time * 8.5) * 0.08) * Math.sqrt(flameHeightScale),
        flameHeightScale * (0.96 + Math.cos(time * 11) * 0.06),
        (0.95 + Math.cos(time * 9.2) * 0.08) * Math.sqrt(flameHeightScale)
      );
      flameCoreMesh.rotation.y = time * 1.8;

      flameOuterMat.color.copy(tipColor);
      flameOuterMesh.scale.set(
        flameCoreMesh.scale.x * 1.35,
        flameCoreMesh.scale.y * 1.18,
        flameCoreMesh.scale.z * 1.35
      );
      flameOuterMesh.rotation.y = -time * 1.4;

      /* ── Mantle Swirling Plasma Sprites ── */
      for (var mi = 0; mi < MANTLE_N; mi++) {
        var mp = mantleFlames[mi];
        mp.age += dt * mp.spd * (0.8 + flameHeightScale * 0.25);
        if (mp.age > 1) {
          mp.age -= 1;
          mp.r = 0.1 + Math.sqrt(Math.random()) * 0.42;
          mp.a = Math.random() * 6.28;
        }
        var ma = mp.age;
        // Natural flame convection: narrower at base, swaying outward, pinched at apex
        var msway = Math.sin(time * 3.4 + mp.ph + ma * 6.5) * turbulence * 0.14 * ma;
        mp.s.position.set(
          Math.cos(mp.a) * mp.r * (1 - ma * 0.65) + msway,
          FLAME_Y + ma * 2.3 * flameHeightScale,
          Math.sin(mp.a) * mp.r * (1 - ma * 0.65) + Math.cos(time * 2.9 + mp.ph) * turbulence * 0.11 * ma
        );
        var msize = (0.32 + 1.2 * Math.pow(1 - ma, 0.85)) * Math.sqrt(flameHeightScale) * Math.min(1, ma * 9 + 0.3);
        mp.s.scale.set(msize * 0.82, msize * 1.5, 1);
        mp.s.material.opacity = 0.48 * Math.pow(1 - ma, 1.2) * flicker;
        mp.s.material.color.copy(coreColor).lerp(tipColor, Math.pow(ma, 0.65));
      }

      /* ── Rising Flame Sparks ── */
      for (var si2 = 0; si2 < SPARK_N; si2++) {
        var spk = sparks[si2];
        spk.age += dt * spk.spd * (0.7 + flameHeightScale * 0.35);
        if (spk.age > 1) {
          spk.age -= 1;
          spk.r = Math.random() * 0.48;
          spk.a = Math.random() * 6.28;
        }
        var sa2 = spk.age;
        var sSpiral = sa2 * 4.5 + spk.rotSpd;
        spk.s.position.set(
          Math.cos(spk.a + sSpiral) * (spk.r * (1 + sa2 * 1.2)),
          FLAME_Y + 0.4 + sa2 * 3.0 * flameHeightScale,
          Math.sin(spk.a + sSpiral) * (spk.r * (1 + sa2 * 1.2))
        );
        var spkSize = (0.09 + 0.16 * (1 - sa2)) * Math.sqrt(flameHeightScale);
        spk.s.scale.setScalar(spkSize);
        spk.s.material.opacity = 0.75 * Math.pow(1 - sa2, 0.85) * flicker;
        spk.s.material.color.copy(coreColor).lerp(tipColor, sa2);
      }

      /* ── REALISTIC COLONNADE DRUM DISPLACEMENT (No Fake Rigid Snapping) ── */
      // In classical architecture, columns under seismic strain experience drum shear-slip
      // along masonry seams rather than pivoting as a rigid wooden pole.
      var seismicStress = Math.pow(clamp01(lv * 0.5), 1.6);

      columnAssemblies.forEach(function (col) {
        var colSide = col.side;
        var colIdx = col.colIndex;

        // Subtle micro-tremor under Elevated
        var tremor = (lv > 0.4 && lv < 1.6) ? Math.sin(time * 26 + colIdx) * 0.002 * (lv - 0.4) : 0;

        // Each individual modular drum shears slightly along its bedding joint
        col.drums.forEach(function (drum) {
          var dIdx = drum.drumIndex;
          // Top drums displace progressively more than base drums
          var dDisplacement = seismicStress * (dIdx / (DRUMS_PER_COLUMN - 1));

          drum.mesh.position.x = drum.origX + drum.shearBiasX * dDisplacement + tremor;
          drum.mesh.position.z = drum.origZ + drum.shearBiasZ * dDisplacement;
          // Very subtle tilt angle as drums settle
          drum.mesh.rotation.z = (drum.shearBiasX * 0.15) * dDisplacement;
          drum.mesh.rotation.x = (drum.shearBiasZ * 0.15) * dDisplacement;
        });

        // Capital follows the topmost drum with slightly greater shift
        var topDrum = col.drums[col.drums.length - 1];
        col.capital.position.x = topDrum.mesh.position.x * 1.15;
        col.capital.position.z = topDrum.mesh.position.z * 1.15;
        col.capital.rotation.z = topDrum.mesh.rotation.z * 1.2;
      });

      // Architrave beams experience dignified settlement along fractures
      architraveBeams.forEach(function (beam) {
        var bDrop = Math.pow(seismicStress, 1.4) * 0.65;
        beam.group.position.y = beam.origY - bDrop;
        beam.group.rotation.z = beam.side * seismicStress * 0.035;
      });

      /* ── Distant Horizon Braziers ── */
      var hotHorizon = hot.set(0xffc870).lerp(tmpC.set(0xff8a40), seismicStress);
      var midHorizon = mid.set(0xe85a1a).lerp(tmpC.set(0xc41c14), seismicStress);
      for (var ei = 0; ei < horizonEmitters.length; ei++) {
        var em = horizonEmitters[ei];
        var tl = clamp01((bgIntensity - em.th) / 0.24) * (0.4 + 0.6 * bgIntensity);
        em.level += (tl - em.level) * (reduce ? 1 : Math.min(1, dt * 2.5));
        var on = em.level > 0.01;
        for (var k = 0; k < em.sp.length; k++) {
          var q = em.sp[k];
          q.s.visible = on;
          if (!on) continue;
          q.age += dt * q.spd;
          if (q.age > 1) q.age -= 1;
          var qa = q.age;
          q.s.position.set(
            em.x + q.ox + Math.sin(time * 2.8 + k + ei) * 0.12 * qa,
            FLOOR + 0.2 + qa * (1.3 + 2.8 * em.level),
            em.z + q.oz
          );
          q.s.scale.setScalar((0.55 + 1.6 * em.level) * (0.28 + 0.95 * Math.pow(1 - qa, 0.9)));
          q.s.material.opacity = 0.48 * Math.pow(1 - qa, 1.2) * Math.min(1, em.level * 2 + 0.15);
          q.s.material.color.copy(hotHorizon).lerp(midHorizon, Math.pow(qa, 0.6));
        }
      }
      horizonGlow.intensity = bgIntensity * 4.6 * (0.92 + Math.sin(time * 9) * 0.08);
      horizonGlow.color.set(0xff8a30).lerp(tmpC.set(0xff3a1c), seismicStress);

      /* ── Floating Starlight Embers ── */
      mixN(P.emberLight, lv, tmpC);
      emberMat.color.copy(tmpC);
      emberMat.size = mixV(0.055, 0.075, 0.11, lv);
      emberMat.opacity = mixV(0.8, 0.9, 0.98, lv);

      var espeed = mixV(0.18, 0.65, 1.5, lv);
      var edrift = mixV(0.002, 0.005, 0.012, lv);

      if (!reduce) {
        var pa = egeo.attributes.position.array;
        for (var j = 0; j < EN; j++) {
          pa[j * 3 + 1] += espd[j] * espeed * dt;
          pa[j * 3] += Math.sin(time * 1.2 + ephase[j]) * edrift;
          if (pa[j * 3 + 1] > FLOOR + 9.5) pa[j * 3 + 1] = FLOOR + 0.1;
          if (pa[j * 3] > 8.5) pa[j * 3] = -8.5;
          if (pa[j * 3] < -8.5) pa[j * 3] = 8.5;
        }
        egeo.attributes.position.needsUpdate = true;
      }

      renderer.render(scene, cam);
      animId = requestAnimationFrame(frame);
    }

    frame();

    return {
      setPosture: function (st) {
        current = st;
        stage.style.setProperty("--state", (STATES[current] || STATES.safe).css);
        if (flavorEl) flavorEl.textContent = FLAVOR[current] || FLAVOR.safe;
      },
      destroy: function () {
        if (animId) cancelAnimationFrame(animId);
        if (stageObserver) stageObserver.disconnect();
        window.removeEventListener("resize", resize);
        stage.removeEventListener("pointermove", onPointerMove);
        stage.removeEventListener("pointerleave", onPointerLeave);
        if (renderer) renderer.dispose();
      }
    };
  }
})();
