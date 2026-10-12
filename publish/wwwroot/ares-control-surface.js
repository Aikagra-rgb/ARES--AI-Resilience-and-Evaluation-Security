(function () {
  var sceneInstance;

  window.AresControlSurface = {
    init: function (stageId, canvasId, initialPosture) {
      if (sceneInstance) sceneInstance.destroy();
      sceneInstance = createControlSurface(stageId, canvasId, initialPosture || "safe");
    },
    setPosture: function (posture) {
      if (sceneInstance) sceneInstance.setPosture(posture);
    },
    destroy: function () {
      if (sceneInstance) sceneInstance.destroy();
      sceneInstance = null;
    }
  };

  function createControlSurface(stageId, canvasId, initialPosture) {
    var stage = document.getElementById(stageId);
    var canvas = document.getElementById(canvasId);
    if (!stage || !canvas || !window.THREE) {
      if (stage) stage.classList.add("nogl");
      return { setPosture: function () {}, destroy: function () {} };
    }

    var T = window.THREE;
    var reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    var palette = {
      safe: { accent: 0x75e5d0, warm: 0xffa45e, label: "GUARDED" },
      elevated: { accent: 0xffc56b, warm: 0xff8d56, label: "ELEVATED" },
      breached: { accent: 0xff5d68, warm: 0xff724f, label: "UNDER ATTACK" }
    };
    var currentPosture = palette[initialPosture] ? initialPosture : "safe";
    var accentMaterials = [];
    var nodes = [];
    var interactiveMeshes = [];
    var root = new T.Group();
    var clock = new T.Clock();
    var frameId = 0;
    var resizeObserver;
    var destroyed = false;
    var hoverNode = null;
    var selectedNode = null;
    var pointerDown = null;
    var dragging = false;
    var pointer = new T.Vector2(2, 2);
    var targetRotation = 0.13;
    var targetTilt = 0;
    var currentRotation = targetRotation;
    var currentTilt = 0;
    var raycaster = new T.Raycaster();
    var accentColor = new T.Color();
    var warmColor = new T.Color();

    var scene = new T.Scene();
    scene.background = new T.Color(0x34383d);
    var camera = new T.PerspectiveCamera(36, 1, 0.1, 120);
    var renderer;
    try {
      renderer = new T.WebGLRenderer({ canvas: canvas, antialias: true, alpha: false, powerPreference: "high-performance" });
    } catch (_) {
      stage.classList.add("nogl");
      return { setPosture: function () {}, destroy: function () {} };
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.6));
    renderer.outputEncoding = T.sRGBEncoding;
    renderer.toneMapping = T.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;

    function standard(color, options) {
      return new T.MeshStandardMaterial(Object.assign({ color: color, roughness: 0.48, metalness: 0.34 }, options || {}));
    }
    function accentMaterial(color, intensity) {
      var mat = standard(color, { emissive: color, emissiveIntensity: intensity || 0.55, roughness: 0.3, metalness: 0.22 });
      accentMaterials.push(mat);
      return mat;
    }
    var dark = standard(0x11151a, { roughness: 0.32, metalness: 0.68 });
    var darkSide = standard(0x20262c, { roughness: 0.34, metalness: 0.58 });
    var silver = standard(0xe9f3f5, { roughness: 0.25, metalness: 0.5 });
    var accent = accentMaterial(palette[currentPosture].accent, 0.72);
    var warm = accentMaterial(palette[currentPosture].warm, 0.74);
    var screenMaterial = standard(0x101a1d, { emissive: 0x102b2e, emissiveIntensity: 0.65, roughness: 0.27, metalness: 0.25 });

    scene.add(new T.HemisphereLight(0xc9e7ef, 0x282126, 2.15));
    var key = new T.DirectionalLight(0xffffff, 3.2);
    key.position.set(-5, 12, 8);
    scene.add(key);
    var cyanLight = new T.PointLight(palette[currentPosture].accent, 28, 18, 2);
    cyanLight.position.set(-4, 5, 0);
    scene.add(cyanLight);
    var amberLight = new T.PointLight(palette[currentPosture].warm, 24, 17, 2);
    amberLight.position.set(4, 5, 1);
    scene.add(amberLight);
    scene.add(root);

    function addMesh(parent, geometry, material, x, y, z, node) {
      var mesh = new T.Mesh(geometry, material);
      mesh.position.set(x || 0, y || 0, z || 0);
      if (node) {
        mesh.userData.controlNode = node;
      }
      parent.add(mesh);
      return mesh;
    }
    function box(parent, w, h, d, material, x, y, z, node, bevel) {
      var mesh = addMesh(parent, new T.BoxGeometry(w, h, d, 1, 1, 1), material, x, y, z, node);
      if (bevel) {
        var edges = new T.LineSegments(new T.EdgesGeometry(mesh.geometry), new T.LineBasicMaterial({ color: 0x75828b, transparent: true, opacity: 0.5 }));
        edges.position.copy(mesh.position);
        parent.add(edges);
      }
      return mesh;
    }
    function cylinder(parent, rt, rb, h, material, x, y, z, node, sides) {
      return addMesh(parent, new T.CylinderGeometry(rt, rb, h, sides || 32, 1), material, x, y, z, node);
    }
    function torus(parent, radius, tube, material, x, y, z, horizontal) {
      var mesh = addMesh(parent, new T.TorusGeometry(radius, tube, 8, 100), material, x, y, z);
      if (horizontal !== false) mesh.rotation.x = Math.PI / 2;
      return mesh;
    }
    function makeNode(name, kind, x, z) {
      var group = new T.Group();
      group.position.set(x, 0, z);
      root.add(group);
      var node = { name: name, kind: kind, group: group, accentParts: [], phase: Math.random() * 6.28, baseY: 0 };
      nodes.push(node);
      group.traverse(function (obj) { if (obj.isMesh) obj.userData.controlNode = node; });
      return node;
    }
    function registerNodeMeshes(node) {
      node.group.traverse(function (obj) { if (obj.isMesh) interactiveMeshes.push(obj); });
    }

    // A dark, satin floor makes the copper circuit paths read like a live network map.
    var floor = addMesh(root, new T.PlaneGeometry(42, 34), standard(0x34383d, { roughness: 0.85, metalness: 0.08 }), 0, -0.12, 0);
    floor.rotation.x = -Math.PI / 2;
    var grid = new T.GridHelper(32, 32, 0x62666a, 0x414449);
    grid.position.y = -0.105;
    grid.material.transparent = true;
    grid.material.opacity = 0.2;
    root.add(grid);

    // Small concentric command dais under the central shield.
    cylinder(root, 1.58, 1.7, 0.28, dark, 0, 0.12, 0, null, 64);
    cylinder(root, 1.34, 1.48, 0.12, silver, 0, 0.31, 0, null, 64);
    cylinder(root, 1.27, 1.31, 0.18, darkSide, 0, 0.43, 0, null, 64);
    torus(root, 1.38, 0.055, accent, 0, 0.39, 0);
    torus(root, 1.05, 0.025, warm, 0, 0.54, 0);
    var hubLight = new T.PointLight(palette[currentPosture].accent, 15, 7, 2);
    hubLight.position.set(0, 1.5, 0);
    root.add(hubLight);

    // Raised shield silhouette, layered face, and check mark.
    function shieldShape(width, height, inset) {
      var w = width / 2;
      var h = height / 2;
      var s = new T.Shape();
      s.moveTo(-w + inset, h);
      s.lineTo(w - inset, h);
      s.lineTo(w, h - 0.13);
      s.lineTo(w * 0.86, -h * 0.12);
      s.quadraticCurveTo(w * 0.62, -h * 0.72, 0, -h);
      s.quadraticCurveTo(-w * 0.62, -h * 0.72, -w * 0.86, -h * 0.12);
      s.lineTo(-w, h - 0.13);
      s.closePath();
      return s;
    }
    var shield = new T.Group();
    shield.position.set(0, 2.55, 0.1);
    root.add(shield);
    addMesh(shield, new T.ExtrudeGeometry(shieldShape(1.72, 2.12, 0.12), { depth: 0.22, bevelEnabled: true, bevelSegments: 3, steps: 1, bevelSize: 0.045, bevelThickness: 0.045 }), silver, 0, 0, 0);
    addMesh(shield, new T.ExtrudeGeometry(shieldShape(1.48, 1.88, 0.1), { depth: 0.08, bevelEnabled: true, bevelSegments: 2, bevelSize: 0.025, bevelThickness: 0.025 }), dark, 0, 0.02, 0.245);
    var shieldRim = addMesh(shield, new T.ExtrudeGeometry(shieldShape(1.31, 1.69, 0.085), { depth: 0.035, bevelEnabled: false }), accent, 0, 0.005, 0.34);
    shieldRim.scale.set(0.96, 0.96, 1);
    var checkGeo = new T.BufferGeometry().setFromPoints([
      new T.Vector3(-0.32, -0.02, 0.405), new T.Vector3(-0.08, -0.28, 0.405), new T.Vector3(0.38, 0.25, 0.405)
    ]);
    var check = new T.Line(checkGeo, new T.LineBasicMaterial({ color: 0xffa45e, linewidth: 4 }));
    shield.add(check);
    var shieldGlow = new T.PointLight(palette[currentPosture].accent, 16, 5, 2);
    shieldGlow.position.set(0, 0.15, 0.8);
    shield.add(shieldGlow);

    // Build the glowing spoke/circuit map before the devices so wires stay beneath them.
    var endpoints = [
      { x: -4.3, z: -2.4 }, { x: -3.0, z: -4.0 }, { x: 3.6, z: -3.3 }, { x: 4.8, z: -1.9 },
      { x: -4.6, z: 2.2 }, { x: -3.35, z: 3.25 }, { x: 3.5, z: 3.45 }, { x: 4.8, z: 2.5 }
    ];
    var traces = [];
    endpoints.forEach(function (p, i) {
      var midX = p.x * 0.48;
      var midZ = p.z * 0.43;
      var points = [new T.Vector3(0, -0.015, 0), new T.Vector3(midX, -0.015, midZ), new T.Vector3(p.x, -0.015, p.z)];
      var line = new T.Line(new T.BufferGeometry().setFromPoints(points), new T.LineBasicMaterial({ color: 0xc58b59, transparent: true, opacity: 0.75 }));
      root.add(line);
      var pulse = addMesh(root, new T.SphereGeometry(0.055, 10, 8), accent, 0, 0.025, 0);
      traces.push({ pulse: pulse, points: points, phase: i / endpoints.length });
      // Branch ticks make the routes feel like board traces rather than spokes.
      var bx = midX + (p.x < 0 ? 0.34 : -0.34);
      var branch = new T.Line(new T.BufferGeometry().setFromPoints([
        new T.Vector3(midX, -0.02, midZ), new T.Vector3(bx, -0.02, midZ), new T.Vector3(bx, -0.02, midZ + (p.z < 0 ? 0.28 : -0.28))
      ]), new T.LineBasicMaterial({ color: 0xffa45e, transparent: true, opacity: 0.55 }));
      root.add(branch);
    });

    // Two upright application servers with illuminated caps and front telemetry.
    function serverNode(name, x, z, height, capMat) {
      var node = makeNode(name, "server", x, z);
      var y = height / 2 + 0.25;
      cylinder(node.group, 0.48, 0.53, 0.16, dark, 0, 0.12, 0, node, 8);
      box(node.group, 0.88, height, 0.86, dark, 0, y, 0, node, true);
      box(node.group, 0.72, height - 0.38, 0.035, screenMaterial, 0, y + 0.03, 0.45, node);
      box(node.group, 0.92, 0.12, 0.92, silver, 0, height + 0.31, 0, node, true);
      box(node.group, 0.65, 0.14, 0.64, capMat, 0, height + 0.43, 0, node, true);
      for (var i = 0; i < 5; i++) {
        box(node.group, 0.055, 0.08, 0.035, i % 2 ? warm : accent, -0.2 + (i % 2) * 0.18, 0.55 + i * 0.17, 0.485, node);
      }
      box(node.group, 0.76, 0.09, 0.9, warm, 0, 0.35, 0, node);
      node.floatMesh = node.group;
      registerNodeMeshes(node);
      return node;
    }
    serverNode("APPLICATION SERVER 01", -4.3, -2.4, 1.85, accent);
    serverNode("APPLICATION SERVER 02", -3.0, -4.0, 2.25, silver);

    // Model / policy controls are stacked low-poly compute tiles.
    function computeNode(name, x, z, count, accentTop) {
      var node = makeNode(name, "compute", x, z);
      cylinder(node.group, 0.7, 0.78, 0.2, dark, 0, 0.12, 0, node, 8);
      for (var i = 0; i < count; i++) {
        var y = 0.36 + i * 0.48;
        box(node.group, 1.16, 0.36, 1.05, darkSide, 0, y, 0, node, true);
        box(node.group, 0.82, 0.035, 0.78, accentTop ? accent : warm, 0, y + 0.2, 0, node);
        box(node.group, 0.76, 0.055, 0.04, i % 2 ? warm : accent, 0, y - 0.04, 0.54, node);
        box(node.group, 0.48, 0.035, 0.045, silver, 0, y - 0.04, 0.57, node);
      }
      node.floatMesh = node.group;
      registerNodeMeshes(node);
      return node;
    }
    computeNode("POLICY ENGINE 01", 3.6, -3.3, 2, false);
    computeNode("MODEL GATEWAY", 4.8, -1.9, 2, true);
    computeNode("RUNTIME GUARD", 3.5, 3.45, 2, true);
    computeNode("AUDIT CONTROLS", 4.8, 2.5, 2, false);

    // Cylindrical vector stores and telemetry archives.
    function storageNode(name, x, z, height) {
      var node = makeNode(name, "storage", x, z);
      cylinder(node.group, 0.56, 0.62, 0.19, dark, 0, 0.12, 0, node, 32);
      cylinder(node.group, 0.5, 0.55, height, darkSide, 0, height / 2 + 0.22, 0, node, 32);
      for (var i = 0; i < 3; i++) torus(node.group, 0.515, 0.045, i === 1 ? accent : warm, 0, 0.43 + i * (height / 3), 0);
      cylinder(node.group, 0.5, 0.5, 0.11, warm, 0, height + 0.24, 0, node, 32);
      node.floatMesh = node.group;
      registerNodeMeshes(node);
      return node;
    }
    storageNode("VECTOR STORE 01", -4.6, 2.2, 0.72);
    storageNode("TELEMETRY ARCHIVE", -3.35, 3.25, 0.9);

    // Orbiting particles move data outward from the shield to each control node.
    var pulseMeshes = traces.map(function (trace) { return trace.pulse; });
    function applyPosture(posture) {
      if (!palette[posture]) return;
      currentPosture = posture;
      var p = palette[posture];
      var flavor = document.getElementById("flavor");
      var health = document.getElementById("component-health");
      stage.style.setProperty("--state", "#" + p.accent.toString(16).padStart(6, "0"));
      if (flavor) flavor.textContent = posture === "safe" ? "The control network is holding. All routes are protected." : posture === "elevated" ? "Elevated activity detected. Inspect the highlighted controls." : "Threat activity is breaching controls. Review affected systems.";
      if (health) health.textContent = selectedNode ? p.label : "ALL SYSTEMS " + p.label;
      if (posture === "breached") {
        stage.style.borderColor = "rgba(255,93,104,.58)";
      } else if (posture === "elevated") {
        stage.style.borderColor = "rgba(255,197,107,.52)";
      } else {
        stage.style.borderColor = "";
      }
    }

    function resize() {
      if (destroyed) return;
      var w = stage.clientWidth;
      var h = stage.clientHeight;
      if (!w || !h) return;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.fov = w < 620 ? 43 : 36;
      camera.position.set(w < 620 ? 0 : 0.6, w < 620 ? 13 : 11, w < 620 ? 22 : 18.5);
      camera.lookAt(0.5, 0.8, 0);
      camera.updateProjectionMatrix();
      renderOnce();
    }
    if (window.ResizeObserver) {
      resizeObserver = new ResizeObserver(resize);
      resizeObserver.observe(stage);
    } else {
      window.addEventListener("resize", resize);
    }
    resize();
    applyPosture(currentPosture);

    function setPointer(event) {
      var rect = canvas.getBoundingClientRect();
      pointer.set(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1);
      raycaster.setFromCamera(pointer, camera);
      var hit = raycaster.intersectObjects(interactiveMeshes, false)[0];
      return hit && hit.object.userData.controlNode ? hit.object.userData.controlNode : null;
    }
    function ignoreControlTarget(target) {
      return target && target.closest && target.closest("button, a, .stage-copy");
    }
    function onPointerDown(event) {
      if (event.button !== 0 || ignoreControlTarget(event.target)) return;
      pointerDown = { x: event.clientX, y: event.clientY, rotation: targetRotation, tilt: targetTilt, id: event.pointerId };
      dragging = false;
      canvas.setPointerCapture(event.pointerId);
    }
    function onPointerMove(event) {
      if (pointerDown && pointerDown.id === event.pointerId) {
        var dx = event.clientX - pointerDown.x;
        var dy = event.clientY - pointerDown.y;
        if (Math.abs(dx) + Math.abs(dy) > 5) dragging = true;
        if (dragging) {
          targetRotation = pointerDown.rotation + dx * 0.008;
          targetTilt = Math.max(-0.12, Math.min(0.12, pointerDown.tilt + dy * 0.002));
          if (reduced) renderOnce();
          return;
        }
      }
      var nextHover = setPointer(event);
      if (nextHover !== hoverNode) {
        hoverNode = nextHover;
        canvas.style.cursor = hoverNode ? "pointer" : pointerDown ? "grabbing" : "grab";
        if (reduced) renderOnce();
      }
    }
    function onPointerUp(event) {
      if (!pointerDown || pointerDown.id !== event.pointerId) return;
      if (!dragging) {
        var picked = setPointer(event);
        if (picked) {
          selectedNode = picked;
          var name = document.getElementById("component-name");
          var health = document.getElementById("component-health");
          if (name) name.textContent = picked.name;
          if (health) health.textContent = palette[currentPosture].label;
        } else {
          selectedNode = null;
          var nameReset = document.getElementById("component-name");
          var healthReset = document.getElementById("component-health");
          if (nameReset) nameReset.textContent = "ARES NETWORK";
          if (healthReset) healthReset.textContent = "ALL SYSTEMS " + palette[currentPosture].label;
        }
      }
      pointerDown = null;
      dragging = false;
      if (reduced) renderOnce();
    }
    function onPointerLeave() {
      if (!pointerDown) {
        hoverNode = null;
        if (reduced) renderOnce();
      }
    }
    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerup", onPointerUp);
    canvas.addEventListener("pointercancel", onPointerUp);
    canvas.addEventListener("pointerleave", onPointerLeave);

    var elapsed = 0;
    function renderOnce() {
      if (destroyed) return;
      var p = palette[currentPosture];
      accentColor.setHex(p.accent);
      warmColor.setHex(p.warm);
      accentMaterials.forEach(function (mat) {
        var isWarm = mat === warm;
        var color = isWarm ? warmColor : accentColor;
        mat.color.copy(color);
        mat.emissive.copy(color);
        if (mat === accent && (hoverNode || selectedNode)) mat.emissiveIntensity = 1.25;
      });
      renderer.render(scene, camera);
    }
    function animate() {
      if (destroyed) return;
      var dt = Math.min(clock.getDelta(), 0.05);
      elapsed += dt;
      currentRotation += (targetRotation - currentRotation) * (1 - Math.exp(-dt * 4));
      currentTilt += (targetTilt - currentTilt) * (1 - Math.exp(-dt * 4));
      root.rotation.y = currentRotation + (reduced ? 0 : Math.sin(elapsed * 0.22) * 0.035);
      root.rotation.x = currentTilt + (reduced ? 0 : Math.sin(elapsed * 0.27) * 0.012);
      shield.rotation.y = reduced ? 0 : Math.sin(elapsed * 0.65) * 0.12;
      shield.position.y = 2.55 + (reduced ? 0 : Math.sin(elapsed * 1.35) * 0.055);
      var pulseColor = palette[currentPosture].accent;
      hubLight.color.setHex(pulseColor);
      shieldGlow.color.setHex(pulseColor);
      cyanLight.color.setHex(pulseColor);
      amberLight.color.setHex(palette[currentPosture].warm);
      hubLight.intensity = 13 + (reduced ? 0 : Math.sin(elapsed * 2.2) * 2.2);
      shieldGlow.intensity = 14 + (reduced ? 0 : Math.sin(elapsed * 2.6) * 3.5);
      nodes.forEach(function (node) {
        if (node.floatMesh && !reduced) node.group.position.y = Math.sin(elapsed * 1.1 + node.phase) * 0.035;
      });
      traces.forEach(function (trace) {
        var t = (elapsed * 0.22 + trace.phase) % 1;
        var a = trace.points[0], b = trace.points[1], c = trace.points[2];
        var pos = t < 0.48 ? a.clone().lerp(b, t / 0.48) : b.clone().lerp(c, (t - 0.48) / 0.52);
        trace.pulse.position.set(pos.x, 0.045, pos.z);
        trace.pulse.scale.setScalar(hoverNode && hoverNode.name ? 1.25 : 1);
      });
      accentMaterials.forEach(function (mat) {
        if (mat !== accent && mat !== warm) return;
        mat.emissiveIntensity = (hoverNode || selectedNode) ? 1.05 : 0.58 + (reduced ? 0 : (Math.sin(elapsed * 2.1) + 1) * 0.1);
      });
      renderer.render(scene, camera);
      if (!reduced) frameId = requestAnimationFrame(animate);
    }

    if (reduced) renderOnce();
    else animate();

    return {
      setPosture: function (posture) {
        if (!palette[posture]) return;
        applyPosture(posture);
        renderOnce();
      },
      destroy: function () {
        destroyed = true;
        if (frameId) cancelAnimationFrame(frameId);
        if (resizeObserver) resizeObserver.disconnect();
        else window.removeEventListener("resize", resize);
        canvas.removeEventListener("pointerdown", onPointerDown);
        canvas.removeEventListener("pointermove", onPointerMove);
        canvas.removeEventListener("pointerup", onPointerUp);
        canvas.removeEventListener("pointercancel", onPointerUp);
        canvas.removeEventListener("pointerleave", onPointerLeave);
        scene.traverse(function (obj) {
          if (obj.geometry) obj.geometry.dispose();
          if (obj.material) {
            if (Array.isArray(obj.material)) obj.material.forEach(function (mat) { mat.dispose(); });
            else obj.material.dispose();
          }
        });
        renderer.dispose();
      }
    };
  }
})();
