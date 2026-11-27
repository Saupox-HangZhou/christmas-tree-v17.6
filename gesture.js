// The hand tracker runs on the camera clock, independently of the particle animation.
const visionBase = new URL('./assets/vision/', import.meta.url);

export async function createHandTracker() {
  const { FilesetResolver, HandLandmarker } = await import(new URL('vision_bundle.mjs', visionBase).href);
  const files = await FilesetResolver.forVisionTasks(new URL('wasm', visionBase).href);
  const modelAssetPath = new URL('hand_landmarker.task', visionBase).href;
  const options = {
    runningMode: 'VIDEO',
    numHands: 1,
    minHandDetectionConfidence: 0.55,
    minHandPresenceConfidence: 0.55,
    minTrackingConfidence: 0.5
  };
  try {
    return await HandLandmarker.createFromOptions(files, { ...options, baseOptions: { modelAssetPath, delegate: 'GPU' } });
  } catch {
    return HandLandmarker.createFromOptions(files, { ...options, baseOptions: { modelAssetPath, delegate: 'CPU' } });
  }
}

function vector(a, b) {
  return [b.x - a.x, b.y - a.y, b.z - a.z];
}
function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
function length(v) {
  return Math.hypot(...v);
}
function alignment(a, b) {
  return dot(a, b) / Math.max(1e-8, length(a) * length(b));
}

// Metric 3D coordinates make the tests independent of handedness and image rotation.
// Wider release thresholds reject jitter without holding a new gesture for extra frames.
export function classifyFingers(points, previous = []) {
  const handWidth = length(vector(points[5], points[17]));
  const distance = (a, b) => length(vector(points[a], points[b])) / Math.max(1e-6, handWidth);
  const acrossPalm = vector(points[17], points[5]);
  const thumbTip = vector(points[5], points[4]);
  const thumbProjection = dot(thumbTip, acrossPalm) / Math.max(1e-8, handWidth * handWidth);
  const thumbStraight = alignment(vector(points[2], points[3]), vector(points[3], points[4]));
  const thumbOpen = thumbStraight > (previous[0] ? 0.45 : 0.60) &&
    (thumbProjection > (previous[0] ? 0.26 : 0.34) ||
      (distance(4, 17) > (previous[0] ? 1.19 : 1.29) && distance(4, 9) > 0.83));
  const flags = [thumbOpen];
  for (const base of [5, 9, 13, 17]) {
    const wasOpen = previous[flags.length];
    const proximal = vector(points[base], points[base + 1]);
    const distal = vector(points[base + 1], points[base + 3]);
    const straight = alignment(proximal, distal);
    const reach = distance(0, base + 3) - distance(0, base + 1);
    flags.push(straight > (wasOpen ? 0.39 : 0.53) && reach > (wasOpen ? 0.06 : 0.12));
  }
  return flags;
}

// A classic worker is needed because the MediaPipe WASM loader uses importScripts.
async function openWorker(signal) {
  const worker = new Worker(new URL('./gesture-worker.js', import.meta.url));
  let pending;
  const wait = () => new Promise((resolve, reject) => { pending = { resolve, reject }; });
  const stop = () => {
    worker.terminate();
    pending?.reject(new Error('Tracker stopped'));
    signal.removeEventListener('abort', stop);
  };
  signal.addEventListener('abort', stop, { once: true });
  worker.onerror = event => pending?.reject(new Error(event.message || 'Worker error'));
  worker.onmessage = ({ data }) => {
    if (data.error) pending?.reject(new Error(data.error));
    else pending?.resolve(data);
  };
  try {
    const ready = wait();
    worker.postMessage({ type: 'init' });
    await ready;
    return {
      async detect(video, timestamp) {
        const image = await createImageBitmap(video);
        if (signal.aborted) { image.close(); return {}; }
        const response = wait();
        try {
          worker.postMessage({ image, timestamp }, [image]);
        } catch (error) {
          image.close();
          throw error;
        }
        return response;
      },
      stop
    };
  } catch (error) {
    stop();
    throw error;
  }
}

async function openMainTracker(signal) {
  const tracker = await createHandTracker();
  if (signal.aborted) { tracker.close(); throw new Error('Tracker stopped'); }
  return {
    detect(video, timestamp) {
      const result = tracker.detectForVideo(video, timestamp);
      return { points: result.landmarks[0], world: result.worldLandmarks[0] };
    },
    stop: () => tracker.close()
  };
}

export async function startGestureCamera(video, onCount, onStatus, onLandmarks = () => {}, externalSignal) {
  const controller = new AbortController();
  const signal = controller.signal;
  let stream, engine, handle;
  let busy = false, previous = [], lastCount, lastTime = -1;
  let workerActive = false;
  const videoClock = typeof video.requestVideoFrameCallback === 'function';
  const stop = () => {
    if (signal.aborted) return;
    controller.abort();
    if (handle !== undefined) {
      if (videoClock) video.cancelVideoFrameCallback(handle);
      else cancelAnimationFrame(handle);
    }
    externalSignal?.removeEventListener('abort', stop);
    window.removeEventListener('pagehide', stop);
    engine?.stop();
    stream?.getTracks().forEach(track => track.stop());
    if (stream && video.srcObject === stream) video.srcObject = null;
    if (lastCount != null) onCount(null);
    onLandmarks([]);
  };
  const fail = status => { if (!signal.aborted) { stop(); onStatus(status); } };
  if (externalSignal?.aborted) return stop;
  externalSignal?.addEventListener('abort', stop, { once: true });
  window.addEventListener('pagehide', stop, { once: true });
  if (!navigator.mediaDevices?.getUserMedia) { fail('no-device'); return stop; }
  onStatus('starting');
  async function openCamera() {
    const source = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 60 } }
    });
    if (signal.aborted) { source.getTracks().forEach(track => track.stop()); return; }
    stream = source;
    stream.getVideoTracks()[0].addEventListener('ended', () => fail('no-device'), { once: true });
    video.muted = true;
    video.playsInline = true;
    video.srcObject = stream;
    await video.play();
  }
  async function loadEngine() {
    if (typeof Worker === 'function' && typeof createImageBitmap === 'function' && typeof OffscreenCanvas === 'function') {
      try {
        engine = await openWorker(signal);
        workerActive = true;
        return;
      } catch { if (signal.aborted) return; }
    }
    engine = await openMainTracker(signal);
  }
  function schedule() {
    handle = videoClock ? video.requestVideoFrameCallback(onFrame) : requestAnimationFrame(onFrame);
  }
  async function onFrame(now) {
    if (signal.aborted) return;
    schedule();
    if (busy || document.hidden || video.readyState < 2 || lastTime === video.currentTime) return;
    lastTime = video.currentTime;
    busy = true;
    try {
      const result = await engine.detect(video, now);
      if (signal.aborted) return;
      const points = result.points || [];
      const world = result.world;
      previous = world?.length === 21 ? classifyFingers(world, previous) : [];
      const count = previous.length ? previous.filter(Boolean).length : null;
      if (count !== lastCount) { lastCount = count; onCount(count); }
      onLandmarks(points);
    } catch {
      if (!signal.aborted && workerActive) {
        workerActive = false;
        engine.stop();
        try { engine = await openMainTracker(signal); } catch { fail('model-error'); }
      } else { fail('model-error'); }
    } finally { busy = false; }
  }
  try {
    await Promise.all([
      openCamera().catch(e => {
        const status = { NotAllowedError: 'denied', SecurityError: 'denied', NotFoundError: 'no-device', DevicesNotFoundError: 'no-device' };
        throw new Error(status[e.name] || 'error');
      }),
      loadEngine().catch(() => { throw new Error('model-error'); })
    ]);
    if (!signal.aborted) { onStatus('live'); schedule(); }
  } catch (error) { fail(error.message); }
  return stop;
}
