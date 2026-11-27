let tracker;
self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'init') {
      const { createHandTracker } = await import('./gesture.js');
      tracker = await createHandTracker();
      self.postMessage({ ready: true });
      return;
    }
    const result = tracker.detectForVideo(data.image, data.timestamp);
    self.postMessage({ points: result.landmarks[0], world: result.worldLandmarks[0] });
  } catch (error) {
    self.postMessage({ error: error.message || 'Hand inference failed' });
  } finally {
    data.image?.close();
  }
};
