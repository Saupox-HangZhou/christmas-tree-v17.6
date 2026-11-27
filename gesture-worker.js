let tracker;
self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'init') {
      const { createHandTracker } = await import('./gesture.js?v=yyj-galaxy-1');
      tracker = await createHandTracker();
      self.postMessage({ ready: true });
      return;
    }
    const result = tracker.detectForVideo(data.image, data.timestamp);
    self.postMessage({ points: result.landmarks, world: result.worldLandmarks });
  } catch (error) {
    self.postMessage({ error: error.message || 'Hand inference failed' });
  } finally {
    data.image?.close();
  }
};
