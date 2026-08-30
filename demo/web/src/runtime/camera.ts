// Web カメラ: getUserMedia とデバイス列挙(エラーは読める文言に変換)。

export interface CameraDevice {
  deviceId: string;
  label: string;
}

export async function listCameras(): Promise<CameraDevice[]> {
  if (!navigator.mediaDevices?.enumerateDevices) {
    return [];
  }
  const devices = await navigator.mediaDevices.enumerateDevices();
  return devices
    .filter((d) => d.kind === 'videoinput')
    .map((d, i) => ({ deviceId: d.deviceId, label: d.label || `Camera ${i + 1}` }));
}

export async function openCamera(deviceId: string | null): Promise<MediaStream> {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('getUserMedia is unavailable in this browser.');
  }
  // カメラ入力は VGA(640×480)に固定する(ユーザー指定 2026-08-30)。検出器の入力は 640×640 への直リサイズなので
  // それ以上の解像度はフレーム取得・転送コストにしかならない
  const constraints: MediaStreamConstraints = {
    audio: false,
    video: {
      width: { ideal: 640 },
      height: { ideal: 480 },
      ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
    },
  };
  try {
    return await navigator.mediaDevices.getUserMedia(constraints);
  } catch (error) {
    const name = error instanceof DOMException ? error.name : '';
    if (name === 'NotFoundError' || name === 'OverconstrainedError') {
      throw new Error('No webcam found. Check the connection or use the video / image file input.');
    }
    if (name === 'NotAllowedError') {
      throw new Error('Camera access was denied.');
    }
    if (name === 'NotReadableError') {
      throw new Error('Cannot open the camera (it may be in use by another app).');
    }
    throw error instanceof Error ? error : new Error(String(error));
  }
}

export function stopStream(stream: MediaStream | null): void {
  if (stream) {
    for (const track of stream.getTracks()) {
      track.stop();
    }
  }
}
