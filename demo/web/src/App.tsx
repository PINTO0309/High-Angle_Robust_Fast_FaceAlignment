import { useCallback, useEffect, useRef, useState } from 'react';
import { listCameras, openCamera, stopStream, type CameraDevice } from './runtime/camera';
import { activeWorkerMode, type Accelerator, type OrtModel } from './runtime/engine';
import { loadOrtModel } from './runtime/ort';
import { WorkerPipeline } from './runtime/workerClient';
import { FaceAligner } from './hrffa/aligner';
import { inferInputNorm, type InputNorm } from './hrffa/constants';
import { HeadDetector } from './hrffa/detector';
import { drawHeads, type DrawOptions } from './hrffa/draw';
import { HrffaPipeline } from './hrffa/pipeline';
import type { FrameOutput, FrameSource } from './hrffa/types';
import type { WorkerReadyInfo } from './workers/inference.worker';

interface ModelEntry {
  name: string;
  bytes: number;
  url: string;
}

type SourceKind = 'camera' | 'video' | 'image';
type InputNormChoice = 'auto' | InputNorm;

interface RunStats {
  fps: number;
  totalMs: number;
  detectMs: number;
  alignMs: number;
  nHeads: number;
  frame: number;
}

// 既定は MIT ライセンス版 YOLOv9-Wholebody34(t)の生 head export(8.2 MB、WebGPU 約 22 ms/run。ユーザー指定 2026-08-30)
const DEFAULT_DETECTOR = 'yolov9_t_wholebody34_0100_1x3x640x640.onnx';
const DEFAULT_ALIGNER = 'hrffa_vitt_ibug68_1x3x256x256.onnx';
const WEBGPU_OK = typeof navigator !== 'undefined' && 'gpu' in navigator;
// URL クエリ: ?backend=wasm|webgpu、?worker=dedicated|main、?detector=<file>、?aligner=<file>、
// ?image=<同一オリジンの画像 URL>、?video=<同一オリジンの動画 URL>、?autostart=1
const QUERY = new URLSearchParams(window.location.search);
// 推論の実行場所はページ読み込み時に固定(Electron は --web-inference-worker を ?worker= に写す)
const WORKER_MODE = activeWorkerMode();

// 検出器は boxes-only 版のみ列挙する(masks 出力付きの export はこのデモでは使わない。ユーザー指定 2026-08-30)
const isDetector = (m: ModelEntry): boolean => /^deimv2_.*boxes_only/i.test(m.name) || /^yolov9_/i.test(m.name);
const isAligner = (m: ModelEntry): boolean => /^hrffa_/i.test(m.name);

function pickDefault(list: ModelEntry[], preferred: string): string {
  return list.find((m) => m.name === preferred)?.name ?? list[0]?.name ?? '';
}

function formatMb(bytes: number): string {
  return bytes > 0 ? ` (${(bytes / 1e6).toFixed(1)} MB)` : '';
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchBytes(url: string): Promise<Uint8Array> {
  const response = await fetch(new URL(url, document.baseURI).href, { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`Failed to fetch model: ${response.status} ${response.statusText} (${url})`);
  }
  return new Uint8Array(await response.arrayBuffer());
}

async function loadImageBitmap(url: string): Promise<ImageBitmap> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load image: ${response.status} ${response.statusText}`);
  }
  return createImageBitmap(await response.blob());
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function sourceFrom(image: CanvasImageSource, width: number, height: number): FrameSource {
  return {
    width,
    height,
    draw: (ctx, w, h) => ctx.drawImage(image, 0, 0, w, h),
    drawCrop: (ctx, sx, sy, sw, sh, dw, dh) => ctx.drawImage(image, sx, sy, sw, sh, 0, 0, dw, dh),
  };
}

// 自動テスト・デバッグ用: 最新の結果を window に置く
function publishResult(output: FrameOutput): void {
  (window as unknown as { __hrffaLast?: FrameOutput }).__hrffaLast = output;
}

// 1 フレーム分の推論をモードの違いから切り離す
interface Runner {
  info: WorkerReadyInfo;
  process(source: FrameSource, grab: () => ImageData): Promise<FrameOutput>;
  setParams(params: { headScoreThreshold?: number; cropPad?: number }): Promise<void>;
  dispose(): void | Promise<void>;
}

// ---- UI スレッド実行(?worker=main): Worker と同じクラスをメインスレッドで動かす
async function createLocalRunner(opts: {
  accelerator: Accelerator;
  numThreads: number;
  detectorUrl: string;
  alignerUrl: string;
  headScoreThreshold: number;
  cropPad: number;
  inputNorm: InputNorm;
}): Promise<Runner> {
  let accel: Accelerator = opts.accelerator;
  let note: string | null = null;
  const engines: OrtModel[] = [];
  const loadWithFallback = async (bytes: Uint8Array): Promise<OrtModel> => {
    try {
      return await loadOrtModel(bytes, accel, opts.numThreads);
    } catch (error) {
      if (accel === 'webgpu') {
        note = `WebGPU init failed — fell back to WASM: ${errorMessage(error)}`;
        accel = 'wasm';
        return loadOrtModel(bytes, accel, opts.numThreads);
      }
      throw error;
    }
  };
  const detectorEngine = await loadWithFallback(await fetchBytes(opts.detectorUrl));
  engines.push(detectorEngine);
  const alignerEngine = await loadWithFallback(await fetchBytes(opts.alignerUrl));
  engines.push(alignerEngine);
  const detector = new HeadDetector(detectorEngine, opts.headScoreThreshold);
  const aligner = new FaceAligner(alignerEngine, opts.inputNorm, opts.cropPad);
  const pipeline = new HrffaPipeline(detector, aligner);
  // ウォームアップ(Worker と同じ)
  const warm = document.createElement('canvas');
  warm.width = detector.inWidth;
  warm.height = detector.inHeight;
  const warmSource = sourceFrom(warm, warm.width, warm.height);
  await detector.detect(warmSource);
  await aligner.align(warmSource, [{ x1: 0, y1: 0, x2: warm.width - 1, y2: warm.height - 1, score: 1 }]);
  return {
    info: {
      accelerator: accel,
      note,
      detectorInput: [detector.inWidth, detector.inHeight],
      detectorFormat: detector.format,
      alignerInput: aligner.size,
      alignerBatch: aligner.dynamicBatch ? 'N' : '1',
    },
    process: (source) => pipeline.process(source),
    setParams: async (params) => {
      if (params.headScoreThreshold !== undefined) {
        detector.scoreThreshold = params.headScoreThreshold;
      }
      if (params.cropPad !== undefined) {
        aligner.cropPad = params.cropPad;
      }
    },
    dispose: () => {
      for (const engine of engines) {
        try {
          engine.dispose();
        } catch {
          // already disposed
        }
      }
    },
  };
}

// ---- 専用 Worker 実行(既定)
async function createWorkerRunner(opts: Parameters<typeof createLocalRunner>[0]): Promise<Runner> {
  const worker = new WorkerPipeline();
  try {
    const info = await worker.init(opts);
    return {
      info,
      process: (_source, grab) => worker.process(grab()),
      setParams: (params) => worker.setParams(params),
      dispose: () => worker.dispose(),
    };
  } catch (error) {
    worker.dispose();
    throw error;
  }
}

export default function App() {
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [detectorName, setDetectorName] = useState<string>('');
  const [alignerName, setAlignerName] = useState<string>('');
  const [backend, setBackend] = useState<Accelerator>(() =>
    QUERY.get('backend') === 'wasm' || !WEBGPU_OK ? 'wasm' : 'webgpu',
  );
  const [numThreads, setNumThreads] = useState<number>(1);
  const [sourceKind, setSourceKind] = useState<SourceKind>(() => (QUERY.get('image') ? 'image' : QUERY.get('video') ? 'video' : 'camera'));
  const [cameras, setCameras] = useState<CameraDevice[]>([]);
  const [cameraId, setCameraId] = useState<string>('');
  const [videoFileUrl, setVideoFileUrl] = useState<string | null>(() => QUERY.get('video'));
  const [videoFileName, setVideoFileName] = useState<string>(() => QUERY.get('video') ?? '');
  const [imageUrl, setImageUrl] = useState<string | null>(() => QUERY.get('image'));
  const [imageName, setImageName] = useState<string>(() => QUERY.get('image') ?? '');
  const [headScore, setHeadScore] = useState<number>(0.5);
  const [cropPad, setCropPad] = useState<number>(0.05);
  const [inputNorm, setInputNorm] = useState<InputNormChoice>('auto');
  const [drawBbox, setDrawBbox] = useState<boolean>(true);
  const [drawLines, setDrawLines] = useState<boolean>(false);
  const [running, setRunning] = useState<boolean>(false);
  const [status, setStatus] = useState<string>('Idle');
  const [engineInfo, setEngineInfo] = useState<string | null>(null);
  const [stats, setStats] = useState<RunStats | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const runningRef = useRef<boolean>(false);
  const streamRef = useRef<MediaStream | null>(null);
  const runnerRef = useRef<Runner | null>(null);
  // 進行中のフレーム推論。Stop はこれの完了を待ってからセッションを解放する(解放が先だと OrtRun が
  // "Buffer was unmapped before mapping was resolved" で失敗する)
  const inflightRef = useRef<Promise<FrameOutput> | null>(null);
  const drawOptsRef = useRef<DrawOptions>({ bbox: true, lines: false, pointRadius: null });
  const autostartedRef = useRef<boolean>(false);

  useEffect(() => {
    drawOptsRef.current = { bbox: drawBbox, lines: drawLines, pointRadius: null };
  }, [drawBbox, drawLines]);

  // ---- model catalog (public/models/manifest.json, staged by scripts/prepare-assets.mjs)
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const response = await fetch(new URL('models/manifest.json', document.baseURI).href, { cache: 'no-store' });
        if (!response.ok) {
          throw new Error(`${response.status} ${response.statusText}`);
        }
        const raw = (await response.json()) as unknown;
        const list: ModelEntry[] = Array.isArray(raw)
          ? raw
              .map((entry) =>
                typeof entry === 'string'
                  ? { name: entry, bytes: 0 }
                  : {
                      name: String((entry as { name?: unknown }).name ?? ''),
                      bytes: Number((entry as { bytes?: unknown }).bytes ?? 0),
                    },
              )
              .filter((entry) => entry.name.endsWith('.onnx'))
              .map((entry) => ({ ...entry, url: `models/${entry.name}` }))
          : [];
        if (!cancelled) {
          setModels(list);
        }
      } catch (error) {
        if (!cancelled) {
          setCatalogError(`Failed to load models/manifest.json (run \`pnpm run prepare:assets\`): ${errorMessage(error)}`);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // 既定モデル(?detector= / ?aligner= で上書き可)
  useEffect(() => {
    const detectors = models.filter(isDetector);
    const aligners = models.filter(isAligner);
    const wantedDetector = QUERY.get('detector') ?? DEFAULT_DETECTOR;
    const wantedAligner = QUERY.get('aligner') ?? DEFAULT_ALIGNER;
    setDetectorName((current) => (current && detectors.some((m) => m.name === current) ? current : pickDefault(detectors, wantedDetector)));
    setAlignerName((current) => (current && aligners.some((m) => m.name === current) ? current : pickDefault(aligners, wantedAligner)));
  }, [models]);

  useEffect(() => {
    listCameras()
      .then(setCameras)
      .catch(() => setCameras([]));
  }, []);

  // ---- stop: runner (worker / engines), camera stream and video element
  // byUser: Stop ボタンから呼ばれたときだけステータスを 'Stopped' にする(エラーや画像 1 枚完了時の文言は残す)
  const stop = useCallback((byUser = false) => {
    const wasRunning = runningRef.current;
    runningRef.current = false;
    setRunning(false);
    if (byUser && wasRunning) {
      setStatus('Stopped');
    }
    const runner = runnerRef.current;
    runnerRef.current = null;
    const inflight = inflightRef.current;
    void (async () => {
      try {
        await inflight;
      } catch {
        // 停止要求後の失敗は無視する
      }
      await runner?.dispose();
    })();
    stopStream(streamRef.current);
    streamRef.current = null;
    const video = videoRef.current;
    if (video) {
      video.pause();
      video.srcObject = null;
      video.removeAttribute('src');
      video.load();
    }
  }, []);

  useEffect(() => () => stop(), [stop]);

  // live parameter updates while running
  useEffect(() => {
    const runner = runnerRef.current;
    if (runner && runningRef.current) {
      void runner.setParams({ headScoreThreshold: headScore, cropPad }).catch(() => undefined);
    }
  }, [headScore, cropPad]);

  // ---- start
  const start = useCallback(async () => {
    if (runningRef.current) {
      return;
    }
    const detector = models.find((m) => m.name === detectorName);
    const aligner = models.find((m) => m.name === alignerName);
    if (!detector || !aligner) {
      setStatus('Select a detector (deimv2_*.onnx) and an alignment model (hrffa_*.onnx).');
      return;
    }
    setRunning(true);
    runningRef.current = true;
    setStats(null);
    setEngineInfo(null);

    const modeLabel = WORKER_MODE === 'dedicated' ? 'worker' : 'main thread';
    try {
      setStatus(`Loading models (${backend}, ${modeLabel}) ...`);
      const norm: InputNorm = inputNorm === 'auto' ? inferInputNorm(aligner.name) : inputNorm;
      const opts = {
        accelerator: backend,
        numThreads,
        detectorUrl: detector.url,
        alignerUrl: aligner.url,
        headScoreThreshold: headScore,
        cropPad,
        inputNorm: norm,
      };
      const runner = WORKER_MODE === 'dedicated' ? await createWorkerRunner(opts) : await createLocalRunner(opts);
      if (!runningRef.current) {
        runner.dispose();
        return;
      }
      runnerRef.current = runner;
      const ready = runner.info;
      setEngineInfo(
        `${ready.accelerator} / ${modeLabel} · detector ${ready.detectorInput[0]}x${ready.detectorInput[1]} (${ready.detectorFormat}) · ` +
          `aligner ${ready.alignerInput}x${ready.alignerInput} (batch ${ready.alignerBatch}, ${norm})`,
      );
      const noteSuffix = ready.note ? ` — ${ready.note}` : '';

      const canvas = canvasRef.current;
      if (!canvas) {
        throw new Error('canvas element unavailable');
      }
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        throw new Error('2D canvas context unavailable');
      }
      // full-resolution frame grabber for the worker path
      const grab = document.createElement('canvas');
      const grabCtx = grab.getContext('2d', { willReadFrequently: true });
      if (!grabCtx) {
        throw new Error('2D canvas context unavailable');
      }

      // ---- still image: run once (the second run is reported — the first one still compiles shaders)
      if (sourceKind === 'image') {
        if (!imageUrl) {
          throw new Error('Select an image file.');
        }
        setStatus(`Loading image ...${noteSuffix}`);
        const bitmap = await loadImageBitmap(imageUrl);
        canvas.width = bitmap.width;
        canvas.height = bitmap.height;
        grab.width = bitmap.width;
        grab.height = bitmap.height;
        const source = sourceFrom(bitmap, bitmap.width, bitmap.height);
        const grabImage = (): ImageData => {
          grabCtx.drawImage(bitmap, 0, 0);
          return grabCtx.getImageData(0, 0, bitmap.width, bitmap.height);
        };
        setStatus(`Running on the image (${ready.accelerator}, ${modeLabel}) ...${noteSuffix}`);
        inflightRef.current = runner.process(source, grabImage);
        await inflightRef.current;
        const t0 = performance.now();
        inflightRef.current = runner.process(source, grabImage);
        const output = await inflightRef.current;
        inflightRef.current = null;
        const totalMs = performance.now() - t0;
        ctx.drawImage(bitmap, 0, 0);
        drawHeads(ctx, output.heads, drawOptsRef.current);
        bitmap.close();
        publishResult(output);
        setStats({
          fps: 0,
          totalMs,
          detectMs: output.stats.detectMs,
          alignMs: output.stats.alignMs,
          nHeads: output.stats.nHeads,
          frame: 1,
        });
        setStatus(
          `Done — ${output.stats.nHeads} head(s), detect ${output.stats.detectMs.toFixed(1)} ms, ` +
            `align ${output.stats.alignMs.toFixed(1)} ms (${ready.accelerator}, ${modeLabel}, second run). Press Start to run again.${noteSuffix}`,
        );
        stop();
        return;
      }

      // ---- camera / video file
      const video = videoRef.current;
      if (!video) {
        throw new Error('video element unavailable');
      }
      if (sourceKind === 'camera') {
        setStatus(`Starting camera ...${noteSuffix}`);
        const stream = await openCamera(cameraId || null);
        streamRef.current = stream;
        video.srcObject = stream;
        video.loop = false;
      } else {
        if (!videoFileUrl) {
          throw new Error('Select a video file.');
        }
        video.srcObject = null;
        video.src = videoFileUrl;
        video.loop = true;
      }
      video.muted = true;
      await video.play();
      const frameW = video.videoWidth;
      const frameH = video.videoHeight;
      if (!frameW || !frameH) {
        throw new Error('Could not determine the source resolution.');
      }
      canvas.width = frameW;
      canvas.height = frameH;
      grab.width = frameW;
      grab.height = frameH;
      const source = sourceFrom(video, frameW, frameH);
      const grabFrame = (): ImageData => {
        grabCtx.drawImage(video, 0, 0, frameW, frameH);
        return grabCtx.getImageData(0, 0, frameW, frameH);
      };
      setStatus(`Running — ${ready.accelerator} / ${modeLabel}${noteSuffix}`);

      let emaFps = 0;
      let lastT = performance.now();
      let frameNo = 0;
      const loop = async (): Promise<void> => {
        while (runningRef.current) {
          if (video.paused || video.ended) {
            await sleep(50);
            continue;
          }
          const pending = runner.process(source, grabFrame);
          inflightRef.current = pending;
          let output: FrameOutput;
          try {
            output = await pending;
          } finally {
            inflightRef.current = null;
          }
          if (!runningRef.current) {
            break;
          }
          frameNo += 1;
          const now = performance.now();
          const fps = 1000 / Math.max(now - lastT, 1);
          lastT = now;
          emaFps = emaFps === 0 ? fps : 0.9 * emaFps + 0.1 * fps;
          ctx.drawImage(video, 0, 0, frameW, frameH);
          drawHeads(ctx, output.heads, drawOptsRef.current);
          publishResult(output);
          setStats({
            fps: emaFps,
            totalMs: 1000 / Math.max(emaFps, 1e-3),
            detectMs: output.stats.detectMs,
            alignMs: output.stats.alignMs,
            nHeads: output.stats.nHeads,
            frame: frameNo,
          });
          // UI スレッドへ制御を戻す
          await sleep(0);
        }
      };
      void loop().catch((error) => {
        if (runningRef.current) {
          setStatus(`Runtime error: ${errorMessage(error)}`);
          stop();
        }
      });
    } catch (error) {
      setStatus(`Error: ${errorMessage(error)}`);
      stop();
    }
  }, [alignerName, backend, cameraId, cropPad, detectorName, headScore, imageUrl, inputNorm, models, numThreads, sourceKind, stop, videoFileUrl]);

  useEffect(() => {
    if (QUERY.get('autostart') === '1' && !autostartedRef.current && detectorName && alignerName) {
      autostartedRef.current = true;
      void start();
    }
  }, [alignerName, detectorName, start]);

  const detectors = models.filter(isDetector);
  const aligners = models.filter(isAligner);

  return (
    <div className="layout">
      <section className="card controls-card">
        <h1>HRFFA Web Demo</h1>
        <p className="subtle">
          Head detection (DEIMv2-Wholebody49 / YOLOv9-Wholebody34) + HRFFA face alignment on head crops, entirely in
          the browser with onnxruntime-web ({WEBGPU_OK ? 'WebGPU / WASM' : 'WASM only — WebGPU unavailable'}).
        </p>
        {catalogError ? <p className="status">{catalogError}</p> : null}
        <div className="control-grid">
          <label>
            Detector (DEIMv2-Wholebody49 / YOLOv9-Wholebody34, class 7 = head)
            <select value={detectorName} onChange={(e) => setDetectorName(e.target.value)} disabled={running}>
              {detectors.map((m) => (
                <option key={m.name} value={m.name}>
                  {m.name}
                  {formatMb(m.bytes)}
                </option>
              ))}
            </select>
          </label>
          <label>
            Alignment model (HRFFA)
            <select value={alignerName} onChange={(e) => setAlignerName(e.target.value)} disabled={running}>
              {aligners.map((m) => (
                <option key={m.name} value={m.name}>
                  {m.name}
                  {formatMb(m.bytes)}
                </option>
              ))}
            </select>
          </label>
          <label>
            Backend
            <select value={backend} onChange={(e) => setBackend(e.target.value as Accelerator)} disabled={running}>
              <option value="webgpu" disabled={!WEBGPU_OK}>
                WebGPU{WEBGPU_OK ? '' : ' (unavailable)'}
              </option>
              <option value="wasm">WASM</option>
            </select>
          </label>
          <label>
            WASM threads (1 = single-threaded; more needs cross-origin isolation)
            <input
              type="number"
              min={1}
              max={16}
              value={numThreads}
              onChange={(e) => setNumThreads(Math.max(1, Math.floor(Number(e.target.value) || 1)))}
              disabled={running}
            />
          </label>
          <label>
            Input normalization of the alignment model
            <select value={inputNorm} onChange={(e) => setInputNorm(e.target.value as InputNormChoice)} disabled={running}>
              <option value="auto">auto (vitl → ImageNet, others → center05)</option>
              <option value="center05">center05: (x/255 − 0.5) / 0.5</option>
              <option value="imagenet">ImageNet mean / std</option>
            </select>
          </label>
          <label>
            Head score threshold: {headScore.toFixed(2)}
            <input type="range" min={0.1} max={0.9} step={0.05} value={headScore} onChange={(e) => setHeadScore(Number(e.target.value))} />
          </label>
          <label>
            Crop pad (training value 0.05): {cropPad.toFixed(2)}
            <input type="range" min={0} max={0.2} step={0.01} value={cropPad} onChange={(e) => setCropPad(Number(e.target.value))} />
          </label>
          <label>
            Source
            <select value={sourceKind} onChange={(e) => setSourceKind(e.target.value as SourceKind)} disabled={running}>
              <option value="camera">Camera (VGA)</option>
              <option value="video">Video file</option>
              <option value="image">Image file</option>
            </select>
          </label>
          {sourceKind === 'camera' ? (
            <label>
              Camera device
              <select value={cameraId} onChange={(e) => setCameraId(e.target.value)} disabled={running}>
                <option value="">Default</option>
                {cameras.map((c) => (
                  <option key={c.deviceId} value={c.deviceId}>
                    {c.label}
                  </option>
                ))}
              </select>
            </label>
          ) : null}
          {sourceKind === 'video' ? (
            <label>
              Video file{videoFileName ? `: ${videoFileName}` : ''}
              <input
                type="file"
                accept="video/*"
                disabled={running}
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) {
                    if (videoFileUrl && videoFileUrl.startsWith('blob:')) {
                      URL.revokeObjectURL(videoFileUrl);
                    }
                    setVideoFileUrl(URL.createObjectURL(file));
                    setVideoFileName(file.name);
                  }
                }}
              />
            </label>
          ) : null}
          {sourceKind === 'image' ? (
            <label>
              Image file{imageName ? `: ${imageName}` : ''}
              <input
                type="file"
                accept="image/*"
                disabled={running}
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) {
                    if (imageUrl && imageUrl.startsWith('blob:')) {
                      URL.revokeObjectURL(imageUrl);
                    }
                    setImageUrl(URL.createObjectURL(file));
                    setImageName(file.name);
                  }
                }}
              />
            </label>
          ) : null}
          <label className="inline">
            <input type="checkbox" checked={drawBbox} onChange={(e) => setDrawBbox(e.target.checked)} />
            Draw head boxes
          </label>
          <label className="inline">
            <input type="checkbox" checked={drawLines} onChange={(e) => setDrawLines(e.target.checked)} />
            Draw ibug68 contour lines
          </label>
        </div>
        <div className="buttons">
          <button type="button" onClick={() => void start()} disabled={running || !detectorName || !alignerName}>
            Start
          </button>
          <button type="button" className="stop" onClick={() => stop(true)} disabled={!running}>
            Stop
          </button>
        </div>
        <div className="status">{status}</div>
        {engineInfo ? <div className="stats">{engineInfo}</div> : null}
        {stats ? (
          <div className="stats">
            {stats.fps > 0 ? (
              <div>
                fps <b>{stats.fps.toFixed(1)}</b> · frame <b>{stats.totalMs.toFixed(1)}</b> ms
              </div>
            ) : (
              <div>
                total <b>{stats.totalMs.toFixed(1)}</b> ms
              </div>
            )}
            <div>
              detect <b>{stats.detectMs.toFixed(1)}</b> ms · align <b>{stats.alignMs.toFixed(1)}</b> ms
            </div>
            <div>
              heads <b>{stats.nHeads}</b> · frames <b>{stats.frame}</b>
            </div>
          </div>
        ) : null}
      </section>
      <section className="card view-card">
        <canvas ref={canvasRef} className="view" />
        <video ref={videoRef} className="hidden-media" playsInline />
      </section>
    </div>
  );
}
