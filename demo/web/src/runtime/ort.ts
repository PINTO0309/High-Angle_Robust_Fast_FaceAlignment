// onnxruntime-web: WebGPU(JSEP)または WASM 実行プロバイダで .onnx を動かす。

// the /webgpu bundle registers both the webgpu (JSEP) and wasm backends
import * as ort from 'onnxruntime-web/webgpu';
import { assetUrl, type Accelerator, type OrtModel, type TensorIn, type TensorOut } from './engine';

let envConfigured = false;

function configureEnv(numThreads: number): void {
  if (envConfigured) {
    return;
  }
  // 絶対 URL: ort は wasmPaths の相対パスを文書ではなくバンドル自身の import.meta.url から解決する
  ort.env.wasm.wasmPaths = assetUrl('wasm/ort/');
  // 既定は単一スレッド(参照実装で検証済みの構成)。COOP/COEP 配信下では UI から増やせる
  ort.env.wasm.numThreads = numThreads > 0 ? numThreads : 1;
  ort.env.logLevel = 'error';
  envConfigured = true;
}

interface ValueMetadataLike {
  name?: string;
  type?: unknown;
  shape?: ReadonlyArray<number | string>;
}

export async function loadOrtModel(
  bytes: Uint8Array,
  accelerator: Accelerator,
  numThreads: number,
): Promise<OrtModel> {
  if (accelerator === 'webgpu' && !('gpu' in navigator)) {
    throw new Error('WebGPU is not available in this browser.');
  }
  configureEnv(numThreads);

  const session = await ort.InferenceSession.create(bytes, {
    executionProviders: accelerator === 'webgpu' ? ['webgpu'] : ['wasm'],
    graphOptimizationLevel: 'all',
  });

  const inputMeta = (session as unknown as { inputMetadata?: ReadonlyArray<ValueMetadataLike> })
    .inputMetadata?.[0];
  const shape = inputMeta?.shape ?? [];
  if (shape.length !== 4) {
    void session.release();
    throw new Error(`Expected a 4-D image input, got shape [${shape.join(', ')}]`);
  }
  if (inputMeta?.type !== undefined && inputMeta.type !== 'float32') {
    void session.release();
    throw new Error(`Model input dtype is ${String(inputMeta.type)} — float32 models are expected.`);
  }
  const inputDims = shape.map((d) => (typeof d === 'number' && d > 0 ? d : -1));

  const run = async (feeds: Record<string, TensorIn>): Promise<Record<string, TensorOut>> => {
    const tensors: Record<string, ort.Tensor> = {};
    for (const [name, t] of Object.entries(feeds)) {
      tensors[name] = new ort.Tensor('float32', t.data, t.dims);
    }
    const results = await session.run(tensors);
    const out: Record<string, TensorOut> = {};
    for (const name of session.outputNames) {
      const r = results[name];
      const data = r.data;
      out[name] = {
        data:
          data instanceof Float32Array
            ? data.slice()
            : Float32Array.from(data as unknown as ArrayLike<number>),
        dims: Array.from(r.dims, (d) => Number(d)),
      };
    }
    return out;
  };

  return {
    accelerator,
    inputNames: [...session.inputNames],
    inputDims,
    outputNames: [...session.outputNames],
    run,
    dispose(): void {
      void session.release();
    },
  };
}
