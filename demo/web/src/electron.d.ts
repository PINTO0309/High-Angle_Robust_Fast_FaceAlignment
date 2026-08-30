export {};

declare global {
  interface Window {
    electronAPI?: {
      versions: Record<string, string | undefined>;
    };
  }
}
