import { contextBridge } from 'electron';

contextBridge.exposeInMainWorld('electronAPI', {
  versions: process.versions,
});
